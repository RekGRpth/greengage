/*-------------------------------------------------------------------------
 *
 * nodeSort.c
 *	  Routines to handle sorting of relations.
 *
 * Portions Copyright (c) 2007-2008, Greenplum inc
 * Portions Copyright (c) 2012-Present VMware, Inc. or its affiliates.
 * Portions Copyright (c) 1996-2019, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 *
 * IDENTIFICATION
 *	  src/backend/executor/nodeSort.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "access/parallel.h"
#include "executor/execdebug.h"
#include "executor/nodeSort.h"
#include "lib/stringinfo.h"             /* StringInfo */
#include "miscadmin.h"
#include "utils/tuplesort.h"
#include "cdb/cdbvars.h" /* CDB *//* gp_sort_flags */
#include "utils/workfile_mgr.h"
#include "executor/instrument.h"
#include "executor/vectorBatch.h"
#include "utils/datum.h"
#include "utils/faultinjector.h"
#include "utils/sortsupport.h"

static void ExecSortExplainEnd(PlanState *planstate, struct StringInfoData *buf);
static void ExecEagerFreeSort(SortState *node);

/* ----------------------------------------------------------------
 *		ExecSort
 *
 *		Sorts tuples from the outer subtree of the node using tuplesort,
 *		which saves the results in a temporary file or memory. After the
 *		initial call, returns a tuple from the file with each call.
 *
 *		Conditions:
 *		  -- none.
 *
 *		Initial States:
 *		  -- the outer child is prepared to return the first tuple.
 * ----------------------------------------------------------------
 */
static TupleTableSlot *
ExecSort(PlanState *pstate)
{
	SortState  *node = castNode(SortState, pstate);
	EState	   *estate;
	ScanDirection dir;
	Tuplesortstate *tuplesortstate;
	TupleTableSlot *slot = NULL;
	Sort 		*plannode = NULL;
	PlanState  *outerNode = NULL;
	TupleDesc	tupDesc = NULL;

	CHECK_FOR_INTERRUPTS();

	SIMPLE_FAULT_INJECTOR("explain_analyze_sort_error");
	/*
	 * get state info from node
	 */
	SO1_printf("ExecSort: %s\n",
			   "entering routine");

	estate = node->ss.ps.state;
	dir = estate->es_direction;
	tuplesortstate = (Tuplesortstate *) node->tuplesortstate;

	/*
	 * In Window node, we might need to call ExecSort again even when
	 * the last tuple in the Sort has been retrieved. Since we might
	 * eager free the tuplestore, the tuplestorestate could be NULL.
	 * We simply return NULL in this case.
	 */
	if (node->sort_Done && tuplesortstate == NULL)
	{
		return NULL;
	}

	plannode = (Sort *) node->ss.ps.plan;


	/*
	 * If called for the first time, initialize tuplesort_state
	 */

	if (!node->sort_Done)
	{
		SO1_printf("ExecSort: %s\n",
				   "sorting subplan");

		/*
		 * Want to scan subplan in the forward direction while creating the
		 * sorted data.
		 */
		estate->es_direction = ForwardScanDirection;

		/*
		 * Initialize tuplesort module.
		 */
		SO1_printf("ExecSort: %s\n",
				   "calling tuplesort_begin");

		outerNode = outerPlanState(node);
		tupDesc = ExecGetResultType(outerNode);

		tuplesortstate = tuplesort_begin_heap(//&node->ss,
											  tupDesc,
											  plannode->numCols,
											  plannode->sortColIdx,
											  plannode->sortOperators,
											  plannode->collations,
											  plannode->nullsFirst,
											  PlanStateOperatorMemKB((PlanState *) node),
											  NULL,
											  node->randomAccess);

		if (node->bounded)
			tuplesort_set_bound(tuplesortstate, node->bound);
		node->tuplesortstate = (void *) tuplesortstate;

		/* CDB */

		/* If EXPLAIN ANALYZE, share our Instrumentation object with sort. */
		if (node->ss.ps.instrument && node->ss.ps.instrument->need_cdb)
			tuplesort_set_instrument(tuplesortstate,
									 node->ss.ps.instrument,
									 node->ss.ps.cdbexplainbuf);
	}

	/*
	 * If first time through,
	 * read all tuples from outer plan and pass them to
	 * tuplesort.c. Subsequent calls just fetch tuples from tuplesort.
	 */
	if (!node->sort_Done)
	{

		Assert(outerNode != NULL);

		/*
		 * Scan the subplan and feed all the tuples to tuplesort.
		 */

		for (;;)
		{
			slot = ExecProcNode(outerNode);

			if (TupIsNull(slot))
				break;

			tuplesort_puttupleslot(tuplesortstate, slot);
		}

		SIMPLE_FAULT_INJECTOR("execsort_before_sorting");

		/*
		 * Complete the sort.
		 */
		tuplesort_performsort(tuplesortstate);

		/*
		 * restore to user specified direction
		 */
		estate->es_direction = dir;

		/*
		 * finally set the sorted flag to true
		 */
		node->sort_Done = true;
		node->bounded_Done = node->bounded;
		node->bound_Done = node->bound;
		if (node->shared_info && node->am_worker)
		{
			TuplesortInstrumentation *si;

			Assert(IsParallelWorker());
			Assert(ParallelWorkerNumber <= node->shared_info->num_workers);
			si = &node->shared_info->sinstrument[ParallelWorkerNumber];
			tuplesort_get_stats(tuplesortstate, si);
		}
		SO1_printf("ExecSort: %s\n", "sorting done");
	}

	SO1_printf("ExecSort: %s\n",
			   "retrieving tuple from tuplesort");

	/*
	 * Get the first or next tuple from tuplesort. Returns NULL if no more
	 * tuples.  Note that we only rely on slot tuple remaining valid until the
	 * next fetch from the tuplesort.
	 */
	slot = node->ss.ps.ps_ResultTupleSlot;
	(void) tuplesort_gettupleslot(tuplesortstate,
								  ScanDirectionIsForward(dir),
								  false, slot, NULL);

	if (TupIsNull(slot) && !node->delayEagerFree)
	{
		ExecEagerFreeSort(node);
	}

	return slot;
}

/* ----------------------------------------------------------------
 *		ExecInitSort
 *
 *		Creates the run-time state information for the sort node
 *		produced by the planner and initializes its outer subtree.
 * ----------------------------------------------------------------
 */
SortState *
ExecInitSort(Sort *node, EState *estate, int eflags)
{
	SortState  *sortstate;

	SO1_printf("ExecInitSort: %s\n",
			   "initializing sort node");

	/*
	 * GPDB
	 */
#ifdef FAULT_INJECTOR
	if (SIMPLE_FAULT_INJECTOR("rg_qmem_qd_qe") == FaultInjectorTypeSkip)
	{
		elog(NOTICE, "op_mem=%d", (int) (((Plan *) node)->operatorMemKB));
	}
#endif

	/*
	 * create state structure
	 */
	sortstate = makeNode(SortState);
	sortstate->ss.ps.plan = (Plan *) node;
	sortstate->ss.ps.state = estate;
	sortstate->ss.ps.ExecProcNode = ExecSort;

	/*
	 * We must have random access to the sort output to do backward scan or
	 * mark/restore.  We also prefer to materialize the sort output if we
	 * might be called on to rewind and replay it many times.
	 */
	sortstate->randomAccess = (eflags & (EXEC_FLAG_REWIND |
										 EXEC_FLAG_BACKWARD |
										 EXEC_FLAG_MARK)) != 0;

	sortstate->bounded = false;
	sortstate->sort_Done = false;
	sortstate->tuplesortstate = NULL;

	/* CDB */

	/* BUT:
	 * The LIMIT optimizations requires exprcontext in which to
	 * evaluate the limit/offset parameters.
	 */
	ExecAssignExprContext(estate, &sortstate->ss.ps);

	/*
	 * Miscellaneous initialization
	 *
	 * Sort nodes don't initialize their ExprContexts because they never call
	 * ExecQual or ExecProject.
	 */

	/*
	 * CDB: Offer extra info for EXPLAIN ANALYZE.
	 */
	if (estate->es_instrument && (estate->es_instrument & INSTRUMENT_CDB))
	{
		/* Allocate string buffer. */
		sortstate->ss.ps.cdbexplainbuf = makeStringInfo();

		/* Request a callback at end of query. */
		sortstate->ss.ps.cdbexplainfun = ExecSortExplainEnd;
	}

	/*
	 * If eflag contains EXEC_FLAG_REWIND or EXEC_FLAG_BACKWARD or EXEC_FLAG_MARK,
	 * then this node is not eager free safe.
	 */
	sortstate->delayEagerFree =
		((eflags & (EXEC_FLAG_REWIND | EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK)) != 0);

	/*
	 * initialize child nodes
	 *
	 * We shield the child node from the need to support BACKWARD, or
	 * MARK/RESTORE.
	 */

	eflags &= ~(EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK);

	/*
	 * If Sort does not have any external parameters, then it
	 * can shield the child node from being rescanned as well, hence
	 * we can clear the EXEC_FLAG_REWIND as well. If there are parameters,
	 * don't clear the REWIND flag, as the child will be rewound.
	 */

	if (node->plan.allParam == NULL || node->plan.extParam == NULL)
	{
		eflags &= ~EXEC_FLAG_REWIND;
	}

	outerPlanState(sortstate) = ExecInitNode(outerPlan(node), estate, eflags);

	/*
	 * If the child node of a Material is a Motion, then this Material node is
	 * not eager free safe.
	 */
	if (IsA(outerPlan((Plan *)node), Motion))
	{
		sortstate->delayEagerFree = true;
	}

	/*
	 * Initialize scan slot and type.
	 */
	ExecCreateScanSlotFromOuterPlan(estate, &sortstate->ss, &TTSOpsVirtual);

	/*
	 * Initialize return slot and type. No need to initialize projection info
	 * because this node doesn't do projections.
	 */
	ExecInitResultTupleSlotTL(&sortstate->ss.ps, &TTSOpsMinimalTuple);
	sortstate->ss.ps.ps_ProjInfo = NULL;

	SO1_printf("ExecInitSort: %s\n",
			   "sort node initialized");

	return sortstate;
}

/* ----------------------------------------------------------------
 *		ExecEndSort(node)
 * ----------------------------------------------------------------
 */
void
ExecEndSort(SortState *node)
{
	SO1_printf("ExecEndSort: %s\n",
			   "shutting down sort node");

	ExecEagerFreeSort(node);

	/*
	 * shut down the subplan
	 */
	ExecEndNode(outerPlanState(node));

	SO1_printf("ExecEndSort: %s\n",
			   "sort node shutdown");
}

/* ----------------------------------------------------------------
 *		ExecSortMarkPos
 *
 *		Calls tuplesort to save the current position in the sorted file.
 * ----------------------------------------------------------------
 */
void
ExecSortMarkPos(SortState *node)
{
	/*
	 * if we haven't sorted yet, just return
	 */
	if (!node->sort_Done)
		return;

	tuplesort_markpos((Tuplesortstate *) node->tuplesortstate);
}

/* ----------------------------------------------------------------
 *		ExecSortRestrPos
 *
 *		Calls tuplesort to restore the last saved sort file position.
 * ----------------------------------------------------------------
 */
void
ExecSortRestrPos(SortState *node)
{
	/*
	 * if we haven't sorted yet, just return.
	 */
	if (!node->sort_Done)
		return;

	/*
	 * restore the scan to the previously marked position
	 */
	tuplesort_restorepos((Tuplesortstate *) node->tuplesortstate);
}

void
ExecReScanSort(SortState *node)
{
	PlanState  *outerPlan = outerPlanState(node);

	/*
	 * If we haven't sorted yet, just return. If outerplan's chgParam is not
	 * NULL then it will be re-scanned by ExecProcNode, else no reason to
	 * re-scan it at all.
	 */
	if (!node->sort_Done)
		return;

	VectorSortReset(node);

	/* must drop pointer to sort result tuple */
	ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);

	/*
	 * If subnode is to be rescanned then we forget previous sort results; we
	 * have to re-read the subplan and re-sort.  Also must re-sort if the
	 * bounded-sort parameters changed or we didn't select randomAccess.
	 *
	 * Otherwise we can just rewind and rescan the sorted output.
	 */
	if (outerPlan->chgParam != NULL ||
		node->bounded != node->bounded_Done ||
		node->bound != node->bound_Done ||
		!node->randomAccess ||
		(node->tuplesortstate == NULL))
	{
		node->sort_Done = false;

		if (node->tuplesortstate != NULL)
		{
			tuplesort_end((Tuplesortstate *) node->tuplesortstate);
			node->tuplesortstate = NULL;
		}

		/*
		 * if chgParam of subnode is not null then plan will be re-scanned by
		 * first ExecProcNode.
		 */
		if (outerPlan->chgParam == NULL)
			ExecReScan(outerPlan);
	}
	else
		tuplesort_rescan((Tuplesortstate *) node->tuplesortstate);
}


/*
 * ExecSortExplainEnd
 *      Called before ExecutorEnd to finish EXPLAIN ANALYZE reporting.
 */
void
ExecSortExplainEnd(PlanState *planstate, struct StringInfoData *buf)
{
	SortState *sortstate = (SortState *) planstate;

	if (sortstate->tuplesortstate)
	{
		tuplesort_finalize_stats(sortstate->tuplesortstate,
								 &sortstate->sortstats);

		if (planstate->instrument)
		{
			planstate->instrument->workfileCreated = (sortstate->sortstats.spaceType == SORT_SPACE_TYPE_DISK);
			planstate->instrument->workmemused = sortstate->sortstats.workmemused;
			planstate->instrument->execmemused = sortstate->sortstats.execmemused;
		}
	}
}                               /* ExecSortExplainEnd */

static void
ExecEagerFreeSort(SortState *node)
{
	/* clean out the tuple table */
	ExecClearTuple(node->ss.ss_ScanTupleSlot);

	/* must drop pointer to sort result tuple */
	ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);

	if (node->vector_sort_state != NULL)
		VectorSortDestroy(node);

	if (node->tuplesortstate != NULL)
	{
		/*
		 * Save stats like in ExecSortExplainEnd, so that we can display
		 * them later in EXPLAIN ANALYZE.
		 */
		tuplesort_finalize_stats(node->tuplesortstate,
								 &node->sortstats);
		if (node->ss.ps.instrument)
		{
			node->ss.ps.instrument->workfileCreated = (node->sortstats.spaceType == SORT_SPACE_TYPE_DISK);
			node->ss.ps.instrument->workmemused = node->sortstats.workmemused;
			node->ss.ps.instrument->execmemused = node->sortstats.execmemused;
		}

		tuplesort_end((Tuplesortstate *) node->tuplesortstate);
		node->tuplesortstate = NULL;
	}
}

void
ExecSquelchSort(SortState *node)
{
	if (!node->delayEagerFree)
	{
		ExecEagerFreeSort(node);
		ExecSquelchNode(outerPlanState(node));
	}
}

/* ----------------------------------------------------------------
 *						Parallel Query Support
 * ----------------------------------------------------------------
 */

/* ----------------------------------------------------------------
 *		ExecSortEstimate
 *
 *		Estimate space required to propagate sort statistics.
 * ----------------------------------------------------------------
 */
void
ExecSortEstimate(SortState *node, ParallelContext *pcxt)
{
	Size		size;

	/* don't need this if not instrumenting or no workers */
	if (!node->ss.ps.instrument || pcxt->nworkers == 0)
		return;

	size = mul_size(pcxt->nworkers, sizeof(TuplesortInstrumentation));
	size = add_size(size, offsetof(SharedSortInfo, sinstrument));
	shm_toc_estimate_chunk(&pcxt->estimator, size);
	shm_toc_estimate_keys(&pcxt->estimator, 1);
}

/* ----------------------------------------------------------------
 *		ExecSortInitializeDSM
 *
 *		Initialize DSM space for sort statistics.
 * ----------------------------------------------------------------
 */
void
ExecSortInitializeDSM(SortState *node, ParallelContext *pcxt)
{
	Size		size;

	/* don't need this if not instrumenting or no workers */
	if (!node->ss.ps.instrument || pcxt->nworkers == 0)
		return;

	size = offsetof(SharedSortInfo, sinstrument)
		+ pcxt->nworkers * sizeof(TuplesortInstrumentation);
	node->shared_info = shm_toc_allocate(pcxt->toc, size);
	/* ensure any unfilled slots will contain zeroes */
	memset(node->shared_info, 0, size);
	node->shared_info->num_workers = pcxt->nworkers;
	shm_toc_insert(pcxt->toc, node->ss.ps.plan->plan_node_id,
				   node->shared_info);
}

/* ----------------------------------------------------------------
 *		ExecSortInitializeWorker
 *
 *		Attach worker to DSM space for sort statistics.
 * ----------------------------------------------------------------
 */
void
ExecSortInitializeWorker(SortState *node, ParallelWorkerContext *pwcxt)
{
	node->shared_info =
		shm_toc_lookup(pwcxt->toc, node->ss.ps.plan->plan_node_id, true);
	node->am_worker = true;
}

/* ----------------------------------------------------------------
 *		ExecSortRetrieveInstrumentation
 *
 *		Transfer sort statistics from DSM to private memory.
 * ----------------------------------------------------------------
 */
void
ExecSortRetrieveInstrumentation(SortState *node)
{
	Size		size;
	SharedSortInfo *si;

	if (node->shared_info == NULL)
		return;

	size = offsetof(SharedSortInfo, sinstrument)
		+ node->shared_info->num_workers * sizeof(TuplesortInstrumentation);
	si = palloc(size);
	memcpy(si, node->shared_info, size);
	node->shared_info = si;
}

bool
VectorSortPlanIsSupported(Sort *sort)
{
	return sort->numCols > 0 && sort->plan.qual == NIL &&
		sort->plan.initPlan == NIL && sort->plan.extParam == NULL &&
		sort->plan.allParam == NULL;
}

typedef struct VectorSortColumn
{
	Datum	   *values;
	uint64	   *nulls;
	int16		typlen;
	bool		typbyval;
} VectorSortColumn;

typedef struct VectorSortState
{
	SortState  *node;
	VectorSortColumn *columns;
	SortSupport sort_keys;
	MemoryContext values_context;
	int32	   *order;
	int			ncols;
	int64		nrows;
	int64		capacity;
	int64		position;
	int64		mem_limit;
	bool		native;
} VectorSortState;

static Tuplesortstate *
VectorSortBegin(SortState *node)
{
	Sort	   *plannode = (Sort *) node->ss.ps.plan;
	Tuplesortstate *tuplesortstate;

	if (node->tuplesortstate != NULL)
		return (Tuplesortstate *) node->tuplesortstate;

	node->ss.ps.state->es_direction = ForwardScanDirection;
	tuplesortstate = tuplesort_begin_heap(ExecGetResultType(outerPlanState(node)),
										 plannode->numCols,
										 plannode->sortColIdx,
										 plannode->sortOperators,
										 plannode->collations,
										 plannode->nullsFirst,
										 PlanStateOperatorMemKB((PlanState *) node),
										 NULL, node->randomAccess);
	if (node->bounded)
		tuplesort_set_bound(tuplesortstate, node->bound);
	node->tuplesortstate = tuplesortstate;
	if (node->ss.ps.instrument && node->ss.ps.instrument->need_cdb)
		tuplesort_set_instrument(tuplesortstate, node->ss.ps.instrument,
								 node->ss.ps.cdbexplainbuf);
	return tuplesortstate;
}

void
VectorSortPutTuple(SortState *node, TupleTableSlot *slot)
{
	Assert(!node->sort_Done);
	tuplesort_puttupleslot(VectorSortBegin(node), slot);
}

static VectorSortState *
VectorSortCreate(SortState *node)
{
	EState	   *estate = node->ss.ps.state;
	Sort	   *sort = (Sort *) node->ss.ps.plan;
	TupleDesc	tupdesc = ExecGetResultType(outerPlanState(node));
	VectorSortState *state;
	MemoryContext oldcontext;
	int			col;

	oldcontext = MemoryContextSwitchTo(estate->es_query_cxt);
	state = palloc0(sizeof(VectorSortState));
	state->node = node;
	state->ncols = tupdesc->natts;
	state->columns = palloc0(sizeof(VectorSortColumn) * Max(state->ncols, 1));
	state->sort_keys = palloc0(sizeof(SortSupportData) * sort->numCols);
	state->mem_limit = (int64) PlanStateOperatorMemKB((PlanState *) node) * 1024;
	state->native = state->mem_limit > 0;
	state->values_context = AllocSetContextCreate(estate->es_query_cxt,
											 "Vector Sort Values",
											 ALLOCSET_DEFAULT_MINSIZE,
											 ALLOCSET_DEFAULT_INITSIZE,
											 ALLOCSET_DEFAULT_MAXSIZE);
	for (col = 0; col < state->ncols; col++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, col);

		state->columns[col].typlen = attr->attlen;
		state->columns[col].typbyval = attr->attbyval;
	}
	for (col = 0; col < sort->numCols; col++)
	{
		SortSupport key = &state->sort_keys[col];

		key->ssup_cxt = estate->es_query_cxt;
		key->ssup_collation = sort->collations[col];
		key->ssup_nulls_first = sort->nullsFirst[col];
		key->ssup_attno = sort->sortColIdx[col];
		PrepareSortSupportFromOrderingOp(sort->sortOperators[col], key);
	}
	node->vector_sort_state = state;
	MemoryContextSwitchTo(oldcontext);
	return state;
}

static void
VectorSortPutBatchRows(SortState *node, const VectorBatch *batch,
					   const int *input_attnos)
{
	TupleTableSlot *input = node->ss.ss_ScanTupleSlot;
	int			pos;

	for (pos = 0; pos < batch->nselected; pos++)
	{
		int			row = VectorBatchSelectedRow(batch, pos);
		int			col;

		ExecClearTuple(input);
		for (col = 0; col < node->ss.ps.ps_ResultTupleDesc->natts; col++)
		{
			int			attno = input_attnos[col];

			Assert(attno >= 0 && attno < batch->ncols);
			input->tts_values[col] = VectorBatchGetDatum(batch, attno, row);
			input->tts_isnull[col] = VectorBatchIsNull(batch, attno, row);
		}
		input->tts_nvalid = node->ss.ps.ps_ResultTupleDesc->natts;
		ExecStoreVirtualTuple(input);
		VectorSortPutTuple(node, input);
	}
}

static void
VectorSortFallback(VectorSortState *state)
{
	SortState  *node = state->node;
	TupleTableSlot *input = node->ss.ss_ScanTupleSlot;
	int64		row;
	int			col;

	Assert(state->native);
	VectorSortBegin(node);
	for (row = 0; row < state->nrows; row++)
	{
		ExecClearTuple(input);
		for (col = 0; col < state->ncols; col++)
		{
			input->tts_values[col] = state->columns[col].values[row];
			input->tts_isnull[col] =
				(state->columns[col].nulls[row / 64] &
				 ((uint64) 1 << (row % 64))) != 0;
		}
		input->tts_nvalid = state->ncols;
		ExecStoreVirtualTuple(input);
		VectorSortPutTuple(node, input);
	}
	MemoryContextReset(state->values_context);
	state->order = NULL;
	for (col = 0; col < state->ncols; col++)
	{
		state->columns[col].values = NULL;
		state->columns[col].nulls = NULL;
	}
	state->nrows = 0;
	state->capacity = 0;
	state->position = 0;
	state->native = false;
}

static void
VectorSortGrow(VectorSortState *state)
{
	int64		new_capacity = state->capacity == 0 ?
		VECTOR_BATCH_DEFAULT_CAPACITY : Min(state->capacity * 2, (int64) INT32_MAX);
	MemoryContext oldcontext;
	int			col;

	Assert(new_capacity > state->capacity);
	oldcontext = MemoryContextSwitchTo(state->values_context);
	if (state->order == NULL)
		state->order = palloc(sizeof(int32) * new_capacity);
	else
		state->order = repalloc(state->order, sizeof(int32) * new_capacity);
	for (col = 0; col < state->ncols; col++)
	{
		VectorSortColumn *column = &state->columns[col];
		Size		nullsize = sizeof(uint64) * VECTOR_BATCH_NULLWORDS(new_capacity);

		if (column->values == NULL)
		{
			column->values = palloc0(sizeof(Datum) * new_capacity);
			column->nulls = palloc0(nullsize);
		}
		else
		{
			column->values = repalloc(column->values, sizeof(Datum) * new_capacity);
			column->nulls = repalloc(column->nulls, nullsize);
		}
	}
	state->capacity = new_capacity;
	MemoryContextSwitchTo(oldcontext);
}

static int
VectorSortCompare(const void *left, const void *right, void *arg)
{
	VectorSortState *state = (VectorSortState *) arg;
	int32		a = *(const int32 *) left;
	int32		b = *(const int32 *) right;
	Sort	   *sort = (Sort *) state->node->ss.ps.plan;
	int			key;

	for (key = 0; key < sort->numCols; key++)
	{
		int			col = sort->sortColIdx[key] - 1;
		VectorSortColumn *column = &state->columns[col];
		bool		anull = (column->nulls[a / 64] & ((uint64) 1 << (a % 64))) != 0;
		bool		bnull = (column->nulls[b / 64] & ((uint64) 1 << (b % 64))) != 0;
		int			cmp = ApplySortComparator(column->values[a], anull,
										  column->values[b], bnull,
										  &state->sort_keys[key]);

		if (cmp != 0)
			return cmp;
	}
	if (a < b)
		return -1;
	if (a > b)
		return 1;
	return 0;
}

void
VectorSortPutBatch(SortState *node, const VectorBatch *batch,
				   const int *input_attnos)
{
	VectorSortState *state = (VectorSortState *) node->vector_sort_state;
	MemoryContext oldcontext;
	int			pos;
	int			col;

	Assert(!node->sort_Done);
	if (state == NULL && !node->bounded && !node->randomAccess)
		state = VectorSortCreate(node);
	if (state == NULL || !state->native)
	{
		VectorSortPutBatchRows(node, batch, input_attnos);
		return;
	}
	if (state->nrows > (int64) INT32_MAX - batch->nselected)
	{
		VectorSortFallback(state);
		VectorSortPutBatchRows(node, batch, input_attnos);
		return;
	}
	while (state->capacity < state->nrows + batch->nselected)
		VectorSortGrow(state);

	oldcontext = MemoryContextSwitchTo(state->values_context);
	for (pos = 0; pos < batch->nselected; pos++)
	{
		int			input_row = VectorBatchSelectedRow(batch, pos);
		int64		row = state->nrows + pos;

		state->order[row] = row;
		for (col = 0; col < state->ncols; col++)
		{
			VectorSortColumn *column = &state->columns[col];
			int			attno = input_attnos[col];
			Datum		value;
			bool		isnull;
			uint64		mask = (uint64) 1 << (row % 64);

			Assert(attno >= 0 && attno < batch->ncols);
			isnull = VectorBatchIsNull(batch, attno, input_row);
			value = VectorBatchGetDatum(batch, attno, input_row);
			if (!isnull && !column->typbyval)
				value = datumCopy(value, column->typbyval, column->typlen);
			column->values[row] = value;
			if (isnull)
				column->nulls[row / 64] |= mask;
			else
				column->nulls[row / 64] &= ~mask;
		}
	}
	state->nrows += batch->nselected;
	MemoryContextSwitchTo(oldcontext);
	if (MemoryContextMemAllocated(state->values_context, true) > state->mem_limit)
		VectorSortFallback(state);
}

void
VectorSortFinish(SortState *node)
{
	VectorSortState *state = (VectorSortState *) node->vector_sort_state;
	Tuplesortstate *tuplesortstate;
	ScanDirection dir = node->ss.ps.state->es_direction;

	if (node->sort_Done)
		return;
	if (state != NULL && state->native)
	{
		if (state->nrows > 1)
			qsort_arg(state->order, state->nrows, sizeof(int32),
					  VectorSortCompare, state);
		state->position = 0;
		node->sort_Done = true;
		node->bounded_Done = node->bounded;
		node->bound_Done = node->bound;
		return;
	}
	tuplesortstate = VectorSortBegin(node);
	tuplesort_performsort(tuplesortstate);
	node->ss.ps.state->es_direction = dir;
	node->sort_Done = true;
	node->bounded_Done = node->bounded;
	node->bound_Done = node->bound;
	if (node->shared_info && node->am_worker)
	{
		TuplesortInstrumentation *si;

		Assert(IsParallelWorker());
		Assert(ParallelWorkerNumber <= node->shared_info->num_workers);
		si = &node->shared_info->sinstrument[ParallelWorkerNumber];
		tuplesort_get_stats(tuplesortstate, si);
	}
}

TupleTableSlot *
VectorSortNext(SortState *node)
{
	VectorSortState *state = (VectorSortState *) node->vector_sort_state;
	Tuplesortstate *tuplesortstate = (Tuplesortstate *) node->tuplesortstate;
	TupleTableSlot *slot;
	ScanDirection dir = node->ss.ps.state->es_direction;

	Assert(node->sort_Done);
	if (state != NULL && state->native)
	{
		int64		position;
		int32		row;
		int			col;

		slot = node->ss.ps.ps_ResultTupleSlot;
		ExecClearTuple(slot);
		if (state->position >= state->nrows)
		{
			if (!node->delayEagerFree)
				ExecEagerFreeSort(node);
			return slot;
		}
		position = state->position++;
		row = ScanDirectionIsForward(dir) ? state->order[position] :
			state->order[state->nrows - position - 1];
		for (col = 0; col < state->ncols; col++)
		{
			slot->tts_values[col] = state->columns[col].values[row];
			slot->tts_isnull[col] =
				(state->columns[col].nulls[row / 64] &
				 ((uint64) 1 << (row % 64))) != 0;
		}
		slot->tts_nvalid = state->ncols;
		return ExecStoreVirtualTuple(slot);
	}
	if (tuplesortstate == NULL)
		return NULL;
	slot = node->ss.ps.ps_ResultTupleSlot;
	(void) tuplesort_gettupleslot(tuplesortstate,
								  ScanDirectionIsForward(dir), false, slot, NULL);
	if (TupIsNull(slot) && !node->delayEagerFree)
		ExecEagerFreeSort(node);
	return slot;
}

bool
VectorSortNextBatch(SortState *node, VectorBatch *batch,
					const int *output_attnos)
{
	VectorSortState *state = (VectorSortState *) node->vector_sort_state;
	bool		native = state != NULL && state->native;
	bool		forward = ScanDirectionIsForward(node->ss.ps.state->es_direction);
	int			nrows = 0;
	int			col;

	Assert(node->sort_Done);
	VectorBatchReset(batch);
	if (native)
	{
		nrows = Min(batch->capacity, state->nrows - state->position);
		for (col = 0; col < state->ncols; col++)
		{
			VectorSortColumn *column = &state->columns[col];
			int			attno = output_attnos[col];
			int			pos;

			Assert(attno >= 0 && attno < batch->ncols);
			for (pos = 0; pos < nrows; pos++)
			{
				int64		position = state->position + pos;
				int32		row = forward ? state->order[position] :
					state->order[state->nrows - position - 1];

				VectorBatchSetValue(batch, attno, pos, column->values[row],
					(column->nulls[row / 64] & ((uint64) 1 << (row % 64))) != 0);
			}
		}
		state->position += nrows;
	}
	else
	{
		while (nrows < batch->capacity)
		{
			TupleTableSlot *slot = VectorSortNext(node);

			if (TupIsNull(slot))
				break;
			slot_getallattrs(slot);
			for (col = 0; col < node->ss.ps.ps_ResultTupleDesc->natts; col++)
			{
				int			attno = output_attnos[col];
				Datum		value = slot->tts_values[col];
				bool		isnull = slot->tts_isnull[col];

				Assert(attno >= 0 && attno < batch->ncols);
				value = VectorBatchCopyDatum(batch, attno, value, isnull);
				VectorBatchSetValue(batch, attno, nrows, value, isnull);
			}
			nrows++;
		}
	}
	VectorBatchSetRowCount(batch, nrows);
	if (nrows == 0 && native && !node->delayEagerFree)
		ExecEagerFreeSort(node);
	return nrows > 0;
}

void
VectorSortReset(SortState *node)
{
	VectorSortState *state = (VectorSortState *) node->vector_sort_state;
	int			col;

	if (state == NULL || !state->native)
		return;
	MemoryContextReset(state->values_context);
	state->order = NULL;
	for (col = 0; col < state->ncols; col++)
	{
		state->columns[col].values = NULL;
		state->columns[col].nulls = NULL;
	}
	state->nrows = 0;
	state->capacity = 0;
	state->position = 0;
}

void
VectorSortDestroy(SortState *node)
{
	VectorSortState *state = (VectorSortState *) node->vector_sort_state;

	if (state == NULL)
		return;
	MemoryContextDelete(state->values_context);
	pfree(state->columns);
	pfree(state->sort_keys);
	pfree(state);
	node->vector_sort_state = NULL;
}

TupleTableSlot *
VectorSortExec(PlanState *pstate)
{
	return VectorSortNext(castNode(SortState, pstate));
}
