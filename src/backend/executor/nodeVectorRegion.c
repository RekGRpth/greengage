/*-------------------------------------------------------------------------
 *
 * nodeVectorRegion.c
 *	  Executor node for a segment-local vectorized plan region.
 *
 * AOCS leaves fill column batches directly from datum streams.  Heap and AO
 * row leaves use the table AM's raw scan slot and deform referenced attributes
 * into the same batch representation.  The SeqScan child retains relation
 * initialization and EXPLAIN structure, but its row-at-a-time projection and
 * qual are not executed.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "access/sysattr.h"
#include "access/tableam.h"
#include "access/tupdesc.h"
#include "cdb/cdbaocsam.h"
#include "executor/executor.h"
#include "executor/execPartition.h"
#include "executor/nodeAppend.h"
#include "executor/nodeCustom.h"
#include "executor/nodeLimit.h"
#include "executor/nodeMaterial.h"
#include "executor/nodeMergejoin.h"
#include "executor/nodeNestloop.h"
#include "executor/nodeSeqscan.h"
#include "executor/nodeSort.h"
#include "executor/nodeUnique.h"
#include "executor/nodeVectorRegion.h"
#include "executor/vectorAgg.h"
#include "executor/vectorBatch.h"
#include "executor/vectorExpr.h"
#include "executor/vectorHashJoin.h"
#include "nodes/extensible.h"
#include "optimizer/optimizer.h"
#include "utils/rel.h"

#define VECTOR_REGION_NAME "VectorRegion"

typedef struct VectorRegionState
{
	CustomScanState css;
	SeqScanState *scan;
	AppendState *append;
	int			append_index;
	VectorExprState **append_quals;
	PlanState  *input;
	AggState   *agg;
	SortState  *sort;
	LimitState *limit;
	UniqueState *unique;
	MergeJoinState *mergejoin;
	MaterialState *material;
	PlanState  *mat_proxy;
	PlanState  *mat_outer;
	TupleTableSlot *mat_slot;
	NestLoopState *nestloop;
	PlanState  *nl_proxy;
	PlanState  *nl_outer;
	TupleTableSlot *nl_slot;
	HashJoinState *hj;
	VectorAggState *vagg;
	VectorHashJoinState *vhj;
	VectorBatch *batch;
	VectorExprState **proj;		/* per output column; NULL for plain columns */
	VectorExprState *qual;
	int		   *output_attnos;
	bool	   *read_attrs;
	int			output_ncols;
	int			position;
	bool		eof;
	/* Per-child layout remap for a divergent Append child (NULL arrays
	 * for children laid out like the first one). */
	AttrNumber **append_remaps;	/* child attno (1-based) -> batch attno, or 0 */
	int		   **append_dstcols;	/* child column -> batch column, or -1 */
	VectorBatch **append_batches;	/* per child: private fill batch */
	bool	   **append_read_attrs;	/* per child: row-reader deform set */
} VectorRegionState;

typedef struct VectorNestLoopProxy
{
	PlanState	ps;
	VectorRegionState *region;
} VectorNestLoopProxy;

static TupleTableSlot *VectorRegionNestLoopOuter(PlanState *pstate);
static TupleTableSlot *VectorRegionMaterialOuter(PlanState *pstate);
static Node *VectorRegionCreateState(CustomScan *cscan);
static void VectorRegionBegin(CustomScanState *node, EState *estate, int eflags);
static TupleTableSlot *VectorRegionExec(CustomScanState *node);
static void VectorRegionEnd(CustomScanState *node);
static void VectorRegionRescan(CustomScanState *node);
static void VectorRegionMarkPos(CustomScanState *node);
static void VectorRegionRestrPos(CustomScanState *node);

static const CustomScanMethods vector_region_scan_methods = {
	VECTOR_REGION_NAME,
	VectorRegionCreateState
};

static const CustomExecMethods vector_region_exec_methods = {
	VECTOR_REGION_NAME,
	VectorRegionBegin,
	VectorRegionExec,
	VectorRegionEnd,
	VectorRegionRescan,
	VectorRegionMarkPos,
	VectorRegionRestrPos,
	NULL,						/* EstimateDSM */
	NULL,						/* InitializeDSM */
	NULL,						/* ReInitializeDSM */
	NULL,						/* InitializeWorker */
	NULL,						/* Shutdown */
	NULL						/* Explain */
};

static bool vector_region_registered = false;

const CustomScanMethods *
GetVectorRegionScanMethods(void)
{
	return &vector_region_scan_methods;
}

void
RegisterVectorRegionScanMethods(void)
{
	if (vector_region_registered)
		return;

	RegisterCustomScanMethods(&vector_region_scan_methods);
	vector_region_registered = true;
}

static Node *
VectorRegionCreateState(CustomScan *cscan)
{
	VectorRegionState *state = palloc0(sizeof(VectorRegionState));

	NodeSetTag(state, T_CustomScanState);
	state->css.methods = &vector_region_exec_methods;
	return (Node *) state;
}

static void
VectorRegionBegin(CustomScanState *node, EState *estate, int eflags)
{
	VectorRegionState *state = (VectorRegionState *) node;
	CustomScan *cscan = (CustomScan *) node->ss.ps.plan;
	PlanState  *child;
	SeqScan	   *scan = NULL;
	Relation	relation = NULL;
	bool		needs_row_reader = false;
	ListCell   *lc;
	int			col = 0;

	foreach(lc, cscan->custom_plans)
	{
		Plan	   *plan = (Plan *) lfirst(lc);
		int			child_eflags = eflags & ~EXEC_FLAG_BACKWARD;
		PlanState  *childstate;

		if (!IsA(plan, Sort) && !IsA(plan, Material))
			child_eflags &= ~EXEC_FLAG_MARK;
		childstate = ExecInitNode(plan, estate, child_eflags);

		node->custom_ps = lappend(node->custom_ps, childstate);
	}

	if (list_length(node->custom_ps) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("vector region expects one plan child")));
	child = (PlanState *) linitial(node->custom_ps);
	if (IsA(child, AggState))
	{
		PlanState  *outer;

		state->agg = (AggState *) child;
		outer = outerPlanState(state->agg);
		if (IsA(outer, SeqScanState))
			state->scan = (SeqScanState *) outer;
		else if (IsA(outer, AppendState))
			state->append = (AppendState *) outer;
		else
			state->input = outer;
	}
	else if (IsA(child, MaterialState))
	{
		VectorNestLoopProxy *proxy;

		state->material = (MaterialState *) child;
		state->mat_outer = outerPlanState(state->material);
		if (!IsA(state->mat_outer, SeqScanState))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("vector material expects a SeqScan input")));
		state->scan = (SeqScanState *) state->mat_outer;
		if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		{
			proxy = MemoryContextAllocZero(estate->es_query_cxt,
										   sizeof(VectorNestLoopProxy));
			NodeSetTag(&proxy->ps, T_SeqScanState);
			proxy->ps.plan = state->mat_outer->plan;
			proxy->ps.state = estate;
			proxy->ps.ExecProcNode = VectorRegionMaterialOuter;
			proxy->region = state;
			state->mat_proxy = &proxy->ps;
			state->mat_slot = ExecInitExtraTupleSlot(estate,
										  state->mat_outer->ps_ResultTupleDesc,
										  &TTSOpsVirtual);
			outerPlanState(state->material) = state->mat_proxy;
		}
	}
	else if (IsA(child, NestLoopState))
	{
		VectorNestLoopProxy *proxy;

		state->nestloop = (NestLoopState *) child;
		state->nl_outer = outerPlanState(state->nestloop);
		if (!IsA(state->nl_outer, SeqScanState))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("vector nest loop expects a SeqScan outer input")));
		state->scan = (SeqScanState *) state->nl_outer;
		if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		{
			proxy = MemoryContextAllocZero(estate->es_query_cxt,
										   sizeof(VectorNestLoopProxy));
			NodeSetTag(&proxy->ps, T_SeqScanState);
			proxy->ps.plan = state->nl_outer->plan;
			proxy->ps.state = estate;
			proxy->ps.ExecProcNode = VectorRegionNestLoopOuter;
			proxy->region = state;
			state->nl_proxy = &proxy->ps;
			state->nl_slot = ExecInitExtraTupleSlot(estate,
										 state->nl_outer->ps_ResultTupleDesc,
										 &TTSOpsVirtual);
			outerPlanState(state->nestloop) = state->nl_proxy;
		}
	}
	else if (IsA(child, MergeJoinState))
	{
		PlanState  *outer;

		state->mergejoin = (MergeJoinState *) child;
		outer = outerPlanState(state->mergejoin);
		if (!IsA(outer, SortState))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("vector merge join expects a Sort outer input")));
		state->sort = (SortState *) outer;
		outer = outerPlanState(state->sort);
		if (IsA(outer, SeqScanState))
			state->scan = (SeqScanState *) outer;
		else
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("vector merge join expects a SeqScan below Sort")));
	}
	else if (IsA(child, UniqueState))
	{
		PlanState  *outer;

		state->unique = (UniqueState *) child;
		outer = outerPlanState(state->unique);
		if (!IsA(outer, SortState))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("vector unique expects a Sort input")));
		state->sort = (SortState *) outer;
		outer = outerPlanState(state->sort);
		if (IsA(outer, SeqScanState))
			state->scan = (SeqScanState *) outer;
		else
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("vector unique expects a SeqScan below Sort")));
	}
	else if (IsA(child, LimitState))
	{
		PlanState  *outer;

		state->limit = (LimitState *) child;
		outer = outerPlanState(state->limit);
		if (IsA(outer, SeqScanState))
			state->scan = (SeqScanState *) outer;
		else
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("vector limit expects a SeqScan input")));
	}
	else if (IsA(child, SortState))
	{
		PlanState  *outer;

		state->sort = (SortState *) child;
		outer = outerPlanState(state->sort);
		if (IsA(outer, SeqScanState))
			state->scan = (SeqScanState *) outer;
		else
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("vector sort expects a SeqScan input")));
	}
	else if (IsA(child, HashJoinState))
	{
		state->hj = (HashJoinState *) child;
		state->scan = (SeqScanState *) outerPlanState(state->hj);
	}
	else if (IsA(child, AppendState))
		state->append = (AppendState *) child;
	else if (IsA(child, SeqScanState))
		state->scan = (SeqScanState *) child;
	else
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("vector region expects a scan, append, aggregate, sort, limit, unique, merge join, nest loop, material or hash join")));

	if (state->append != NULL)
	{
		TupleDesc	firstdesc;
		int			i;

		if (state->append->as_nplans <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("vector append expects at least one input")));
		state->append_index = -1;
		state->scan = (SeqScanState *) state->append->appendplans[0];
		if (!IsA(state->scan, SeqScanState))
			elog(ERROR, "vector append child is not a SeqScan");
		firstdesc = RelationGetDescr(state->scan->ss.ss_currentRelation);
		needs_row_reader = !RelationIsAoCols(state->scan->ss.ss_currentRelation);

		/*
		 * Children laid out like the first one fill the shared batch
		 * directly; a divergent child (UNION ALL over independently
		 * created relations) fills a private batch of its own shape and
		 * the columns move over at the batch boundary.  The planner
		 * verified position-wise type equality and that every referenced
		 * attribute is covered by the targetlist mapping.
		 */
		state->append_remaps = MemoryContextAllocZero(estate->es_query_cxt,
											  sizeof(AttrNumber *) * state->append->as_nplans);
		state->append_dstcols = MemoryContextAllocZero(estate->es_query_cxt,
											  sizeof(int *) * state->append->as_nplans);
		state->append_batches = MemoryContextAllocZero(estate->es_query_cxt,
											  sizeof(VectorBatch *) * state->append->as_nplans);
		state->append_read_attrs = MemoryContextAllocZero(estate->es_query_cxt,
											  sizeof(bool *) * state->append->as_nplans);
		for (i = 1; i < state->append->as_nplans; i++)
		{
			SeqScanState *childscan = (SeqScanState *) state->append->appendplans[i];
			Relation	childrel = childscan->ss.ss_currentRelation;
			Relation	firstrel = state->scan->ss.ss_currentRelation;
			SeqScan   *childplan = (SeqScan *) childscan->ss.ps.plan;
			SeqScan   *firstplan = (SeqScan *) state->scan->ss.ps.plan;
			int			natts = RelationGetNumberOfAttributes(childrel);
			AttrNumber *remap;
			int		   *dstcol;
			ListCell   *firstlc,
					   *scanlc;

			if (!IsA(childscan, SeqScanState))
				elog(ERROR, "vector append child is not a SeqScan");
			if (!RelationIsAoCols(childrel))
				needs_row_reader = true;
			if (equalTupleDescs(RelationGetDescr(firstrel),
								RelationGetDescr(childrel), false))
				continue;		/* identical layout: shared batch */

			remap = MemoryContextAllocZero(estate->es_query_cxt,
										   sizeof(AttrNumber) * natts);
			dstcol = MemoryContextAlloc(estate->es_query_cxt,
										sizeof(int) * natts);
			MemSet(dstcol, -1, sizeof(int) * natts);
			forboth(firstlc, firstplan->plan.targetlist,
					scanlc, childplan->plan.targetlist)
			{
				TargetEntry *firstte = (TargetEntry *) lfirst(firstlc);
				TargetEntry *scante = (TargetEntry *) lfirst(scanlc);
				AttrNumber	a0 = ((Var *) firstte->expr)->varattno;
				AttrNumber	ai = ((Var *) scante->expr)->varattno;

				if (ai <= 0 || ai > natts)
					elog(ERROR, "vector append child column is out of range");
				if (TupleDescAttr(firstdesc, a0 - 1)->atttypid !=
					TupleDescAttr(RelationGetDescr(childrel), ai - 1)->atttypid)
					elog(ERROR,
						 "vector append children have incompatible tuple descriptors");
				remap[ai - 1] = a0;
				dstcol[ai - 1] = a0 - 1;
			}
			state->append_remaps[i] = remap;
			state->append_dstcols[i] = dstcol;
			state->append_batches[i] = VectorBatchCreate(estate->es_query_cxt,
														 RelationGetDescr(childrel),
														 VECTOR_BATCH_DEFAULT_CAPACITY);
			if (!RelationIsAoCols(childrel))
			{
				Bitmapset *attrs = NULL;
				bool	   *read_attrs = MemoryContextAllocZero(estate->es_query_cxt,
													   sizeof(bool) * natts);
				int			a;

				pull_varattnos((Node *) childplan->plan.targetlist,
							   childplan->scanrelid, &attrs);
				pull_varattnos((Node *) childplan->plan.qual,
							   childplan->scanrelid, &attrs);
				while ((a = bms_first_member(attrs)) > 0)
				{
					a += FirstLowInvalidHeapAttributeNumber;
					if (a > 0 && a <= natts && remap[a - 1] != 0)
						read_attrs[a - 1] = true;
				}
				bms_free(attrs);
				state->append_read_attrs[i] = read_attrs;
			}
		}
	}

	if (state->scan != NULL)
	{
		if (!IsA(state->scan, SeqScanState))
			elog(ERROR, "vector region leaf is not a SeqScan");
		scan = (SeqScan *) state->scan->ss.ps.plan;
		relation = state->scan->ss.ss_currentRelation;
	}

	if (state->agg == NULL && state->hj == NULL)
	{
		state->output_ncols = list_length(scan->plan.targetlist);
		if (state->mergejoin == NULL && state->nestloop == NULL &&
			state->output_ncols != node->ss.ss_ScanTupleSlot->tts_tupleDescriptor->natts)
			elog(ERROR, "vector region output shape does not match its scan");

		state->output_attnos = MemoryContextAlloc(estate->es_query_cxt,
										 sizeof(int) * state->output_ncols);
		foreach(lc, scan->plan.targetlist)
		{
			TargetEntry *te = (TargetEntry *) lfirst(lc);
			Var		   *var;

			/* plain column: read straight from the batch column */
			if (IsA(te->expr, Var) &&
				(var = (Var *) te->expr, true) &&
				var->varattno > 0 &&
				var->varattno <= RelationGetNumberOfAttributes(relation))
			{
				state->output_attnos[col++] = var->varattno - 1;
				continue;
			}
			state->output_attnos[col++] = -1;	/* compiled projection */
		}
	}

	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
	{
		state->batch = VectorBatchCreate(estate->es_query_cxt,
										 state->scan != NULL ?
										 RelationGetDescr(relation) :
										 state->agg->ss.ss_ScanTupleSlot->tts_tupleDescriptor,
										 VECTOR_BATCH_DEFAULT_CAPACITY);
		if (state->scan != NULL &&
			(needs_row_reader || !RelationIsAoCols(relation)))
		{
			Bitmapset  *attrs = NULL;
			int			natts = RelationGetNumberOfAttributes(relation);

			/* A row slot is reused on every fetch: deform only referenced attrs. */
			state->read_attrs = MemoryContextAllocZero(estate->es_query_cxt,
											 sizeof(bool) * natts);
			pull_varattnos((Node *) scan->plan.targetlist, scan->scanrelid, &attrs);
			pull_varattnos((Node *) scan->plan.qual, scan->scanrelid, &attrs);
			if (state->append != NULL)
			{
				int			i;

				for (i = 1; i < state->append->as_nplans; i++)
				{
					SeqScanState *childscan = (SeqScanState *) state->append->appendplans[i];
					SeqScan   *childplan = (SeqScan *) childscan->ss.ps.plan;

					pull_varattnos((Node *) childplan->plan.targetlist,
								   childplan->scanrelid, &attrs);
					pull_varattnos((Node *) childplan->plan.qual,
								   childplan->scanrelid, &attrs);
				}
			}
			for (col = 0; col < natts; col++)
				state->read_attrs[col] =
					bms_is_member(col + 1 - FirstLowInvalidHeapAttributeNumber, attrs);
			bms_free(attrs);
		}
		if (state->append != NULL)
		{
			int			i;

			state->append_quals = MemoryContextAllocZero(estate->es_query_cxt,
										   sizeof(VectorExprState *) * state->append->as_nplans);
			for (i = 0; i < state->append->as_nplans; i++)
			{
				SeqScanState *childscan = (SeqScanState *) state->append->appendplans[i];
				SeqScan   *childplan = (SeqScan *) childscan->ss.ps.plan;

				if (childplan->plan.qual == NIL)
					continue;
				{
					VectorExprVarMap vars = {childplan->scanrelid, NULL, 0, estate,
											 state->append_remaps != NULL ?
											 state->append_remaps[i] : NULL};

					vars.parent = &childscan->ss.ps;
					vars.allow_correlated = true;
					state->append_quals[i] = VectorExprCompile(estate->es_query_cxt,
															(Expr *) childplan->plan.qual,
															state->batch, &vars);
					if (state->append_quals[i] == NULL)
						elog(ERROR, "vector region could not compile its append qual");
				}
			}
			state->qual = state->append_quals[0];
		}
		else if (state->scan != NULL && scan->plan.qual != NIL)
		{
			VectorExprVarMap vars = {scan->scanrelid, NULL, 0, estate};

			vars.parent = &state->css.ss.ps;
			vars.allow_correlated = (state->append == NULL);
			state->qual = VectorExprCompile(estate->es_query_cxt,
											(Expr *) scan->plan.qual,
											state->batch, &vars);
			if (state->qual == NULL)
				elog(ERROR, "vector region could not compile its scan qual");
		}
		/*
		 * Projection entries of the scan targetlist (ORCA keeps computed
		 * expressions there): one compiled step per output column, NULL for
		 * plain columns.
		 */
		if (state->scan != NULL && state->agg == NULL && state->hj == NULL)
		{
			VectorExprVarMap vars = {scan->scanrelid, NULL, 0, estate};

			vars.parent = &state->css.ss.ps;
			vars.allow_correlated = true;
			col = 0;
			foreach(lc, scan->plan.targetlist)
			{
				TargetEntry *te = (TargetEntry *) lfirst(lc);

				if (state->output_attnos[col] == -1)
				{
					if (state->proj == NULL)
						state->proj = MemoryContextAllocZero(estate->es_query_cxt,
											  sizeof(VectorExprState *) * state->output_ncols);
					state->proj[col] = VectorExprCompile(estate->es_query_cxt,
														 te->expr,
														 state->batch, &vars);
					if (state->proj[col] == NULL)
						elog(ERROR, "vector region could not compile its projection");
				}
				col++;
			}
		}
		if (state->agg != NULL)
		{
			state->vagg = VectorAggCreate(estate->es_query_cxt, state->agg,
										  state->batch);
			if (state->vagg == NULL)
				elog(ERROR, "vector region could not initialize its aggregate");
		}
		if (state->hj != NULL)
		{
			state->vhj = VectorHashJoinCreate(estate->es_query_cxt, state->hj);
			if (state->vhj == NULL)
				elog(ERROR, "vector region could not initialize its hash join");
		}
	}
}

static bool
VectorRegionReadAdapter(VectorRegionState *state)
{
	VectorBatch *batch = state->batch;
	int			nrows = 0;
	bool		exhausted = false;

	if (state->eof)
		return false;
	VectorBatchReset(batch);
	while (nrows < batch->capacity)
	{
		TupleTableSlot *slot = ExecProcNode(state->input);
		int			col;

		if (TupIsNull(slot))
		{
			exhausted = true;
			break;
		}
		slot_getallattrs(slot);
		if (slot->tts_nvalid != batch->ncols)
			elog(ERROR, "vector region input shape changed during execution");
		for (col = 0; col < batch->ncols; col++)
		{
			Datum		value = slot->tts_values[col];
			bool		isnull = slot->tts_isnull[col];

			value = VectorBatchCopyDatum(batch, col, value, isnull);
			VectorBatchSetValue(batch, col, nrows, value, isnull);
		}
		nrows++;
	}
	VectorBatchSetRowCount(batch, nrows);
	state->position = 0;
	state->eof = exhausted;
	if (nrows == 0)
		return false;
	return true;
}

/*
 * Read a heap or AO row scan without executing its row-at-a-time projection
 * and qual.  The table AM fills the raw scan slot; referenced attributes are
 * deformed into columns and by-ref datums are copied before the slot is reused.
 */
static bool
VectorRegionReadRowScan(VectorRegionState *state, TableScanDesc scandesc,
						VectorBatch *batch)
{
	TupleTableSlot *slot = state->scan->ss.ss_ScanTupleSlot;
	bool	   *read_attrs = state->read_attrs;
	int			nrows = 0;

	/* a divergent append child deforms its own attribute set */
	if (batch != state->batch && state->append != NULL)
		read_attrs = state->append_read_attrs[state->append_index];

	VectorBatchReset(batch);
	while (nrows < batch->capacity)
	{
		int			col;

		if (!table_scan_getnextslot(scandesc,
								   state->scan->ss.ps.state->es_direction, slot))
		{
			state->eof = true;
			break;
		}

		for (col = 0; col < batch->ncols; col++)
		{
			Datum		value;
			bool		isnull;

			if (!read_attrs[col])
				continue;
			value = slot_getattr(slot, col + 1, &isnull);
			value = VectorBatchCopyDatum(batch, col, value, isnull);
			VectorBatchSetValue(batch, col, nrows, value, isnull);
		}
		nrows++;
	}

	VectorBatchSetRowCount(batch, nrows);
	return nrows > 0;
}

/*
 * Choose the first child to scan.  Exec pruning parameters are available
 * only at execution time, so the valid set is computed on first use (and
 * recomputed by VectorRegionRescan for nested-loop rescans).  Returns false
 * when no subplan survives pruning.
 */
static bool
append_first_valid(VectorRegionState *state)
{
	AppendState *append = state->append;
	Append	   *plan = (Append *) append->ps.plan;
	int			idx;

	if (append->as_prune_state != NULL && append->as_valid_subplans == NULL)
		append->as_valid_subplans = ExecFindMatchingSubPlans(
			append->as_prune_state, state->css.ss.ps.state,
			list_length(plan->appendplans), plan->join_prune_paramids);

	if (append->as_valid_subplans != NULL)
		idx = bms_next_member(append->as_valid_subplans, -1);
	else
		idx = 0;
	if (idx < 0)
		return false;
	state->append_index = idx;
	state->scan = (SeqScanState *) append->appendplans[idx];
	state->qual = state->append_quals[idx];
	state->eof = false;
	return true;
}

static bool
VectorRegionNextBatch(VectorRegionState *state)
{
	CustomScanState *node = &state->css;

	if (state->input != NULL)
		return VectorRegionReadAdapter(state);

	/* EOF may be known while the final partial batch still has unread rows. */
	if (state->position < state->batch->nselected)
		return true;

	for (;;)
	{
		if (state->append != NULL && state->append_index < 0 &&
			!append_first_valid(state))
			return false;
		while (!state->eof)
		{
			TableScanDesc scandesc;
			VectorBatch *fill = state->batch;
			bool		remapped = false;

			scandesc = state->scan->ss.ss_currentScanDesc;
			if (scandesc == NULL)
			{
				MemoryContext oldcontext = MemoryContextSwitchTo(node->ss.ps.state->es_query_cxt);

				scandesc = table_beginscan_es(state->scan->ss.ss_currentRelation,
											  node->ss.ps.state->es_snapshot,
											  state->scan->ss.ps.plan->targetlist,
											  state->scan->ss.ps.plan->qual,
											  NULL, NULL);
				state->scan->ss.ss_currentScanDesc = scandesc;
				MemoryContextSwitchTo(oldcontext);
			}

			if (state->append_dstcols != NULL &&
				state->append_dstcols[state->append_index] != NULL)
			{
				/* a divergent child fills its own batch; columns move over */
				fill = state->append_batches[state->append_index];
				remapped = true;
			}

			if (RelationIsAoCols(state->scan->ss.ss_currentRelation))
			{
				if (!aocs_getnext_batch((AOCSScanDesc) scandesc, fill))
				{
					state->eof = true;
					break;
				}
			}
			else if (!VectorRegionReadRowScan(state, scandesc, fill))
				break;
			if (remapped)
			{
				VectorBatchReset(state->batch);
				VectorBatchCopyColumns(state->batch, fill,
									   state->append_dstcols[state->append_index]);
			}
			state->position = 0;
			VectorExprFilter(state->batch, state->qual);
			/* A row scan can discover EOF while filling its final partial batch. */
			if (state->batch->nselected > 0)
				return true;
		}

		if (state->append == NULL)
			return false;
		if (state->append->as_valid_subplans != NULL)
			state->append_index = bms_next_member(state->append->as_valid_subplans,
											  state->append_index);
		else
			state->append_index++;
		if (state->append_index < 0)
		{
			/* no more children: park at the end, do not restart the append */
			state->append_index = state->append->as_nplans;
			return false;
		}
		if (state->append_index >= state->append->as_nplans)
			return false;
		state->scan = (SeqScanState *) state->append->appendplans[state->append_index];
		state->qual = state->append_quals[state->append_index];
		state->position = 0;
		state->eof = false;
	}
}

static void
VectorRegionSortInput(VectorRegionState *state)
{
	ScanDirection dir = state->css.ss.ps.state->es_direction;

	state->css.ss.ps.state->es_direction = ForwardScanDirection;
	while (VectorRegionNextBatch(state))
	{
		int			pos;

		for (pos = 0; pos < state->batch->nselected; pos++)
		{
			TupleTableSlot *input = state->sort->ss.ss_ScanTupleSlot;
			int			row = VectorBatchSelectedRow(state->batch, pos);
			int			col;

			ExecClearTuple(input);
			for (col = 0; col < state->output_ncols; col++)
			{
				int			attno = state->output_attnos[col];

				Assert(attno >= 0);	/* simple scan targetlist */
				input->tts_values[col] = state->batch->columns[attno].values[row];
				input->tts_isnull[col] = VectorBatchIsNull(state->batch, attno, row);
			}
			input->tts_nvalid = state->output_ncols;
			ExecStoreVirtualTuple(input);
			VectorSortPutTuple(state->sort, input);
		}
		state->position = state->batch->nselected;
	}
	VectorSortFinish(state->sort);
	state->css.ss.ps.state->es_direction = dir;
}

static TupleTableSlot *
VectorRegionMaterialOuter(PlanState *pstate)
{
	VectorNestLoopProxy *proxy = (VectorNestLoopProxy *) pstate;
	VectorRegionState *state = proxy->region;
	TupleTableSlot *slot = state->mat_slot;
	int			row;
	int			col;

	ExecClearTuple(slot);
	if (!VectorRegionNextBatch(state))
		return slot;
	row = VectorBatchSelectedRow(state->batch, state->position++);
	for (col = 0; col < state->output_ncols; col++)
	{
		int			attno = state->output_attnos[col];

		Assert(attno >= 0);	/* simple scan targetlist */
		slot->tts_values[col] = state->batch->columns[attno].values[row];
		slot->tts_isnull[col] = VectorBatchIsNull(state->batch, attno, row);
	}
	slot->tts_nvalid = state->output_ncols;
	return ExecStoreVirtualTuple(slot);
}

static TupleTableSlot *
VectorRegionNestLoopOuter(PlanState *pstate)
{
	VectorNestLoopProxy *proxy = (VectorNestLoopProxy *) pstate;
	VectorRegionState *state = proxy->region;
	TupleTableSlot *slot = state->nl_slot;
	int			row;
	int			col;

	ExecClearTuple(slot);
	if (!VectorRegionNextBatch(state))
		return slot;
	row = VectorBatchSelectedRow(state->batch, state->position++);
	for (col = 0; col < state->output_ncols; col++)
	{
		int			attno = state->output_attnos[col];

		Assert(attno >= 0);	/* simple scan targetlist */
		slot->tts_values[col] = state->batch->columns[attno].values[row];
		slot->tts_isnull[col] = VectorBatchIsNull(state->batch, attno, row);
	}
	slot->tts_nvalid = state->output_ncols;
	return ExecStoreVirtualTuple(slot);
}

static TupleTableSlot *
VectorRegionExec(CustomScanState *node)
{
	VectorRegionState *state = (VectorRegionState *) node;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	int			row;
	int			col;

	if (state->hj != NULL)
	{
		for (;;)
		{
			TupleTableSlot *result;

			if (!VectorRegionNextBatch(state))
				return VectorHashJoinFinish(state->vhj);
			if (VectorHashJoinInProgress(state->vhj))
				result = VectorHashJoinContinue(state->vhj);
			else
				result = VectorHashJoinNextRow(state->vhj, state->batch,
										   state->position++);
			if (result != NULL)
				return result;
		}
	}

	if (state->material != NULL)
		return VectorMaterialNext(state->material, state->mat_proxy, state->mat_outer);

	if (state->nestloop != NULL)
		return VectorNestLoopNext(state->nestloop, state->nl_proxy, state->nl_outer);

	if (state->mergejoin != NULL)
	{
		VectorRegionSortInput(state);
		state->sort->ss.ps.ExecProcNode = VectorSortExec;
		return VectorMergeJoinNext(state->mergejoin);
	}

	if (state->unique != NULL)
	{
		VectorRegionSortInput(state);
		for (;;)
		{
			TupleTableSlot *input = VectorSortNext(state->sort);
			TupleTableSlot *result;

			if (TupIsNull(input))
				return input;
			result = VectorUniqueNext(state->unique, input);
			if (result != NULL)
				return result;
		}
	}

	if (state->sort != NULL)
	{
		VectorRegionSortInput(state);
		return VectorSortNext(state->sort);
	}

	if (state->agg != NULL)
	{
		AggState   *agg = state->agg;

		if (agg->aggstrategy == AGG_SORTED)
		{
			while (!agg->input_done &&
				   (state->position < state->batch->nselected ||
					VectorRegionNextBatch(state)))
			{
				TupleTableSlot *result = VectorAggProcessBatchSorted(state->vagg,
															 state->batch,
															 &state->position);

				if (result != NULL)
					return result;
			}
		}
		else
		{
			while (VectorRegionNextBatch(state))
			{
				VectorAggProcessBatch(state->vagg, state->batch);
				state->position = state->batch->nselected;
			}
		}
		VectorAggFinish(state->vagg);
		return VectorAggNext(state->vagg);
	}

	if (state->limit != NULL && VectorLimitStart(state->limit))
	{
		ExecClearTuple(slot);
		return slot;
	}

	for (;;)
	{
		ExecClearTuple(slot);
		if (!VectorRegionNextBatch(state))
			return slot;

		/* evaluate projection steps once per batch, after the qual filter */
		if (state->proj != NULL)
		{
			for (col = 0; col < state->output_ncols; col++)
				if (state->proj[col] != NULL)
					VectorExprEval(state->batch, state->proj[col]);
		}

		while (state->position < state->batch->nselected)
		{
			row = VectorBatchSelectedRow(state->batch, state->position++);
			if (state->limit != NULL)
			{
				VectorLimitResult action = VectorLimitNext(state->limit);

				if (action == VECTOR_LIMIT_EOF)
					return slot;
				if (action == VECTOR_LIMIT_SKIP)
					continue;
			}
			for (col = 0; col < state->output_ncols; col++)
			{
				int			attno = state->output_attnos[col];

				if (attno >= 0)
				{
					slot->tts_values[col] = state->batch->columns[attno].values[row];
					slot->tts_isnull[col] = VectorBatchIsNull(state->batch, attno, row);
				}
				else
				{
					slot->tts_values[col] = VectorExprGetValue(state->proj[col], row,
															   &slot->tts_isnull[col]);
				}
			}
			slot->tts_nvalid = state->output_ncols;
			return ExecStoreVirtualTuple(slot);
		}
	}
}

static void
VectorRegionEnd(CustomScanState *node)
{
	VectorRegionState *state = (VectorRegionState *) node;
	ListCell   *lc;

	if (state->nestloop != NULL)
		outerPlanState(state->nestloop) = state->nl_outer;
	if (state->material != NULL)
		outerPlanState(state->material) = state->mat_outer;
	foreach(lc, node->custom_ps)
		ExecEndNode((PlanState *) lfirst(lc));
	node->custom_ps = NIL;

	if (state->vhj != NULL)
		VectorHashJoinDestroy(state->vhj);
	if (state->proj != NULL)
	{
		int			col;

		for (col = 0; col < state->output_ncols; col++)
			if (state->proj[col] != NULL)
				VectorExprDestroy(state->proj[col]);
		pfree(state->proj);
	}
	if (state->vagg != NULL)
		VectorAggDestroy(state->vagg);
	if (state->batch != NULL)
		VectorBatchDestroy(state->batch);
	if (state->append_quals != NULL)
	{
		int			i;

		for (i = 0; i < state->append->as_nplans; i++)
			if (state->append_quals[i] != NULL)
				VectorExprDestroy(state->append_quals[i]);
		pfree(state->append_quals);
	}
	else if (state->qual != NULL)
		VectorExprDestroy(state->qual);
	if (state->output_attnos != NULL)
		pfree(state->output_attnos);
	if (state->read_attrs != NULL)
		pfree(state->read_attrs);
	if (state->nl_proxy != NULL)
		pfree(state->nl_proxy);
	if (state->mat_proxy != NULL)
		pfree(state->mat_proxy);
}

static void
VectorRegionRescan(CustomScanState *node)
{
	VectorRegionState *state = (VectorRegionState *) node;
	ListCell   *lc;

	state->position = 0;
	state->eof = false;
	if (state->nestloop != NULL)
		outerPlanState(state->nestloop) = state->nl_outer;
	if (state->material != NULL)
		outerPlanState(state->material) = state->mat_outer;
	if (state->batch != NULL)
		VectorBatchReset(state->batch);
	if (state->vhj != NULL)
		VectorHashJoinReset(state->vhj);
	else if (state->vagg != NULL)
		VectorAggReset(state->vagg);
	else
	{
		foreach(lc, node->custom_ps)
			ExecReScan((PlanState *) lfirst(lc));
	}
	if (state->append != NULL)
	{
		if (state->append->as_prune_state != NULL &&
			state->append->as_prune_state->do_exec_prune)
			state->append->as_valid_subplans = NULL;
		state->append_index = -1;
		state->scan = (SeqScanState *) state->append->appendplans[0];
		state->qual = state->append_quals[0];
	}
	if (state->nestloop != NULL)
		outerPlanState(state->nestloop) = state->nl_proxy;
	if (state->material != NULL)
		outerPlanState(state->material) = state->mat_proxy;
}

static void
VectorRegionMarkPos(CustomScanState *node)
{
	VectorRegionState *state = (VectorRegionState *) node;

	if (state->sort != NULL)
		ExecSortMarkPos(state->sort);
	else if (state->material != NULL)
		ExecMaterialMarkPos(state->material);
	else
		elog(ERROR, "vector region supports MarkPos only for Sort or Material");
}

static void
VectorRegionRestrPos(CustomScanState *node)
{
	VectorRegionState *state = (VectorRegionState *) node;

	if (state->sort != NULL)
		ExecSortRestrPos(state->sort);
	else if (state->material != NULL)
		ExecMaterialRestrPos(state->material);
	else
		elog(ERROR, "vector region supports RestrPos only for Sort or Material");
}
