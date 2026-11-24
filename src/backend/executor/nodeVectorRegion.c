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
#include "executor/nodeMotion.h"
#include "nodes/extensible.h"
#include "optimizer/optimizer.h"
#include "utils/rel.h"

#define VECTOR_REGION_NAME "VectorRegion"

/*
 * Batch-leaf for the build side of a nested join chain: a second scan whose
 * batches feed the hash build, independent of the region's own leaf.
 */
typedef struct VectorBuildScanState
{
	MemoryContext context;		/* owns batch, qual, read_attrs */
	SeqScanState *scan;
	VectorBatch *batch;
	VectorExprState *qual;
	bool	   *read_attrs;
	bool		eof;
} VectorBuildScanState;

/*
 * Probe-leaf source for a join chain whose leaf is a row producer (a Motion
 * receiver or a nested region's output): rows convert into batches through
 * the same deform-and-copy loop as the region's input adapter.
 */
typedef struct VectorRowLeafState
{
	PlanState  *input;
	VectorBatch *batch;
	bool		eof;
} VectorRowLeafState;

/*
 * Adaptive leaf over a Motion receiver: the first pull decides whether
 * the stream carries columnar batches (batch interconnect) or plain
 * tuples; tuples fall back to the embedded row adapter, which picks up
 * exactly the tuples the probe already saw.
 */
typedef struct VectorMotionLeafState
{
	MotionState *motion;
	VectorRowLeafState row;		/* row.input == (PlanState *) motion */
	int			decided;		/* 0 = first pull, 1 = batch, 2 = rows, 3 = eos */
	VectorBatch *current;		/* handed out; destroyed on replace/EOF */
} VectorMotionLeafState;

typedef struct VectorRegionState
{
	CustomScanState css;
	SeqScanState *scan;
	/*
	 * Row source feeding a join chain's probe side when the leaf is not a
	 * scan: a Motion receiver or a nested region's row output, converted to
	 * batches by the row adapter.
	 */
	struct VectorRowLeafState *row_leaf;
	struct VectorMotionLeafState *motion_leaf;
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
	VectorHashJoinState *build_vhj;
	VectorBuildScanState *build_scan;
	/*
	 * Row source feeding a build chain whose leaf is neither a scan nor a
	 * join: a Motion or a nested region's row output, converted to batches
	 * by the row adapter (e.g. the NOT IN anti-semi leaf of q16-like
	 * shapes).
	 */
	struct VectorRowLeafState *build_row_leaf;
	/*
	 * Adaptive leaf over a Motion receiver feeding a build chain: batch
	 * interconnect frames hand over as batches directly, plain tuple
	 * streams fall back to the embedded row adapter (kept in sync with
	 * build_row_leaf above so a chain never carries both).  Kept at the
	 * struct end: ABI-stable extension like the other build fields.
	 */
	struct VectorMotionLeafState *build_motion_leaf;
	VectorBatch *batch;
	VectorBatch *join_batch;
	bool		nested_batches;
	int			join_position;	/* output cursor within join_batch */
	VectorExprState **proj;		/* per output column; NULL for plain columns */
	VectorExprState *qual;
	int		   *output_attnos;
	int		   *sort_attnos;	/* identity column map for a region-driven Sort */
	bool	   *read_attrs;
	int			output_ncols;
	int			position;
	bool		eof;
	bool		packed_i32;
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
static bool VectorRegionInitJoinChain(VectorRegionState *state);
static void VectorRowLeafDestroy(VectorRowLeafState *leaf);
static bool VectorRegionSortIsPassthrough(SortState *sort);
static void VectorRegionInitBuildChain(VectorRegionState *state);
static bool VectorRegionNextScanBatch(VectorRegionState *state);
static VectorBatch *VectorRegionInputBatch(VectorRegionState *state);
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

static bool
packed_i32_targetlist(List *targetlist)
{
	ListCell   *lc;

	foreach(lc, targetlist)
		if (!VectorExprPackedI32IsSupported(((TargetEntry *) lfirst(lc))->expr))
			return false;
	return true;
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
		/*
		 * A pass-through Sort below the aggregate (the sorted-aggregate plan
		 * shape) is driven by the region: the join chain feeds the native
		 * batch sort, and the aggregate reads sorted batches from it.  Only
		 * the join-chain shape is wired this way; anything else keeps the
		 * row-wise input adapter over the Sort.
		 */
		if (IsA(outer, SortState) &&
			VectorRegionSortIsPassthrough((SortState *) outer) &&
			IsA(outerPlanState((SortState *) outer), HashJoinState))
		{
			state->sort = (SortState *) outer;
			outer = outerPlanState(state->sort);
		}
		if (IsA(outer, SeqScanState))
			state->scan = (SeqScanState *) outer;
		else if (IsA(outer, AppendState))
			state->append = (AppendState *) outer;
		else if (IsA(outer, HashJoinState))
			state->hj = (HashJoinState *) outer;
		else
			state->input = state->sort != NULL ? (PlanState *) state->sort : outer;
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
			/* the stat walkers visit this hand-made state; give it stats */
			if (estate->es_instrument)
				proxy->ps.instrument = GpInstrAlloc(proxy->ps.plan,
													estate->es_instrument);
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
			/* the stat walkers visit this hand-made state; give it stats */
			if (estate->es_instrument)
				proxy->ps.instrument = GpInstrAlloc(proxy->ps.plan,
													estate->es_instrument);
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
		/*
		 * A Motion probe leaf (a producer join on another slice feeding this
		 * join) is consumed through the row adapter: received rows convert
		 * into the probe batches of the join chain.
		 */
		if (IsA(outerPlanState(state->hj), SeqScanState))
			state->scan = (SeqScanState *) outerPlanState(state->hj);
		else
			state->input = outerPlanState(state->hj);
	}
	else if (IsA(child, AppendState))
		state->append = (AppendState *) child;
	else if (IsA(child, SeqScanState))
		state->scan = (SeqScanState *) child;
	else
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("vector region expects a scan, append, aggregate, sort, limit, unique, merge join, nest loop, material or hash join")));

	/*
	 * An aggregate over a hash join chain consumes join output batches;
	 * an unsupported chain falls back to the row-wise input adapter.
	 */
	if (state->agg != NULL && state->hj != NULL &&
		!VectorRegionInitJoinChain(state))
	{
		state->scan = NULL;
		state->hj = NULL;
		state->sort = NULL;
		VectorRowLeafDestroy(state->row_leaf);
		state->row_leaf = NULL;
		state->input = outerPlanState(state->agg);
	}
	else if (state->nested_batches)
		VectorRegionInitBuildChain(state);

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
			{
				/*
				 * Identical layout: shared batch.  A non-AO child still
				 * gets its own row-reader deform set so it does not pay
				 * for columns only its siblings reference.
				 */
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
						if (a > 0 && a <= natts)
							read_attrs[a - 1] = true;
					}
					bms_free(attrs);
					state->append_read_attrs[i] = read_attrs;
				}
				continue;
			}

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
														 VECTOR_BATCH_DEFAULT_CAPACITY,
													 state->packed_i32);
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

	if (state->agg != NULL && state->scan != NULL && state->append == NULL &&
		(scan->plan.qual == NIL ||
		 VectorExprPackedI32QualIsSupported((Expr *) scan->plan.qual)))
	{
		if (state->nested_batches)
		{
			/*
			 * The join consumes the scan batch through widening accessors
			 * and the aggregate consumes the (unpacked) join output, so
			 * only the batch qual must be packed-capable; the scan
			 * targetlist may hold any column types.
			 */
			state->packed_i32 = true;
		}
		else if (VectorAggPackedI32IsSupported(state->agg) &&
				 packed_i32_targetlist(scan->plan.targetlist))
			state->packed_i32 = true;
	}
	else if (state->sort != NULL && state->scan != NULL && state->append == NULL &&
			 (scan->plan.qual == NIL ||
			  VectorExprPackedI32QualIsSupported((Expr *) scan->plan.qual)) &&
			 packed_i32_targetlist(scan->plan.targetlist))
		state->packed_i32 = true;

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
										 state->agg != NULL ?
										 state->agg->ss.ss_ScanTupleSlot->tts_tupleDescriptor :
										 /* a Motion probe leaf delivers its
										  * producer's output columns as-is */
										 ExecGetResultType(outerPlanState(state->hj)),
										 VECTOR_BATCH_DEFAULT_CAPACITY,
										 state->packed_i32);
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
			/*
			 * A whole-row var arrives as attno 0 in the pulled set and reads
			 * every attribute; without this no column would be deformed and
			 * the whole-row expression would assemble an all-zero tuple.
			 */
			if (bms_is_member(0 - FirstLowInvalidHeapAttributeNumber, attrs))
			{
				for (col = 0; col < natts; col++)
					state->read_attrs[col] = true;
				bms_free(attrs);
			}
			else
			{
				/*
				 * Siblings carry their own deform sets (append_read_attrs);
				 * only this child's referenced attributes belong here.
				 */
				for (col = 0; col < natts; col++)
					state->read_attrs[col] =
						bms_is_member(col + 1 - FirstLowInvalidHeapAttributeNumber, attrs);
				bms_free(attrs);
			}
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
		if (state->sort != NULL)
		{
			/* the sort consumes and produces the join output columns as-is */
			int			ncols = ExecGetResultType(outerPlanState(state->sort))->natts;
			int			i;

			state->sort_attnos = MemoryContextAlloc(estate->es_query_cxt,
											 sizeof(int) * Max(ncols, 1));
			for (i = 0; i < ncols; i++)
				state->sort_attnos[i] = i;
		}
		if (state->agg != NULL)
		{
			state->vagg = VectorAggCreate(estate->es_query_cxt, state->agg,
										  VectorRegionInputBatch(state));
			if (state->vagg == NULL)
				elog(ERROR, "vector region could not initialize its aggregate");
		}
		if (state->hj != NULL && !state->nested_batches)
		{
			/*
			 * An HJ-root region can drive its join from batches like an
			 * aggregate-over-join region does: the probe feed comes from the
			 * chain machinery and the output rows are the join batches
			 * themselves.  Fall back to the per-row driver when the chain
			 * is not batchable.
			 */
			if (VectorRegionInitJoinChain(state))
			{
				state->output_ncols = state->join_batch->ncols;
				state->output_attnos = MemoryContextAlloc(estate->es_query_cxt,
													sizeof(int) * state->output_ncols);
				for (col = 0; col < state->output_ncols; col++)
					state->output_attnos[col] = col;
				VectorRegionInitBuildChain(state);
			}
			else
			{
				state->vhj = VectorHashJoinCreate(estate->es_query_cxt, state->hj);
				if (state->vhj == NULL)
					elog(ERROR, "vector region could not initialize its hash join");
				/*
				 * The build side may itself be a join chain (e.g. a NOT IN
				 * anti-semi leaf); feed the hash build from its output batches
				 * when it batch-probes.  On any failure the build stays with
				 * the row-wise MultiExecHash pull.
				 */
				VectorRegionInitBuildChain(state);
			}
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
/* Fill a batch from a row slot stream; shared by region and build leaves. */
static bool
VectorScanRowsToBatch(VectorBatch *batch, TupleTableSlot *slot,
					  TableScanDesc scandesc, ScanDirection dir,
					  bool *read_attrs, bool *eof)
{
	int			nrows = 0;
	int			col;

	VectorBatchReset(batch);
	while (nrows < batch->capacity)
	{
		if (!table_scan_getnextslot(scandesc, dir, slot))
		{
			*eof = true;
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

static bool
VectorRegionReadRowScan(VectorRegionState *state, TableScanDesc scandesc,
						VectorBatch *batch)
{
	TupleTableSlot *slot = state->scan->ss.ss_ScanTupleSlot;
	bool	   *read_attrs = state->read_attrs;

	/* an append child with its own attribute set deforms only those */
	if (state->append != NULL &&
		state->append_read_attrs[state->append_index] != NULL)
		read_attrs = state->append_read_attrs[state->append_index];

	return VectorScanRowsToBatch(batch, slot, scandesc,
								 state->scan->ss.ps.state->es_direction,
								 read_attrs, &state->eof);
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
VectorRegionLeafSource(void *arg, VectorBatch **batch)
{
	VectorRegionState *state = (VectorRegionState *) arg;

	*batch = state->batch;
	if (!VectorRegionNextScanBatch(state))
		return false;
	/* the join driver walks rows itself: mark the batch fully consumed */
	state->position = state->batch->nselected;
	return true;
}

static bool
VectorHashJoinSourceNextBatch(void *arg, VectorBatch **batch)
{
	VectorHashJoinState *state = (VectorHashJoinState *) arg;

	*batch = VectorHashJoinOutput(state);
	return VectorHashJoinNextBatch(state);
}

static void
VectorRowLeafDestroy(VectorRowLeafState *leaf)
{
	if (leaf == NULL)
		return;
	if (leaf->batch != NULL)
		VectorBatchDestroy(leaf->batch);
	pfree(leaf);
}

/* Free an adaptive motion leaf and whichever of its two paths holds data. */
static void
VectorMotionLeafDestroy(VectorMotionLeafState *leaf)
{
	if (leaf == NULL)
		return;
	if (leaf->current != NULL)
		VectorBatchDestroy(leaf->current);
	if (leaf->row.batch != NULL)
		VectorBatchDestroy(leaf->row.batch);
	pfree(leaf);
}

static bool
VectorRowLeafNext(VectorRowLeafState *leaf)
{
	VectorBatch *batch = leaf->batch;
	int			nrows = 0;
	int			col;

	if (leaf->eof)
		return false;
	VectorBatchReset(batch);
	while (nrows < batch->capacity)
	{
		TupleTableSlot *slot = ExecProcNode(leaf->input);

		if (TupIsNull(slot))
		{
			leaf->eof = true;
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
	return nrows > 0;
}

static bool
VectorRowLeafSource(void *arg, VectorBatch **batch)
{
	VectorRowLeafState *leaf = (VectorRowLeafState *) arg;

	*batch = leaf->batch;
	/* every call hands out one fresh batch from the row source */
	return VectorRowLeafNext(leaf);
}

static bool
VectorMotionLeafNext(void *arg)
{
	VectorMotionLeafState *leaf = (VectorMotionLeafState *) arg;

	if (leaf->decided == 2)
		return VectorRowLeafNext(&leaf->row);

	if (leaf->decided == 3)
		return false;

	{
		bool		tuples_only = false;
		VectorBatch *batch = MotionStateRecvBatch(leaf->motion, &tuples_only);

		if (batch == NULL)
		{
			leaf->decided = tuples_only ? 2 : 3;
			if (leaf->current != NULL)
			{
				VectorBatchDestroy(leaf->current);
				leaf->current = NULL;
			}
			return leaf->decided == 2 ? VectorRowLeafNext(&leaf->row) : false;
		}

		leaf->decided = 1;
		if (leaf->current != NULL)
			VectorBatchDestroy(leaf->current);
		leaf->current = batch;
		return true;
	}
}

static bool
VectorMotionLeafSource(void *arg, VectorBatch **batch)
{
	VectorMotionLeafState *leaf = (VectorMotionLeafState *) arg;

	if (leaf->decided == 2)
	{
		*batch = leaf->row.batch;
		return VectorRowLeafNext(&leaf->row);
	}

	if (!VectorMotionLeafNext(arg))
		return false;
	*batch = leaf->decided == 1 ? leaf->current : leaf->row.batch;
	return true;
}

static VectorHashJoinState *
VectorRegionCreateJoinChain(VectorRegionState *region, HashJoinState *hj)
{
	PlanState  *outer = outerPlanState(hj);
	VectorHashJoinState *outer_join = NULL;
	VectorHashJoinSourceNext source = VectorRegionLeafSource;
	void	   *source_arg = region;
	VectorHashJoinState *vhj;

	if (!VectorHashJoinPlanIsSupported((HashJoin *) hj->js.ps.plan))
		return NULL;
	if (IsA(outer, HashJoinState))
	{
		outer_join = VectorRegionCreateJoinChain(region, (HashJoinState *) outer);
		if (outer_join == NULL)
			return NULL;
		source = VectorHashJoinSourceNextBatch;
		source_arg = outer_join;
	}
	else if (IsA(outer, SeqScanState))
		region->scan = (SeqScanState *) outer;
	else if (IsA(outer, MotionState))
	{
		region->motion_leaf = MemoryContextAllocZero(region->css.ss.ps.state->es_query_cxt,
													 sizeof(VectorMotionLeafState));
		region->motion_leaf->motion = (MotionState *) outer;
		region->motion_leaf->row.input = outer;
		region->motion_leaf->row.batch = VectorBatchCreate(region->css.ss.ps.state->es_query_cxt,
														   ExecGetResultType(outer),
														   VECTOR_BATCH_DEFAULT_CAPACITY,
														   false);
		source = VectorMotionLeafSource;
		source_arg = region->motion_leaf;
	}
	else if (IsA(outer, CustomScanState))
	{
		region->row_leaf = MemoryContextAllocZero(region->css.ss.ps.state->es_query_cxt,
												  sizeof(VectorRowLeafState));
		region->row_leaf->input = outer;
		region->row_leaf->batch = VectorBatchCreate(region->css.ss.ps.state->es_query_cxt,
													ExecGetResultType(outer),
													VECTOR_BATCH_DEFAULT_CAPACITY,
													false);
		source = VectorRowLeafSource;
		source_arg = region->row_leaf;
	}
	else
		return NULL;
	vhj = VectorHashJoinCreateBatch(region->css.ss.ps.state->es_query_cxt,
									hj, source, source_arg, outer_join);
	if (vhj == NULL)
	{
		VectorHashJoinDestroy(outer_join);
		VectorRowLeafDestroy(region->row_leaf);
		region->row_leaf = NULL;
		if (region->motion_leaf != NULL)
		{
			if (region->motion_leaf->current != NULL)
				VectorBatchDestroy(region->motion_leaf->current);
			VectorBatchDestroy(region->motion_leaf->row.batch);
			pfree(region->motion_leaf);
			region->motion_leaf = NULL;
		}
	}
	return vhj;
}

static bool
VectorRegionInitJoinChain(VectorRegionState *state)
{
	VectorHashJoinState *vhj = VectorRegionCreateJoinChain(state, state->hj);

	if (vhj == NULL)
		return false;
	state->join_batch = VectorHashJoinOutput(vhj);
	state->nested_batches = true;
	state->vhj = vhj;
	return true;
}

/*
 * A Sort the region may drive itself between the aggregate and the join
 * chain: a plain pass-through of its input, so the aggregate reads the join
 * output columns unchanged.  Junk entries (sort keys) are fine; anything
 * computed in the targetlist is not.
 */
static bool
VectorRegionSortIsPassthrough(SortState *sort)
{
	Sort	   *plan = (Sort *) sort->ss.ps.plan;
	ListCell   *lc;
	int			ncols = 0;

	if (plan->plan.qual != NIL)
		return false;
	foreach(lc, plan->plan.targetlist)
	{
		TargetEntry *te = lfirst_node(TargetEntry, lc);
		Var		   *var;

		if (te->resjunk)
			continue;
		if (!IsA(te->expr, Var))
			return false;
		var = (Var *) te->expr;
		if (var->varno != OUTER_VAR || var->varattno != te->resno)
			return false;
		ncols++;
	}
	return ncols == ExecGetResultType(outerPlanState(sort))->natts;
}

static void
VectorBuildScanDestroy(VectorBuildScanState *vs)
{
	if (vs == NULL)
		return;
	if (vs->qual != NULL)
		VectorExprDestroy(vs->qual);
	if (vs->context != NULL)
		MemoryContextDelete(vs->context);	/* owns batch and read_attrs */
	pfree(vs);
}

static VectorBuildScanState *
VectorBuildScanCreate(VectorRegionState *region, SeqScanState *scan,
					  EState *estate)
{
	VectorBuildScanState *vs;
	Relation	relation = scan->ss.ss_currentRelation;
	SeqScan    *plan = (SeqScan *) scan->ss.ps.plan;
	MemoryContext oldcontext;
	Bitmapset  *attrs = NULL;
	int			natts = RelationGetNumberOfAttributes(relation);
	int			a;

	if (!RelationIsAoCols(relation))
		return NULL;			/* batch reads only for AOCS in the build leaf */

	vs = MemoryContextAllocZero(estate->es_query_cxt, sizeof(VectorBuildScanState));
	vs->context = AllocSetContextCreate(estate->es_query_cxt,
										"Vector Build Scan",
										ALLOCSET_DEFAULT_MINSIZE,
										ALLOCSET_DEFAULT_INITSIZE,
										ALLOCSET_DEFAULT_MAXSIZE);
	vs->scan = scan;
	oldcontext = MemoryContextSwitchTo(vs->context);
	vs->batch = VectorBatchCreate(vs->context, RelationGetDescr(relation),
								  VECTOR_BATCH_DEFAULT_CAPACITY,
								  plan->plan.qual == NIL ||
								  VectorExprPackedI32QualIsSupported(
									  (Expr *) plan->plan.qual));

	vs->read_attrs = MemoryContextAllocZero(vs->context, sizeof(bool) * natts);
	pull_varattnos((Node *) plan->plan.targetlist, plan->scanrelid, &attrs);
	pull_varattnos((Node *) plan->plan.qual, plan->scanrelid, &attrs);
	while ((a = bms_first_member(attrs)) > 0)
	{
		a += FirstLowInvalidHeapAttributeNumber;
		if (a > 0 && a <= natts)
			vs->read_attrs[a - 1] = true;
	}
	bms_free(attrs);

	if (plan->plan.qual != NIL)
	{
		VectorExprVarMap vars = {plan->scanrelid, NULL, 0, estate};

		vars.parent = &scan->ss.ps;
		vars.allow_correlated = false;
		vs->qual = VectorExprCompile(vs->context, (Expr *) plan->plan.qual,
									 vs->batch, &vars);
		if (vs->qual == NULL)
		{
			MemoryContextSwitchTo(oldcontext);
			VectorBuildScanDestroy(vs);
			return NULL;
		}
	}

	MemoryContextSwitchTo(oldcontext);
	return vs;
}

static bool
VectorBuildScanNext(VectorBuildScanState *vs)
{
	TableScanDesc scandesc;

	for (;;)
	{
		if (vs->eof)
			return false;
		scandesc = vs->scan->ss.ss_currentScanDesc;
		if (scandesc == NULL)
		{
			MemoryContext oldcontext = MemoryContextSwitchTo(vs->context);

			scandesc = table_beginscan_es(vs->scan->ss.ss_currentRelation,
										  vs->scan->ss.ps.state->es_snapshot,
										  ((SeqScan *) vs->scan->ss.ps.plan)->plan.targetlist,
										  ((SeqScan *) vs->scan->ss.ps.plan)->plan.qual,
										  NULL, NULL);
			vs->scan->ss.ss_currentScanDesc = scandesc;
			MemoryContextSwitchTo(oldcontext);
		}
		{
			bool		got;
			Instrumentation *instr = vs->scan->ss.ps.instrument;

			if (instr)
				InstrStartNode(instr);
			got = aocs_getnext_batch((AOCSScanDesc) scandesc, vs->batch);
			if (instr)
				InstrStopNode(instr, got ? (uint64) vs->batch->nrows : 0);
			if (!got)
			{
				vs->eof = true;
				return false;
			}
		}
		VectorExprFilter(vs->batch, vs->qual);
		if (vs->batch->nselected > 0)
			return true;
	}
}

static bool
VectorBuildLeafSource(void *arg, VectorBatch **batch)
{
	VectorBuildScanState *vs = (VectorBuildScanState *) arg;

	*batch = vs->batch;
	/* every call hands out one fresh scan batch */
	return VectorBuildScanNext(vs);
}

/*
 * The build side of the top join is a Hash over a join chain; when that
 * chain's probe side is batchable, run it batch-wise and feed the hash build
 * from its output batches instead of MultiExecHash's row-wise pull.
 */
static bool
VectorRegionOutputBatchable(VectorRegionState *state)
{
	int			col;

	/* an aggregate or sort consumes the join batches: row-wise output */
	if (state->agg != NULL || state->sort != NULL)
		return false;
	if (state->nested_batches)
		return true;
	if (state->hj != NULL || state->nestloop != NULL ||
		state->mergejoin != NULL || state->material != NULL ||
		state->unique != NULL ||
		state->scan == NULL || state->append != NULL ||
		state->limit != NULL || state->packed_i32 ||
		state->motion_leaf != NULL || state->row_leaf != NULL)
		return false;
	/* computed projections do not materialize into batch columns */
	if (state->proj != NULL)
		return false;
	/* the motion output columns must be the batch columns as-is; a batch
	 * wider than the output is a fine prefix, but an output column with no
	 * batch column behind it (e.g. a void column from a nested-loop inner)
	 * must fall back to the row-wise output */
	if (state->output_ncols > state->batch->ncols)
		return false;
	/* the output columns must be the batch columns as-is */
	for (col = 0; col < state->output_ncols; col++)
	{
		if (state->output_attnos[col] != col)
			return false;
	}
	return true;
}

/* Pull-convention feed over a nested build join's output batches. */
static bool
VectorBuildChainSourceNext(void *arg, VectorBatch **batch)
{
	VectorHashJoinState *state = (VectorHashJoinState *) arg;

	if (!VectorHashJoinNextBatch(state))
	{
		*batch = NULL;
		return true;
	}
	*batch = VectorHashJoinOutput(state);
	return true;
}

/* Pull-convention feed straight from a region's native output batches. */
static bool
VectorRegionBuildSourceNext(void *arg, VectorBatch **batch)
{
	(void) VectorRegionPullOutputBatch((PlanState *) arg, batch);
	return *batch != NULL;
}

static VectorHashJoinState *
VectorRegionCreateBuildChain(VectorRegionState *region, HashJoinState *hj)
{
	PlanState  *outer = outerPlanState(hj);
	VectorHashJoinState *outer_join = NULL;
	VectorHashJoinSourceNext source;
	void	   *source_arg;
	VectorHashJoinState *vhj;

	if (!VectorHashJoinPlanIsSupported((HashJoin *) hj->js.ps.plan))
		return NULL;
	if (IsA(outer, HashJoinState))
	{
		outer_join = VectorRegionCreateBuildChain(region, (HashJoinState *) outer);
		if (outer_join == NULL)
			return NULL;
		source = VectorHashJoinSourceNextBatch;
		source_arg = outer_join;
	}
	else if (IsA(outer, SeqScanState))
	{
		region->build_scan = VectorBuildScanCreate(region, (SeqScanState *) outer,
												   region->css.ss.ps.state);
		if (region->build_scan == NULL)
			return NULL;
		source = VectorBuildLeafSource;
		source_arg = region->build_scan;
	}
	else if (IsA(outer, MotionState) || IsA(outer, CustomScanState))
	{
		region->build_row_leaf = MemoryContextAllocZero(
			region->css.ss.ps.state->es_query_cxt,
			sizeof(VectorRowLeafState));
		region->build_row_leaf->input = outer;
		region->build_row_leaf->batch = VectorBatchCreate(
			region->css.ss.ps.state->es_query_cxt,
			ExecGetResultType(outer),
			VECTOR_BATCH_DEFAULT_CAPACITY,
			false);
		source = VectorRowLeafSource;
		source_arg = region->build_row_leaf;
	}
	else
		return NULL;
	vhj = VectorHashJoinCreateBatch(region->css.ss.ps.state->es_query_cxt,
									hj, source, source_arg, outer_join);
	if (vhj == NULL)
	{
		VectorHashJoinDestroy(outer_join);
		VectorBuildScanDestroy(region->build_scan);
		region->build_scan = NULL;
		VectorRowLeafDestroy(region->build_row_leaf);
		region->build_row_leaf = NULL;
	}
	return vhj;
}

static void
VectorRegionInitBuildChain(VectorRegionState *state)
{
	PlanState  *hash = innerPlanState(state->hj);
	PlanState  *child;
	VectorHashJoinState *build;

	if (!IsA(hash, HashState))
		return;
	child = outerPlanState(hash);
	if (IsA(child, HashJoinState))
	{
		build = VectorRegionCreateBuildChain(state, (HashJoinState *) child);
		if (build == NULL)
		{
			/* the leaf may have been created before a deeper failure */
			VectorBuildScanDestroy(state->build_scan);
			state->build_scan = NULL;
			VectorRowLeafDestroy(state->build_row_leaf);
			state->build_row_leaf = NULL;
			VectorMotionLeafDestroy(state->build_motion_leaf);
			state->build_motion_leaf = NULL;
			return;
		}
		VectorHashJoinSetBuildSource(state->vhj,
									 VectorBuildChainSourceNext, build);
		state->build_vhj = build;
	}
	else if (IsA(child, MotionState))
	{
		/*
		 * The build side is a Motion receiver: use the adaptive leaf, the
		 * same one the probe chain uses.  A batch interconnect stream hands
		 * over as batches (the wire's columns are the motion's output as-is,
		 * which VectorHashBuildFromBatches reads directly); a plain tuple
		 * stream falls back to the embedded row adapter.
		 */
		state->build_motion_leaf = MemoryContextAllocZero(
			state->css.ss.ps.state->es_query_cxt,
			sizeof(VectorMotionLeafState));
		state->build_motion_leaf->motion = (MotionState *) child;
		state->build_motion_leaf->row.input = child;
		state->build_motion_leaf->row.batch = VectorBatchCreate(
			state->css.ss.ps.state->es_query_cxt,
			ExecGetResultType(child),
			VECTOR_BATCH_DEFAULT_CAPACITY,
			false);
		VectorHashJoinSetBuildSource(state->vhj,
									 VectorMotionLeafSource,
									 state->build_motion_leaf);
	}
	else if (IsA(child, CustomScanState) &&
			 ((CustomScanState *) child)->methods == &vector_region_exec_methods &&
			 VectorRegionOutputBatchable((VectorRegionState *) child))
	{
		/*
		 * The build side is a region that produces output batches
		 * natively: feed the hash build straight from them instead of
		 * MultiExecHash pulling the region's row output one slot at a
		 * time.  The batchable gate must match VectorRegionPullOutput-
		 * Batch's own decline conditions, or a mid-stream decline would
		 * end the feed as an empty build.
		 */
		VectorHashJoinSetBuildSource(state->vhj,
									 VectorRegionBuildSourceNext, child);
	}
}

static VectorBatch *
VectorRegionInputBatch(VectorRegionState *state)
{
	return state->nested_batches ? state->join_batch : state->batch;
}

static bool
VectorRegionNextBatch(VectorRegionState *state)
{
	if (state->nested_batches)
	{
		if (state->join_position < state->join_batch->nselected)
			return true;
		/*
		 * Reset the cursor only after a successful pull: the join chain has
		 * reached EOF, and a stale zero cursor would replay the final batch
		 * once per output row.
		 */
		if (!VectorHashJoinNextBatch(state->vhj))
			return false;
		state->join_position = 0;
		return true;
	}
	return VectorRegionNextScanBatch(state);
}

static bool
VectorRegionNextScanBatch(VectorRegionState *state)
{
	CustomScanState *node = &state->css;

	/*
	 * A batch already in hand must be finished first: the row-wise consumers
	 * (the hash join driver) call this per row, and the adapter path must
	 * not pull fresh input while rows of the current batch remain.
	 */
	if (state->position < state->batch->nselected)
		return true;

	if (state->input != NULL)
		return VectorRegionReadAdapter(state);

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
				bool		got;
				Instrumentation *instr = state->scan->ss.ps.instrument;

				/* the region drives the leaf directly, so ExecProcNode's
				 * instrumentation never runs; approximate it per batch */
				if (instr)
					InstrStartNode(instr);
				got = aocs_getnext_batch((AOCSScanDesc) scandesc, fill);
				if (instr)
					InstrStopNode(instr, got ? (uint64) fill->nrows : 0);
				if (!got)
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
VectorRegionSortInput(VectorRegionState *state, const int *sort_attnos)
{
	ScanDirection dir = state->css.ss.ps.state->es_direction;
	VectorBatch *input = VectorRegionInputBatch(state);
	int		   *cursor = state->nested_batches ? &state->join_position : &state->position;

	if (state->sort->sort_Done)
		return;
	state->css.ss.ps.state->es_direction = ForwardScanDirection;
	while (VectorRegionNextBatch(state))
	{
		VectorSortPutBatch(state->sort, input, sort_attnos);
		*cursor = input->nselected;
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
		slot->tts_values[col] = VectorBatchGetDatum(state->batch, attno, row);
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
		slot->tts_values[col] = VectorBatchGetDatum(state->batch, attno, row);
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

	if (state->hj != NULL && !state->nested_batches)
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
		VectorRegionSortInput(state, state->output_attnos);
		state->sort->ss.ps.ExecProcNode = VectorSortExec;
		return VectorMergeJoinNext(state->mergejoin);
	}

	if (state->unique != NULL)
	{
		VectorRegionSortInput(state, state->output_attnos);
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

	if (state->sort != NULL && state->sort->randomAccess)
	{
		VectorRegionSortInput(state, state->output_attnos);
		return VectorSortNext(state->sort);
	}

	if (state->agg != NULL)
	{
		AggState   *agg = state->agg;
		VectorBatch *input = VectorRegionInputBatch(state);
		int		   *cursor = state->nested_batches ? &state->join_position : &state->position;
		/* the region drives the aggregate directly, so ExecProcNode's
		 * instrumentation never runs; approximate it per output row */
		Instrumentation *instr = agg->ss.ps.instrument;

		if (instr)
			InstrStartNode(instr);

		if (state->sort != NULL)
		{
			/*
			 * A native sort sits between the join chain and the aggregate:
			 * drain the chain into it once, then feed the aggregate sorted
			 * batches (the sorted-aggregate machinery needs group-contiguous
			 * input).
			 */
			VectorRegionSortInput(state, state->sort_attnos);
			for (;;)
			{
				if (*cursor >= input->nselected)
				{
					if (!VectorSortNextBatch(state->sort, input,
											 state->sort_attnos))
						break;
					*cursor = 0;
				}
				if (agg->aggstrategy == AGG_SORTED)
				{
					TupleTableSlot *result = VectorAggProcessBatchSorted(state->vagg,
																		 input,
																		 cursor);

					if (result != NULL)
					{
						if (instr)
							InstrStopNode(instr, 1);
						return result;
					}
				}
				else
				{
					VectorAggProcessBatch(state->vagg, input);
					*cursor = input->nselected;
				}
			}
		}
		else if (agg->aggstrategy == AGG_SORTED)
		{
			while (!agg->input_done &&
				   (*cursor < input->nselected ||
					VectorRegionNextBatch(state)))
			{
				TupleTableSlot *result = VectorAggProcessBatchSorted(state->vagg,
															 input,
															 cursor);

				if (result != NULL)
				{
					if (instr)
						InstrStopNode(instr, 1);
					return result;
				}
			}
		}
		else
		{
			while (VectorRegionNextBatch(state))
			{
				VectorAggProcessBatch(state->vagg, input);
				*cursor = input->nselected;
			}
		}
		VectorAggFinish(state->vagg);
		{
			TupleTableSlot *result = VectorAggNext(state->vagg);

			if (instr)
				InstrStopNode(instr, result ? 1 : 0);
			return result;
		}
	}

	if (state->limit != NULL && VectorLimitStart(state->limit))
	{
		ExecClearTuple(slot);
		return slot;
	}

	for (;;)
	{
		VectorBatch *out_batch;
		int		   *cursor;

		ExecClearTuple(slot);
		if (state->sort != NULL)
		{
			if (state->position >= state->batch->nselected)
			{
				VectorRegionSortInput(state, state->output_attnos);
				if (!VectorSortNextBatch(state->sort, state->batch,
										 state->output_attnos))
					return slot;
				state->position = 0;
			}
			out_batch = state->batch;
			cursor = &state->position;
		}
		else
		{
			if (!VectorRegionNextBatch(state))
				return slot;
			/*
			 * A batch-driven HJ-root region outputs the join batches
			 * themselves; everything else outputs the scan/adapter batch.
			 */
			out_batch = state->nested_batches ? state->join_batch : state->batch;
			cursor = state->nested_batches ? &state->join_position : &state->position;
		}

		/* evaluate projection steps once per batch, after the qual filter */
		if (state->proj != NULL)
		{
			for (col = 0; col < state->output_ncols; col++)
				if (state->proj[col] != NULL)
					VectorExprEval(out_batch, state->proj[col]);
		}

		while (*cursor < out_batch->nselected)
		{
			row = VectorBatchSelectedRow(out_batch, (*cursor)++);
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
					slot->tts_values[col] = VectorBatchGetDatum(out_batch, attno, row);
					slot->tts_isnull[col] = VectorBatchIsNull(out_batch, attno, row);
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
	if (state->build_vhj != NULL)
		VectorHashJoinDestroy(state->build_vhj);
	VectorBuildScanDestroy(state->build_scan);
	VectorRowLeafDestroy(state->row_leaf);
	VectorRowLeafDestroy(state->build_row_leaf);
	VectorMotionLeafDestroy(state->build_motion_leaf);
	if (state->motion_leaf != NULL)
		VectorMotionLeafDestroy(state->motion_leaf);
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
	if (state->sort_attnos != NULL)
		pfree(state->sort_attnos);
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
	state->join_position = 0;
	state->eof = false;
	if (state->nestloop != NULL)
		outerPlanState(state->nestloop) = state->nl_outer;
	if (state->material != NULL)
		outerPlanState(state->material) = state->mat_outer;
	if (state->batch != NULL)
		VectorBatchReset(state->batch);
	if (state->scan != NULL && state->input == NULL &&
		state->scan->ss.ss_currentScanDesc != NULL)
	{
		/*
		 * A rescan must re-read the leaf from the start.  The NestLoop
		 * prefetch path (any motion below the join makes it fetch one inner
		 * row under a fake outer and rescan immediately) reaches here with
		 * the current batch only partially consumed; resetting the batch
		 * cursor alone would silently drop the rest of the batch because
		 * the table scan keeps its position.  Close the scan descriptor:
		 * the next fill re-opens it from the first row.
		 */
		table_endscan(state->scan->ss.ss_currentScanDesc);
		state->scan->ss.ss_currentScanDesc = NULL;
	}
	if (state->material != NULL &&
		state->material->tuplestorestate != NULL &&
		!state->material->ts_destroyed)
	{
		/* rows cached by the previous scan iteration must not leak in */
		tuplestore_clear(state->material->tuplestorestate);
		state->material->eof_underlying = false;
	}
	if (state->sort != NULL)
	{
		/* same reset as ExecReScanSort, without recursing into the chain */
		VectorSortReset(state->sort);
		state->sort->sort_Done = false;
		if (state->sort->tuplesortstate != NULL)
		{
			tuplesort_end((Tuplesortstate *) state->sort->tuplesortstate);
			state->sort->tuplesortstate = NULL;
		}
		ExecClearTuple(state->sort->ss.ps.ps_ResultTupleSlot);
	}
	if (state->vhj != NULL)
		VectorHashJoinReset(state->vhj);
	if (state->row_leaf != NULL)
		state->row_leaf->eof = false;
	if (state->build_vhj != NULL)
	{
		VectorHashJoinReset(state->build_vhj);
		if (state->build_scan != NULL)
			state->build_scan->eof = false;
	}
	if (state->motion_leaf != NULL)
	{
		/*
		 * The adaptive Motion leaf must decide again on the first pull of
		 * the re-run chain: a stale EOS decision or a handed-out batch would
		 * make every execution after the first read an empty stream.
		 */
		state->motion_leaf->decided = 0;
		if (state->motion_leaf->current != NULL)
		{
			VectorBatchDestroy(state->motion_leaf->current);
			state->motion_leaf->current = NULL;
		}
		state->motion_leaf->row.eof = false;
	}
	if (state->build_row_leaf != NULL)
		state->build_row_leaf->eof = false;
	if (state->build_motion_leaf != NULL)
	{
		/*
		 * Same re-decision rule as the probe motion leaf: a stale EOS
		 * decision or a handed-out batch would make every execution after
		 * the first read an empty build.
		 */
		state->build_motion_leaf->decided = 0;
		if (state->build_motion_leaf->current != NULL)
		{
			VectorBatchDestroy(state->build_motion_leaf->current);
			state->build_motion_leaf->current = NULL;
		}
		state->build_motion_leaf->row.eof = false;
	}
	if (state->vagg != NULL)
		VectorAggReset(state->vagg);
	else if (state->vhj == NULL)
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
		/* rewind every child scan, not only the one current at rescan time */
		for (int i = 0; i < state->append->as_nplans; i++)
		{
			SeqScanState *child = (SeqScanState *) state->append->appendplans[i];

			if (child->ss.ss_currentScanDesc != NULL)
			{
				table_endscan(child->ss.ss_currentScanDesc);
				child->ss.ss_currentScanDesc = NULL;
			}
		}
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
		VectorSortMarkPos(state->sort);
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
		VectorSortRestrPos(state->sort);
	else if (state->material != NULL)
		ExecMaterialRestrPos(state->material);
	else
		elog(ERROR, "vector region supports RestrPos only for Sort or Material");
}

/*
 * Sender-side batch interconnect support: when this plan node is a
 * vector region whose output rows are produced natively as batches (a
 * plain scan region or a batch-driven HJ-root region, with no computed
 * projection, limit or packed layout between the batch and the
 * region's output), hand the next output batch out directly.  Returns
 * false when the node is not such a region; *batch == NULL marks EOF.
 */
bool
VectorRegionPullOutputBatch(PlanState *ps, VectorBatch **batch)
{
	VectorRegionState *state;
	int			col;

	if (!IsA(ps, CustomScanState) ||
		((CustomScanState *) ps)->methods != &vector_region_exec_methods)
		return false;
	state = (VectorRegionState *) ps;

	/* an aggregate or sort consumes the join batches: row-wise output */
	if (state->agg != NULL || state->sort != NULL)
		return false;

	if (state->nested_batches)
	{
		/* HJ-root region: join output batches are the region output. */
		if (VectorHashJoinNextBatch(state->vhj))
		{
			*batch = state->join_batch;
			return true;
		}
		*batch = NULL;
		return true;
	}

	if (state->hj != NULL || state->nestloop != NULL ||
		state->mergejoin != NULL || state->material != NULL ||
		state->unique != NULL ||
		state->scan == NULL || state->append != NULL ||
		state->limit != NULL || state->packed_i32 ||
		state->motion_leaf != NULL || state->row_leaf != NULL)
		return false;

	/* computed projections do not materialize into batch columns */
	if (state->proj != NULL)
		return false;

	/* the motion output columns must be the batch columns as-is; a batch
	 * wider than the output is a fine prefix, but an output column with no
	 * batch column behind it (e.g. a void column from a nested-loop inner)
	 * must fall back to the row-wise output */
	if (state->output_ncols > state->batch->ncols)
		return false;
	for (col = 0; col < state->output_ncols; col++)
	{
		if (state->output_attnos[col] != col)
			return false;
	}

	while (VectorRegionNextScanBatch(state))
	{
		if (state->batch->nselected > 0)
		{
			/*
			 * The motion encodes the batch before its next pull, so the
			 * batch is consumed here like the join-chain leaf does;
			 * without this NextScanBatch would keep returning it.
			 */
			state->position = state->batch->nselected;
			*batch = state->batch;
			return true;
		}
	}
	*batch = NULL;
	return true;
}
