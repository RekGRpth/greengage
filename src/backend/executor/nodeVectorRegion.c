/*-------------------------------------------------------------------------
 *
 * nodeVectorRegion.c
 *	  Executor node for a segment-local vectorized plan region.
 *
 * The initial region reads a simple AOCS SeqScan into column batches and
 * materializes one virtual tuple per selected row at the region boundary.
 * The SeqScan child retains relation initialization and EXPLAIN structure,
 * but its row-at-a-time access method is not executed.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "access/tableam.h"
#include "cdb/cdbaocsam.h"
#include "executor/executor.h"
#include "executor/nodeCustom.h"
#include "executor/nodeSeqscan.h"
#include "executor/nodeVectorRegion.h"
#include "executor/vectorBatch.h"
#include "nodes/extensible.h"
#include "utils/rel.h"

#define VECTOR_REGION_NAME "GGVectorRegion"

typedef struct VectorRegionState
{
	CustomScanState css;
	SeqScanState *scan;
	VectorBatch *batch;
	int		   *output_attnos;
	int			output_ncols;
	int			position;
	bool		eof;
} VectorRegionState;

static Node *VectorRegionCreateState(CustomScan *cscan);
static void VectorRegionBegin(CustomScanState *node, EState *estate, int eflags);
static TupleTableSlot *VectorRegionExec(CustomScanState *node);
static void VectorRegionEnd(CustomScanState *node);
static void VectorRegionRescan(CustomScanState *node);

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
	NULL,						/* MarkPos */
	NULL,						/* RestrPos */
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
	SeqScan	   *scan;
	Relation	relation;
	ListCell   *lc;
	int			col = 0;

	foreach(lc, cscan->custom_plans)
	{
		Plan	   *plan = (Plan *) lfirst(lc);
		PlanState  *child = ExecInitNode(plan, estate,
									 eflags & ~(EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK));

		node->custom_ps = lappend(node->custom_ps, child);
	}

	if (list_length(node->custom_ps) != 1 ||
		!IsA(linitial(node->custom_ps), SeqScanState))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("vector region expects one AOCS SeqScan leaf")));

	state->scan = (SeqScanState *) linitial(node->custom_ps);
	scan = (SeqScan *) state->scan->ss.ps.plan;
	relation = state->scan->ss.ss_currentRelation;
	if (!RelationIsAoCols(relation))
		elog(ERROR, "vector region leaf is not an AOCS relation");

	state->output_ncols = list_length(scan->plan.targetlist);
	if (state->output_ncols != node->ss.ss_ScanTupleSlot->tts_tupleDescriptor->natts)
		elog(ERROR, "vector region output shape does not match its scan");

	state->output_attnos = MemoryContextAlloc(estate->es_query_cxt,
										 sizeof(int) * state->output_ncols);
	foreach(lc, scan->plan.targetlist)
	{
		TargetEntry *te = (TargetEntry *) lfirst(lc);
		Var		   *var;

		if (!IsA(te->expr, Var))
			elog(ERROR, "vector region supports only plain column projections");
		var = (Var *) te->expr;
		if (var->varattno <= 0 || var->varattno > RelationGetNumberOfAttributes(relation))
			elog(ERROR, "vector region encountered an invalid output attribute");
		state->output_attnos[col++] = var->varattno - 1;
	}

	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		state->batch = VectorBatchCreate(estate->es_query_cxt,
										 RelationGetDescr(relation),
										 VECTOR_BATCH_DEFAULT_CAPACITY);
}

static TupleTableSlot *
VectorRegionExec(CustomScanState *node)
{
	VectorRegionState *state = (VectorRegionState *) node;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	int			row;
	int			col;

	ExecClearTuple(slot);
	if (state->eof)
		return slot;

	if (state->position >= state->batch->nselected)
	{
		TableScanDesc scandesc = state->scan->ss.ss_currentScanDesc;

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

		if (!aocs_getnext_batch((AOCSScanDesc) scandesc, state->batch))
		{
			state->eof = true;
			return slot;
		}
		state->position = 0;
	}

	row = VectorBatchSelectedRow(state->batch, state->position++);
	for (col = 0; col < state->output_ncols; col++)
	{
		int			attno = state->output_attnos[col];

		slot->tts_values[col] = state->batch->columns[attno].values[row];
		slot->tts_isnull[col] = VectorBatchIsNull(state->batch, attno, row);
	}
	slot->tts_nvalid = state->output_ncols;
	return ExecStoreVirtualTuple(slot);
}

static void
VectorRegionEnd(CustomScanState *node)
{
	VectorRegionState *state = (VectorRegionState *) node;
	ListCell   *lc;

	if (state->batch != NULL)
		VectorBatchDestroy(state->batch);
	if (state->output_attnos != NULL)
		pfree(state->output_attnos);

	foreach(lc, node->custom_ps)
		ExecEndNode((PlanState *) lfirst(lc));
	node->custom_ps = NIL;
}

static void
VectorRegionRescan(CustomScanState *node)
{
	VectorRegionState *state = (VectorRegionState *) node;
	ListCell   *lc;

	state->position = 0;
	state->eof = false;
	if (state->batch != NULL)
		VectorBatchReset(state->batch);
	foreach(lc, node->custom_ps)
		ExecReScan((PlanState *) lfirst(lc));
}
