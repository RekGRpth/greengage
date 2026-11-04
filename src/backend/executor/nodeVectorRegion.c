/*-------------------------------------------------------------------------
 *
 * nodeVectorRegion.c
 *	  Executor node for a segment-local vectorized plan region.
 *
 * Region discovery is deliberately added separately from this executor
 * skeleton.  Until an operator has a batch implementation, no plan is
 * replaced by a vector region.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "executor/executor.h"
#include "executor/nodeCustom.h"
#include "executor/nodeVectorRegion.h"
#include "nodes/extensible.h"

#define VECTOR_REGION_NAME "GGVectorRegion"

typedef struct VectorRegionState
{
	CustomScanState css;
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
	CustomScan *cscan = (CustomScan *) node->ss.ps.plan;
	ListCell   *lc;

	/* Keep the boundary plan nodes visible to EXPLAIN and instrumentation. */
	foreach(lc, cscan->custom_plans)
	{
		Plan	   *plan = (Plan *) lfirst(lc);
		PlanState  *child = ExecInitNode(plan, estate,
									 eflags & ~(EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK));

		node->custom_ps = lappend(node->custom_ps, child);
	}
}

static TupleTableSlot *
VectorRegionExec(CustomScanState *node)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("vectorized region has no executable operators")));
	return NULL;
}

static void
VectorRegionEnd(CustomScanState *node)
{
	ListCell   *lc;

	foreach(lc, node->custom_ps)
		ExecEndNode((PlanState *) lfirst(lc));
	node->custom_ps = NIL;
}

static void
VectorRegionRescan(CustomScanState *node)
{
	ListCell   *lc;

	foreach(lc, node->custom_ps)
		ExecReScan((PlanState *) lfirst(lc));
}
