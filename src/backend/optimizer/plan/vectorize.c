/*-------------------------------------------------------------------------
 *
 * vectorize.c
 *	  Replace eligible segment-local plan regions with vectorized regions.
 *
 * The pass runs after either optimizer has produced a complete PlannedStmt.
 * The initial implementation covers simple AOCS scans in force mode.  The
 * cost-gated auto mode is enabled together with vectorized operators that
 * provide a benefit over row-at-a-time execution.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "catalog/pg_am.h"
#include "cdb/cdbllize.h"
#include "cdb/cdbplan.h"
#include "cdb/cdbvars.h"
#include "executor/nodeVectorRegion.h"
#include "executor/vectorAgg.h"
#include "executor/vectorExpr.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/planner.h"
#include "optimizer/walkers.h"
#include "parser/parsetree.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"

typedef struct VectorizeContext
{
	plan_tree_base_prefix base;
	PlannedStmt *stmt;
	int			slice;
	int			next_plan_node_id;
	int			wrapped;
} VectorizeContext;

static Node *vectorize_mutator(Node *node, VectorizeContext *ctx);

static bool
max_plan_node_id_walker(Node *node, VectorizeContext *ctx)
{
	if (node == NULL)
		return false;
	if (is_plan_node(node))
	{
		int			id = ((Plan *) node)->plan_node_id;

		if (id >= ctx->next_plan_node_id)
			ctx->next_plan_node_id = id + 1;
	}
	return plan_tree_walker(node, max_plan_node_id_walker, ctx, true);
}

static bool
slice_runs_on_segments(VectorizeContext *ctx)
{
	PlanSlice  *slice;

	if (ctx->stmt->numSlices <= 0 || ctx->slice < 0 ||
		ctx->slice >= ctx->stmt->numSlices)
		return false;
	slice = &ctx->stmt->slices[ctx->slice];
	return slice->gangType == GANGTYPE_PRIMARY_READER ||
		slice->gangType == GANGTYPE_PRIMARY_WRITER ||
		slice->gangType == GANGTYPE_SINGLETON_READER;
}

static bool
is_simple_aoco_scan(VectorizeContext *ctx, SeqScan *scan)
{
	RangeTblEntry *rte;
	ListCell   *lc;
	VectorExprVarMap vars = {scan->scanrelid, NULL, 0};

	if (scan->scanrelid <= 0 ||
		scan->plan.lefttree != NULL || scan->plan.righttree != NULL ||
		scan->plan.initPlan != NIL || scan->plan.extParam != NULL ||
		scan->plan.allParam != NULL)
		return false;

	rte = rt_fetch(scan->scanrelid, ctx->stmt->rtable);
	if (rte->rtekind != RTE_RELATION ||
		get_rel_am(rte->relid) != AO_COLUMN_TABLE_AM_OID)
		return false;
	if (!VectorExprIsSupported((Expr *) scan->plan.qual, &vars))
		return false;

	foreach(lc, scan->plan.targetlist)
	{
		TargetEntry *te = (TargetEntry *) lfirst(lc);
		Var		   *var;

		if (te->resjunk || !IsA(te->expr, Var))
			return false;
		var = (Var *) te->expr;
		if (var->varno != scan->scanrelid || var->varattno <= 0 ||
			var->varlevelsup != 0)
			return false;
	}
	return true;
}

/*
 * Build a scan-only region.  The original SeqScan remains a custom child so
 * executor initialization keeps its relation lock and EXPLAIN shows the leaf.
 */
static Plan *
make_region(VectorizeContext *ctx, Plan *root)
{
	List	   *tlist = root->targetlist;
	List	   *colnames = NIL;
	List	   *coltypes = NIL;
	List	   *coltypmods = NIL;
	List	   *colcollations = NIL;
	List	   *scan_tlist = NIL;
	List	   *out_tlist = NIL;
	RangeTblEntry *rte;
	CustomScan *cs;
	ListCell   *lc;
	int			rteidx = list_length(ctx->stmt->rtable) + 1;
	int			k = 1;

	foreach(lc, tlist)
	{
		TargetEntry *te = (TargetEntry *) lfirst(lc);
		Node	   *expr = (Node *) te->expr;
		char	   *name = te->resname ? pstrdup(te->resname) : psprintf("c%d", k);
		Var		   *scanvar = makeVar(rteidx, k, exprType(expr), exprTypmod(expr),
									exprCollation(expr), 0);
		Var		   *outvar = makeVar(INDEX_VAR, k, exprType(expr), exprTypmod(expr),
								   exprCollation(expr), 0);

		colnames = lappend(colnames, makeString(name));
		coltypes = lappend_oid(coltypes, exprType(expr));
		coltypmods = lappend_int(coltypmods, exprTypmod(expr));
		colcollations = lappend_oid(colcollations, exprCollation(expr));
		scan_tlist = lappend(scan_tlist,
							 makeTargetEntry((Expr *) scanvar, k, name, te->resjunk));
		out_tlist = lappend(out_tlist,
							makeTargetEntry((Expr *) outvar, k, name, te->resjunk));
		k++;
	}

	rte = makeNode(RangeTblEntry);
	rte->rtekind = RTE_NAMEDTUPLESTORE;
	rte->enrname = "gg_vector_region";
	rte->enrtuples = root->plan_rows;
	rte->eref = makeAlias("gg_vector_region", colnames);
	rte->coltypes = coltypes;
	rte->coltypmods = coltypmods;
	rte->colcollations = colcollations;
	rte->inFromCl = false;
	rte->lateral = false;
	rte->inh = false;
	rte->requiredPerms = 0;
	ctx->stmt->rtable = lappend(ctx->stmt->rtable, rte);

	cs = makeNode(CustomScan);
	cs->custom_plans = list_make1(root);
	cs->custom_scan_tlist = scan_tlist;
	cs->custom_relids = bms_make_singleton(rteidx);
	cs->methods = GetVectorRegionScanMethods();
	cs->scan.scanrelid = 0;
	cs->scan.plan.targetlist = out_tlist;
	cs->scan.plan.startup_cost = root->startup_cost;
	cs->scan.plan.total_cost = root->total_cost;
	cs->scan.plan.plan_rows = root->plan_rows;
	cs->scan.plan.plan_width = root->plan_width;
	cs->scan.plan.plan_node_id = ctx->next_plan_node_id++;
	cs->scan.plan.parallel_aware = root->parallel_aware;
	cs->scan.plan.parallel_safe = root->parallel_safe;
	cs->scan.plan.flow = root->flow;

	ctx->wrapped++;
	return (Plan *) cs;
}

static Node *
vectorize_mutator(Node *node, VectorizeContext *ctx)
{
	if (node == NULL)
		return NULL;

	if (IsA(node, Motion))
	{
		Motion	   *motion = (Motion *) node;
		int			saved_slice = ctx->slice;
		Node	   *result;

		ctx->slice = motion->motionID;
		result = plan_tree_mutator(node, vectorize_mutator, ctx, false);
		ctx->slice = saved_slice;
		return result;
	}

	if (IsA(node, Agg) && slice_runs_on_segments(ctx) &&
		VectorAggPlanIsSupported((Agg *) node) &&
		(!IsA(outerPlan(node), SeqScan) ||
		 is_simple_aoco_scan(ctx, (SeqScan *) outerPlan(node))))
		return (Node *) make_region(ctx, (Plan *) node);

	if (IsA(node, SeqScan) && slice_runs_on_segments(ctx) &&
		((SeqScan *) node)->plan.targetlist != NIL &&
		is_simple_aoco_scan(ctx, (SeqScan *) node))
		return (Node *) make_region(ctx, (Plan *) node);

	return plan_tree_mutator(node, vectorize_mutator, ctx, false);
}

PlannedStmt *
VectorizePlan(PlannedStmt *stmt)
{
	VectorizeContext ctx;

	if (gp_vectorized_execution != GP_VECTORIZED_EXECUTION_FORCE ||
		Gp_role != GP_ROLE_DISPATCH ||
		stmt == NULL || stmt->commandType != CMD_SELECT ||
		stmt->planTree == NULL)
		return stmt;

	memset(&ctx, 0, sizeof(ctx));
	exec_init_plan_tree_base(&ctx.base, stmt);
	ctx.stmt = stmt;
	ctx.next_plan_node_id = 1;
	max_plan_node_id_walker((Node *) stmt->planTree, &ctx);
	stmt->planTree = (Plan *) vectorize_mutator((Node *) stmt->planTree, &ctx);

	if (ctx.wrapped > 0)
		elog(DEBUG1, "vectorized execution: %d AOCS scan region(s) planned",
			 ctx.wrapped);
	return stmt;
}
