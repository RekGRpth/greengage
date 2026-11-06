/*-------------------------------------------------------------------------
 *
 * vectorize.c
 *	  Replace eligible segment-local plan regions with vectorized regions.
 *
 * The pass runs after either optimizer has produced a complete PlannedStmt.
 * The initial implementation covers simple AOCS scans in force mode.  Heap
 * and AO row scans use the same regions through a raw-slot batch reader.  The
 * cost-gated auto mode is enabled together with vectorized operators that
 * provide a benefit over row-at-a-time execution.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "cdb/cdbllize.h"
#include "cdb/cdbplan.h"
#include "cdb/cdbvars.h"
#include "cdb/memquota.h"
#include "executor/nodeVectorRegion.h"
#include "executor/nodeLimit.h"
#include "executor/nodeMergejoin.h"
#include "executor/nodeSort.h"
#include "executor/nodeUnique.h"
#include "executor/vectorAgg.h"
#include "executor/vectorExpr.h"
#include "executor/vectorHashJoin.h"
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
is_simple_scan(VectorizeContext *ctx, SeqScan *scan)
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
	if (rte->rtekind != RTE_RELATION)
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
 * A scan that a region may wrap as its own root: like is_simple_scan,
 * but the targetlist may also carry projection expressions (ORCA keeps them
 * in the scan's targetlist) and junk entries (sort keys of an ORDER BY
 * above the region).
 */
static bool
is_projection_scan(VectorizeContext *ctx, SeqScan *scan)
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
	if (rte->rtekind != RTE_RELATION)
		return false;
	if (!VectorExprIsSupported((Expr *) scan->plan.qual, &vars))
		return false;

	foreach(lc, scan->plan.targetlist)
	{
		TargetEntry *te = (TargetEntry *) lfirst(lc);

		if (!VectorExprIsSupported(te->expr, &vars))
			return false;
	}
	return true;
}

static const char *
plan_node_name(Plan *plan)
{
	switch (nodeTag(plan))
	{
		case T_Agg: return "Agg";
		case T_HashJoin: return "HashJoin";
		case T_Sort: return "Sort";
		case T_Limit: return "Limit";
		case T_Unique: return "Unique";
		case T_MergeJoin: return "MergeJoin";
		case T_SeqScan: return "SeqScan";
		default: return "plan node";
	}
}

static void
decision(VectorizeContext *ctx, Plan *plan, const char *reason)
{
	if (!gp_vectorized_explain_decisions)
		return;
	elog(DEBUG1, "vectorized execution: %s (plan node %d): %s",
		 plan_node_name(plan), plan->plan_node_id, reason);
}

static double
input_rows(Plan *plan)
{
	if (IsA(plan, Agg))
		return outerPlan(plan)->plan_rows;
	if (IsA(plan, HashJoin))
		return outerPlan(plan)->plan_rows + innerPlan(plan)->plan_rows;
	return plan->plan_rows;
}

static double
conversion_cost(Plan *plan)
{
	double		cost = 0;

	/*
	 * The row-to-batch adapter converts every value crossing its boundary.
	 * A scan region has no boundary to cross: it fills its own batches, and
	 * measured batch reads beat row scans on by-value data.
	 */
	if (IsA(plan, Agg) && !IsA(outerPlan(plan), SeqScan))
		cost += input_rows(plan) * list_length(outerPlan(plan)->targetlist) *
			(17.0 / 15000.0);
	return cost;
}

static double
workset_kb(Plan *plan)
{
	if (IsA(plan, Agg) && ((Agg *) plan)->aggstrategy == AGG_HASHED)
	{
		Agg		   *agg = (Agg *) plan;
		double		bytes = Max(agg->numGroups, 1) *
			(outerPlan(agg)->plan_width +
			 list_length(agg->plan.targetlist) * sizeof(AggStatePerGroupData));

		return bytes * 2.0 / 1024.0;
	}
	if (IsA(plan, HashJoin))
	{
		Plan	   *inner = outerPlan(innerPlan(plan));

		return inner->plan_rows * inner->plan_width * 2.0 / 1024.0;
	}
	return 0;
}

static uint64
memory_quota_kb(Plan *plan)
{
	if (IsA(plan, HashJoin))
		return innerPlan(plan)->operatorMemKB;
	if (IsA(plan, Unique) || IsA(plan, MergeJoin))
		return outerPlan(plan)->operatorMemKB;
	return plan->operatorMemKB;
}

/*
 * Measured on TPC-H SF1 (6M-row AOCO lineitem, 3 primaries, -O0 build):
 * regions over pass-by-value columns ran 13-18x faster than the row
 * executor, while every referenced pass-by-reference column pays a
 * datumCopy per row on the way into a batch and made numeric-heavy
 * queries ~15% slower.  Until by-ref columns get cheap handling, auto
 * only accepts leaves whose referenced columns are all by-value.
 */
static bool
leaf_has_byref(Plan *leaf)
{
	ListCell   *lc;

	if (leaf == NULL)
		return false;
	foreach(lc, leaf->targetlist)
	{
		TargetEntry *te = (TargetEntry *) lfirst(lc);
		int16		typlen;
		bool		typbyval;

		get_typlenbyval(exprType((Node *) te->expr), &typlen, &typbyval);
		if (!typbyval)
			return true;
	}
	return false;
}

static bool
auto_allowed(VectorizeContext *ctx, Plan *plan)
{
	double		pg_cost;
	double		vector_cost;
	double		quota;

	if (gp_vectorized_execution == GP_VECTORIZED_EXECUTION_FORCE)
		return true;
	/*
	 * A bare scan is worth wrapping under auto only when its targetlist
	 * carries projection expressions: the batch scan alone beats the row
	 * scan on pass-by-value data, but the planner already delivers plain
	 * columns row-wise just as fast as we can materialize them, while a
	 * projection evaluates once per batch instead of once per row.
	 */
	if (IsA(plan, SeqScan))
	{
		ListCell   *lc;
		bool		has_projection = false;

		foreach(lc, plan->targetlist)
		{
			TargetEntry *te = (TargetEntry *) lfirst(lc);

			if (!IsA(te->expr, Var))
			{
				has_projection = true;
				break;
			}
		}
		if (!has_projection)
		{
			decision(ctx, plan, "auto mode does not wrap a plain scan");
			return false;
		}
	}
	else if (!IsA(plan, Agg) && !IsA(plan, HashJoin))
	{
		decision(ctx, plan, "auto mode does not wrap this node");
		return false;
	}
	if (input_rows(plan) < (double) gp_vectorized_min_rows)
	{
		decision(ctx, plan, "estimated input rows are below gp_vectorized_min_rows");
		return false;
	}
	if (leaf_has_byref(IsA(plan, SeqScan) ? plan : outerPlan(plan)) ||
		(IsA(plan, HashJoin) && leaf_has_byref(outerPlan(innerPlan(plan)))))
	{
		decision(ctx, plan, "the leaf reads pass-by-reference columns");
		return false;
	}

	quota = memory_quota_kb(plan);
	if (quota > 0 && workset_kb(plan) > quota)
	{
		decision(ctx, plan, "the estimated working set exceeds the operator memory quota");
		return false;
	}

	pg_cost = plan->total_cost;
	vector_cost = pg_cost * gp_vectorized_cost_op_factor + conversion_cost(plan);
	if (vector_cost * (1.0 + gp_vectorized_cost_margin) >= pg_cost)
	{
		decision(ctx, plan, "the cost gate prefers the standard executor");
		return false;
	}
	decision(ctx, plan, "vector region accepted by the cost gate");
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
	cs->scan.plan.operatorMemKB = memory_quota_kb(root);
	if ((IsA(root, Agg) && ((Agg *) root)->aggstrategy == AGG_HASHED) ||
		IsA(root, HashJoin) || IsA(root, Sort) || IsA(root, Unique) ||
		IsA(root, MergeJoin))
		cs->flags |= CUSTOMSCAN_GP_MEMORY_INTENSIVE | CUSTOMSCAN_GP_BLOCKING;

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

	/* Prefer a hashed aggregate below a plain partial/final aggregate. */
	if (IsA(node, Agg) && ((Agg *) node)->aggstrategy == AGG_PLAIN &&
		IsA(outerPlan(node), Agg))
		return plan_tree_mutator(node, vectorize_mutator, ctx, false);

	if (IsA(node, HashJoin) && slice_runs_on_segments(ctx) &&
		VectorHashJoinPlanIsSupported((HashJoin *) node) &&
		is_simple_scan(ctx, (SeqScan *) outerPlan(node)) &&
		/* the build side reads row-wise through the Hash node: any input */
		IsA(innerPlan(node), Hash) &&
		auto_allowed(ctx, (Plan *) node))
		return (Node *) make_region(ctx, (Plan *) node);

	if (IsA(node, MergeJoin) && slice_runs_on_segments(ctx) &&
		VectorMergeJoinPlanIsSupported((MergeJoin *) node) &&
		IsA(outerPlan(node), Sort) &&
		VectorSortPlanIsSupported((Sort *) outerPlan(node)) &&
		IsA(outerPlan(outerPlan(node)), SeqScan) &&
		is_simple_scan(ctx, (SeqScan *) outerPlan(outerPlan(node))) &&
		auto_allowed(ctx, (Plan *) node))
		return (Node *) make_region(ctx, (Plan *) node);

	if (IsA(node, Unique) && slice_runs_on_segments(ctx) &&
		VectorUniquePlanIsSupported((Unique *) node) &&
		IsA(outerPlan(node), Sort) &&
		VectorSortPlanIsSupported((Sort *) outerPlan(node)) &&
		IsA(outerPlan(outerPlan(node)), SeqScan) &&
		is_simple_scan(ctx, (SeqScan *) outerPlan(outerPlan(node))) &&
		auto_allowed(ctx, (Plan *) node))
		return (Node *) make_region(ctx, (Plan *) node);

	if (IsA(node, Limit) && slice_runs_on_segments(ctx) &&
		VectorLimitPlanIsSupported((Limit *) node) &&
		IsA(outerPlan(node), SeqScan) &&
		is_simple_scan(ctx, (SeqScan *) outerPlan(node)) &&
		auto_allowed(ctx, (Plan *) node))
		return (Node *) make_region(ctx, (Plan *) node);

	if (IsA(node, Sort) && slice_runs_on_segments(ctx) &&
		VectorSortPlanIsSupported((Sort *) node) &&
		is_simple_scan(ctx, (SeqScan *) outerPlan(node)) &&
		auto_allowed(ctx, (Plan *) node))
		return (Node *) make_region(ctx, (Plan *) node);

	if (IsA(node, Agg) && slice_runs_on_segments(ctx) &&
		VectorAggPlanIsSupported((Agg *) node) &&
		(!IsA(outerPlan(node), SeqScan) ||
		 is_simple_scan(ctx, (SeqScan *) outerPlan(node))) &&
		auto_allowed(ctx, (Plan *) node))
		return (Node *) make_region(ctx, (Plan *) node);

	if (IsA(node, SeqScan) && slice_runs_on_segments(ctx) &&
		((SeqScan *) node)->plan.targetlist != NIL &&
		is_projection_scan(ctx, (SeqScan *) node) &&
		auto_allowed(ctx, (Plan *) node))
		return (Node *) make_region(ctx, (Plan *) node);

	return plan_tree_mutator(node, vectorize_mutator, ctx, false);
}

PlannedStmt *
VectorizePlan(PlannedStmt *stmt)
{
	VectorizeContext ctx;

	if ((gp_vectorized_execution != GP_VECTORIZED_EXECUTION_FORCE &&
		  gp_vectorized_execution != GP_VECTORIZED_EXECUTION_AUTO) ||
		Gp_role != GP_ROLE_DISPATCH ||
		stmt == NULL || stmt->commandType != CMD_SELECT ||
		stmt->planTree == NULL)
		return stmt;

	memset(&ctx, 0, sizeof(ctx));
	exec_init_plan_tree_base(&ctx.base, stmt);
	ctx.stmt = stmt;
	ctx.next_plan_node_id = 1;
	max_plan_node_id_walker((Node *) stmt->planTree, &ctx);

	if (gp_vectorized_execution == GP_VECTORIZED_EXECUTION_AUTO)
	{
		uint64		bytes = (uint64) statement_mem * 1024L;

		if (IsResManagerMemoryPolicyAuto())
			PolicyAutoAssignOperatorMemoryKB(stmt, bytes);
		else if (IsResManagerMemoryPolicyEagerFree())
			PolicyEagerFreeAssignOperatorMemoryKB(stmt, bytes);
	}

	stmt->planTree = (Plan *) vectorize_mutator((Node *) stmt->planTree, &ctx);

	/*
	 * InitPlans and correlated subqueries are plans of their own, rooted in
	 * the slice subplan_sliceIds names (-1: never dispatched, nothing to
	 * do).  A subplan is held to its output order: its query's ORDER BY is
	 * not visible here.
	 */
	if (stmt->subplans != NIL)
	{
		ListCell   *lc;
		int			i = 0;

		foreach(lc, stmt->subplans)
		{
			Plan	   *sub = (Plan *) lfirst(lc);

			if (sub != NULL)
			{
				ctx.slice = stmt->subplan_sliceIds ?
					stmt->subplan_sliceIds[i] : -1;
				lfirst(lc) = vectorize_mutator((Node *) sub, &ctx);
			}
			i++;
		}
	}

	if (ctx.wrapped > 0)
		elog(DEBUG1, "vectorized execution: %d region(s) planned",
			 ctx.wrapped);
	return stmt;
}
