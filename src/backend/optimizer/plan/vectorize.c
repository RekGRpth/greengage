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

#include "access/table.h"
#include "access/tupdesc.h"
#include "cdb/cdbllize.h"
#include "cdb/cdbplan.h"
#include "cdb/cdbvars.h"
#include "cdb/memquota.h"
#include "utils/resgroup.h"
#include "utils/resource_manager.h"
#include "catalog/pg_am.h"
#include "executor/nodeVectorRegion.h"
#include "executor/nodeLimit.h"
#include "executor/nodeMaterial.h"
#include "executor/nodeMergejoin.h"
#include "executor/nodeNestloop.h"
#include "executor/nodeSort.h"
#include "executor/nodeUnique.h"
#include "executor/vectorAgg.h"
#include "executor/vectorExpr.h"
#include "executor/vectorHashJoin.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/optimizer.h"
#include "optimizer/planner.h"
#include "optimizer/tlist.h"
#include "optimizer/walkers.h"
#include "parser/parsetree.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"

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

/*
 * A Motion probe leaf maps its output columns into batch columns
 * positionally, so its targetlist must be a plain Var projection.
 */
static bool
motion_probe_tlist_is_simple(Plan *motion)
{
	ListCell   *lc;

	foreach(lc, motion->targetlist)
	{
		TargetEntry *te = lfirst_node(TargetEntry, lc);

		if (te->resjunk || !IsA(te->expr, Var))
			return false;
	}
	return true;
}

static bool
is_simple_scan(VectorizeContext *ctx, SeqScan *scan)
{
	RangeTblEntry *rte;
	ListCell   *lc;
	VectorExprVarMap vars = {scan->scanrelid, NULL, 0};
	vars.allow_correlated = true;

	if (scan->scanrelid <= 0 ||
		scan->plan.lefttree != NULL || scan->plan.righttree != NULL)
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
 * A static Append can share one batch layout when every child has the same
 * physical tuple descriptor and the same simple targetlist.  Runtime pruning,
 * parallel workers and more general input mappings stay with the row executor.
 */
typedef struct AppendQualVarsContext
{
	Index		scanrelid;
	Bitmapset  *attrs;
} AppendQualVarsContext;

static bool
append_qual_subplan_vars_walker(Node *node, void *context)
{
	AppendQualVarsContext *ctx = (AppendQualVarsContext *) context;

	if (node == NULL)
		return false;
	if (IsA(node, SubPlan))
	{
		SubPlan    *sp = (SubPlan *) node;

		/* pull_varattnos deliberately does not descend into SubPlan args,
		 * but those outer Vars also need a column in the shared batch. */
		pull_varattnos((Node *) sp->args, ctx->scanrelid, &ctx->attrs);
	}
	return expression_tree_walker(node, append_qual_subplan_vars_walker, context);
}

static bool
is_simple_append(VectorizeContext *ctx, Append *append)
{
	SeqScan	   *first = NULL;
	ListCell   *lc;
	bool		supported = false;

	if (append->appendplans == NIL || append->plan.parallel_aware)
		return false;

	foreach(lc, append->appendplans)
	{
		SeqScan	   *scan;
		ListCell   *scanlc;

		if (!IsA(lfirst(lc), SeqScan))
			goto done;
		scan = (SeqScan *) lfirst(lc);
		if (!is_simple_scan(ctx, scan))
			goto done;


		if (first == NULL)
		{
			ListCell   *rootlc;

			first = scan;
			if (list_length(append->plan.targetlist) !=
				list_length(scan->plan.targetlist))
				goto done;
			forboth(rootlc, append->plan.targetlist,
					scanlc, scan->plan.targetlist)
			{
				TargetEntry *rootte = (TargetEntry *) lfirst(rootlc);
				TargetEntry *scante = (TargetEntry *) lfirst(scanlc);
				Var		   *var;

				if (rootte->resjunk || !IsA(rootte->expr, Var))
					goto done;
				var = (Var *) rootte->expr;
				if (var->varattno != rootte->resno || var->varlevelsup != 0 ||
					exprType((Node *) rootte->expr) != exprType((Node *) scante->expr) ||
					exprTypmod((Node *) rootte->expr) != exprTypmod((Node *) scante->expr))
					goto done;
			}
		}
		else
		{
			ListCell   *firstlc;

			if (list_length(first->plan.targetlist) != list_length(scan->plan.targetlist))
				goto done;
			forboth(firstlc, first->plan.targetlist,
					scanlc, scan->plan.targetlist)
			{
				TargetEntry *firstte = (TargetEntry *) lfirst(firstlc);
				TargetEntry *scante = (TargetEntry *) lfirst(scanlc);

				/*
				 * The child may number its columns differently than the first
				 * one (UNION ALL over independently created relations); the
				 * executor remaps such a child onto the first one's batch
				 * layout, provided every attribute its qual references is one
				 * the targetlist maps.
				 */
				if (exprType((Node *) firstte->expr) != exprType((Node *) scante->expr) ||
					exprTypmod((Node *) firstte->expr) != exprTypmod((Node *) scante->expr))
					goto done;
			}
			if (scan->plan.qual != NIL)
			{
					Bitmapset *mapped = NULL;
					Bitmapset *qattrs = NULL;
					AppendQualVarsContext subctx;
					int		attno;
					bool		covered = true;

					{
						ListCell   *map1lc,
								   *map2lc;

						forboth(map1lc, first->plan.targetlist,
								  map2lc, scan->plan.targetlist)
							mapped = bms_add_member(mapped,
									((Var *) ((TargetEntry *) lfirst(map2lc))->expr)->varattno -
									FirstLowInvalidHeapAttributeNumber);
					}
					pull_varattnos((Node *) scan->plan.qual, scan->scanrelid, &qattrs);
					subctx.scanrelid = scan->scanrelid;
					subctx.attrs = qattrs;
					append_qual_subplan_vars_walker((Node *) scan->plan.qual, &subctx);
					qattrs = subctx.attrs;
					while ((attno = bms_first_member(qattrs)) > 0)
					{
						attno += FirstLowInvalidHeapAttributeNumber;
						if (!bms_is_member(attno - FirstLowInvalidHeapAttributeNumber, mapped))
						{
							covered = false;
							break;
						}
					}
					bms_free(mapped);
					bms_free(qattrs);
					if (!covered)
						goto done;
			}
		}
	}
	supported = true;

done:
	return supported;
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
	vars.allow_correlated = true;

	if (scan->scanrelid <= 0 ||
		scan->plan.lefttree != NULL || scan->plan.righttree != NULL)
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
		case T_NestLoop: return "NestLoop";
		case T_Material: return "Material";
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

/*
 * AOCS block buffers remain alive until the batch has been consumed, so their
 * by-ref values need no per-row copy.  Row scans and row adapters have no such
 * ownership guarantee and still copy every referenced by-ref value.
 */
static bool
leaf_has_expensive_byref(VectorizeContext *ctx, Plan *leaf)
{
	SeqScan	   *scan;
	RangeTblEntry *rte;

	if (!leaf_has_byref(leaf))
		return false;
	if (!IsA(leaf, SeqScan))
		return true;

	scan = (SeqScan *) leaf;
	if (scan->scanrelid <= 0 || scan->scanrelid > list_length(ctx->stmt->rtable))
		return true;
	rte = rt_fetch(scan->scanrelid, ctx->stmt->rtable);
	return rte->rtekind != RTE_RELATION ||
		get_rel_am(rte->relid) != AO_COLUMN_TABLE_AM_OID;
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
	if (leaf_has_expensive_byref(ctx, IsA(plan, SeqScan) ? plan : outerPlan(plan)) ||
		(IsA(plan, HashJoin) &&
		 leaf_has_expensive_byref(ctx, outerPlan(innerPlan(plan)))))
	{
		decision(ctx, plan, "the leaf copies pass-by-reference columns into batches");
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

typedef struct DisplayCtx
{
	Plan	   *child;
	List	   *rtable;
	int			rteidx;
	List	  **colnames;
	List	  **coltypes;
	List	  **coltypmods;
	List	  **colcollations;
} DisplayCtx;

static Node *
display_var_mutator(Node *node, void *context)
{
	DisplayCtx *dctx = (DisplayCtx *) context;
	Var		   *var = (Var *) node;
	Var		   *nv;
	char	   *name = NULL;
	int			attno;

	if (node == NULL)
		return NULL;
	if (IsA(node, Var))
	{
		if (var->varno == OUTER_VAR && dctx->child != NULL &&
			var->varattno >= 1 && var->varattno <= list_length(dctx->child->targetlist))
		{
			TargetEntry *te = list_nth(dctx->child->targetlist, var->varattno - 1);

			if (te->resname)
				name = pstrdup(te->resname);
			else if (IsA(te->expr, Var))
			{
				Var		   *cv = (Var *) te->expr;

				if (cv->varno > 0 && cv->varno <= list_length(dctx->rtable) &&
					cv->varattno > 0)
				{
					RangeTblEntry *rte = rt_fetch(cv->varno, dctx->rtable);

					if (rte->rtekind == RTE_RELATION)
						name = get_attname(rte->relid, cv->varattno, true);
					else if (rte->eref && cv->varattno <= list_length(rte->eref->colnames))
						name = pstrdup(strVal(list_nth(rte->eref->colnames, cv->varattno - 1)));
				}
			}
		}
		if (name == NULL)
			name = psprintf("in%d", var->varattno);
		*dctx->colnames = lappend(*dctx->colnames, makeString(name));
		*dctx->coltypes = lappend_oid(*dctx->coltypes, var->vartype);
		*dctx->coltypmods = lappend_int(*dctx->coltypmods, var->vartypmod);
		*dctx->colcollations = lappend_oid(*dctx->colcollations, var->varcollid);
		attno = list_length(*dctx->colnames);
		nv = makeVar(dctx->rteidx, attno, var->vartype, var->vartypmod,
					 var->varcollid, 0);
		return (Node *) nv;
	}
	return expression_tree_mutator(node, display_var_mutator, context);
}

/* Find an aggregate output through pass-through Vars, not a final phase. */
static Aggref *
find_output_aggref(Plan *plan, int attno, Plan **child)
{
	int			depth;

	for (depth = 0; plan != NULL && depth < 8; depth++)
	{
		TargetEntry *te;
		Node	   *expr;

		if (attno < 1 || attno > list_length(plan->targetlist))
			return NULL;
		te = (TargetEntry *) list_nth(plan->targetlist, attno - 1);
		expr = (Node *) te->expr;
		if (IsA(expr, Aggref))
		{
			Aggref	   *agg = (Aggref *) expr;

			if (agg->aggsplit == AGGSPLIT_FINAL_DESERIAL || !IsA(plan, Agg))
				return NULL;
			*child = plan->lefttree;
			return agg;
		}
		if (!IsA(expr, Var) || ((Var *) expr)->varno != OUTER_VAR)
			return NULL;
		attno = ((Var *) expr)->varattno;
		plan = plan->lefttree;
	}
	return NULL;
}

/*
 * An aggregate is memory-intensive when it computes an ordered aggregate
 * (IsAggMemoryIntensive in memquota.c); the same condition re-checked for
 * the wrapper so a vectorized region keeps the replaced node's class.
 */
static bool
vectorize_ordered_agg_walker(Node *node, void *context)
{
	if (node == NULL)
		return false;
	if (IsA(node, Aggref))
	{
		Aggref	   *aggref = (Aggref *) node;

		if (aggref->aggorder || aggref->aggdistinct)
			return true;
	}
	return expression_tree_walker(node, vectorize_ordered_agg_walker, context);
}

static bool
vectorize_has_ordered_aggs(Node *node)
{
	return vectorize_ordered_agg_walker(node, NULL);
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
		Expr	   *scanexpr = (Expr *) scanvar;

		{
			Plan	   *child = NULL;
			Aggref	   *agg = find_output_aggref(root, k, &child);

			if (agg != NULL)
			{
				DisplayCtx	dctx;

				dctx.child = child;
				dctx.rtable = ctx->stmt->rtable;
				dctx.rteidx = rteidx;
				dctx.colnames = &colnames;
				dctx.coltypes = &coltypes;
				dctx.coltypmods = &coltypmods;
				dctx.colcollations = &colcollations;
				scanexpr = (Expr *) display_var_mutator((Node *) copyObject(agg), &dctx);
			}
		}

		colnames = lappend(colnames, makeString(name));
		coltypes = lappend_oid(coltypes, exprType(expr));
		coltypmods = lappend_int(coltypmods, exprTypmod(expr));
		colcollations = lappend_oid(colcollations, exprCollation(expr));
		scan_tlist = lappend(scan_tlist,
							 makeTargetEntry(scanexpr, k, name, te->resjunk));
		out_tlist = lappend(out_tlist,
							makeTargetEntry((Expr *) outvar, k, name, te->resjunk));
		k++;
	}

	rte = makeNode(RangeTblEntry);
	rte->rtekind = RTE_NAMEDTUPLESTORE;
	rte->enrname = "vector_region";
	rte->enrtuples = root->plan_rows > (double) INT_MAX ?
		INT_MAX : (int) root->plan_rows;
	rte->eref = makeAlias("vector_region", colnames);
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
	cs->scan.plan.initPlan = root->initPlan;
	cs->scan.plan.extParam = root->extParam;
	cs->scan.plan.allParam = root->allParam;
	cs->scan.plan.operatorMemKB = memory_quota_kb(root);
	/*
	 * Carry the replaced node's blocking/memory-intensive classification
	 * onto the wrapper, mirroring IsAggBlockingOperator/IsAggMemoryIntensive
	 * in memquota.c, so the memory quota walkers see the same shape they
	 * would without vectorization.
	 */
	if (IsA(root, Agg))
	{
		Agg	   *agg = (Agg *) root;

		if (agg->aggstrategy == AGG_PLAIN ||
			(agg->aggstrategy == AGG_HASHED && !agg->streaming))
			cs->flags |= CUSTOMSCAN_GP_BLOCKING;
		if (agg->aggstrategy == AGG_HASHED || agg->aggstrategy == AGG_MIXED ||
			vectorize_has_ordered_aggs((Node *) agg->plan.targetlist) ||
			vectorize_has_ordered_aggs((Node *) agg->plan.qual))
			cs->flags |= CUSTOMSCAN_GP_MEMORY_INTENSIVE;
	}
	else if (IsA(root, HashJoin) || IsA(root, Sort) || IsA(root, Unique) ||
			 IsA(root, MergeJoin) || IsA(root, Material))
		cs->flags |= CUSTOMSCAN_GP_MEMORY_INTENSIVE | CUSTOMSCAN_GP_BLOCKING;

	ctx->wrapped++;
	return (Plan *) cs;
}

/*
 * Capturing a region swallows the whole subtree below its root, so Motion
 * producer subtrees hanging inside it never reach the top-down mutator.
 * Vectorize them first: each Motion switches to its producer slice and the
 * mutator wraps eligible joins and scans there as sibling regions under the
 * Motion (initialized and executed by the Motion's own gang, exactly like
 * the scan regions that already appear under Motions).
 */
static void
vectorize_motion_subtrees(VectorizeContext *ctx, Plan *plan)
{
	if (plan == NULL || IsA(plan, CustomScan))
		return;
	if (IsA(plan, Motion))
	{
		Motion	   *motion = (Motion *) plan;
		int			saved_slice = ctx->slice;

		ctx->slice = motion->motionID;
		outerPlan(motion) = (Plan *) vectorize_mutator((Node *) outerPlan(motion),
													   ctx);
		ctx->slice = saved_slice;
		return;
	}
	vectorize_motion_subtrees(ctx, outerPlan(plan));
	vectorize_motion_subtrees(ctx, innerPlan(plan));
}

/*
 * Walk a join chain's probe side and vectorize a leaf the chain builder
 * cannot consume natively (an aggregate over a flattened EXISTS subquery,
 * a Result, ...).  Capture swallows the whole subtree, so without this the
 * leaf never reaches the top-down mutator and stays row-wise inside the
 * region.  SeqScan stays native; Motions vectorize their producer
 * subtrees; a pass-through Sort belongs to the region's own Sort driver,
 * so the walk continues below it.
 */
static void
vectorize_chain_leaves(VectorizeContext *ctx, Plan **link)
{
	Plan	   *plan = *link;

	while (plan != NULL)
	{
		if (IsA(plan, SeqScan) || IsA(plan, CustomScan))
			return;
		if (IsA(plan, Motion))
		{
			vectorize_motion_subtrees(ctx, plan);
			return;
		}
		if (IsA(plan, Sort))
		{
			link = &outerPlan(plan);
			plan = *link;
			continue;
		}
		if (!IsA(plan, HashJoin))
		{
			/* replace the parent's pointer, never the leaf's own outer */
			*link = (Plan *) vectorize_mutator((Node *) plan, ctx);
			return;
		}
		link = &outerPlan(plan);
		plan = *link;
	}
}

static Node *
vectorize_mutator(Node *node, VectorizeContext *ctx)
{
	if (node == NULL)
		return NULL;

	/*
	 * A captured region is frozen: the pre-processing walks
	 * (vectorize_motion_subtrees, vectorize_chain_leaves) may reach an
	 * already-built region through plan_tree_mutator's descent into
	 * custom_plans, and re-wrapping nodes inside it would nest regions
	 * within regions the executor cannot initialize.
	 */
	if (IsA(node, CustomScan))
		return node;

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
		((IsA(outerPlan(node), SeqScan) &&
		  is_simple_scan(ctx, (SeqScan *) outerPlan(node))) ||
		 (IsA(outerPlan(node), Motion) &&
		  motion_probe_tlist_is_simple(outerPlan(node)))) &&
		/* the build side reads row-wise through the Hash node: any input */
		IsA(innerPlan(node), Hash))
	{
		Plan	   *hash = innerPlan(node);

		/*
		 * A Motion probe leaf receives its rows from another slice: capture
		 * swallows the producer subtree, so vectorize it first.  The region
		 * converts the received rows into probe batches.
		 */
		if (IsA(outerPlan(node), Motion))
			vectorize_motion_subtrees(ctx, outerPlan(node));
		outerPlan(hash) = (Plan *) vectorize_mutator((Node *) outerPlan(hash), ctx);
		if (auto_allowed(ctx, (Plan *) node))
			return (Node *) make_region(ctx, (Plan *) node);
		return node;
	}

	if (IsA(node, Material) && slice_runs_on_segments(ctx) &&
		VectorMaterialPlanIsSupported((Material *) node) &&
		IsA(outerPlan(node), SeqScan) &&
		is_simple_scan(ctx, (SeqScan *) outerPlan(node)) &&
		auto_allowed(ctx, (Plan *) node))
		return (Node *) make_region(ctx, (Plan *) node);

	if (IsA(node, NestLoop) && slice_runs_on_segments(ctx) &&
		VectorNestLoopPlanIsSupported((NestLoop *) node) &&
		IsA(outerPlan(node), SeqScan) &&
		is_simple_scan(ctx, (SeqScan *) outerPlan(node)))
	{
		innerPlan(node) = (Plan *) vectorize_mutator((Node *) innerPlan(node), ctx);
		if (auto_allowed(ctx, (Plan *) node))
			return (Node *) make_region(ctx, (Plan *) node);
		return node;
	}

	if (IsA(node, MergeJoin) && slice_runs_on_segments(ctx) &&
		VectorMergeJoinPlanIsSupported((MergeJoin *) node) &&
		IsA(outerPlan(node), Sort) &&
		VectorSortPlanIsSupported((Sort *) outerPlan(node)) &&
		IsA(outerPlan(outerPlan(node)), SeqScan) &&
		is_simple_scan(ctx, (SeqScan *) outerPlan(outerPlan(node))))
	{
		innerPlan(node) = (Plan *) vectorize_mutator((Node *) innerPlan(node), ctx);
		if (auto_allowed(ctx, (Plan *) node))
			return (Node *) make_region(ctx, (Plan *) node);
		return node;
	}

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
		IsA(outerPlan(node), SeqScan) &&
		is_simple_scan(ctx, (SeqScan *) outerPlan(node)) &&
		auto_allowed(ctx, (Plan *) node))
		return (Node *) make_region(ctx, (Plan *) node);

	if (IsA(node, Append) && slice_runs_on_segments(ctx) &&
		is_simple_append(ctx, (Append *) node) &&
		auto_allowed(ctx, (Plan *) node))
		return (Node *) make_region(ctx, (Plan *) node);

	if (IsA(node, Agg) && slice_runs_on_segments(ctx) &&
		VectorAggPlanIsSupported((Agg *) node) &&
		(!IsA(outerPlan(node), Append) ||
		 is_simple_append(ctx, (Append *) outerPlan(node))) &&
		(!IsA(outerPlan(node), SeqScan) ||
		 is_simple_scan(ctx, (SeqScan *) outerPlan(node))) &&
		auto_allowed(ctx, (Plan *) node))
	{
		vectorize_motion_subtrees(ctx, outerPlan(node));
		vectorize_chain_leaves(ctx, &outerPlan(node));
		return (Node *) make_region(ctx, (Plan *) node);
	}

	if (IsA(node, SeqScan) && slice_runs_on_segments(ctx) &&
		/* an empty targetlist is worth wrapping while a qual filters rows */
		(((SeqScan *) node)->plan.targetlist != NIL ||
		 ((SeqScan *) node)->plan.qual != NIL) &&
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
	/*
	 * set_plan_references assigns ids from one shared counter, main tree
	 * first and subplans after, so subplan ids start right where the main
	 * tree ends; regions must not reuse them.
	 */
	max_plan_node_id_walker((Node *) stmt->subplans, &ctx);

	if (gp_vectorized_execution == GP_VECTORIZED_EXECUTION_AUTO)
	{
		uint64		bytes = (uint64) statement_mem * 1024L;

		/*
		 * The gate below validates workset estimates against the memory
		 * the query will actually hold: PortalStart replaces query_mem
		 * with the resource manager grant before ExecutorStart re-runs
		 * this very assignment.  At plan time only the fixed-mem
		 * resource group configuration is knowable (the dynamic group
		 * cap needs the not-yet-assigned group); a dynamically capped
		 * group can still grant less, which the runtime spill then
		 * handles.
		 */
		if (IsResGroupEnabled() && gp_resgroup_memory_query_fixed_mem > 0)
			bytes = Min(bytes, (uint64) gp_resgroup_memory_query_fixed_mem * 1024L);

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
