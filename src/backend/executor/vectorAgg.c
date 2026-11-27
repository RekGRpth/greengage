/*-------------------------------------------------------------------------
 *
 * vectorAgg.c
 *	  Feed column batches into PostgreSQL aggregate transition states.
 *
 * Aggregate metadata, grouping hash table, transition functions and output
 * finalization stay owned by nodeAgg.  This file evaluates aggregate
 * arguments in vector form and supplies per-row values to those states.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "catalog/pg_aggregate.h"
#include "catalog/pg_type_d.h"
#include "executor/executor.h"
#include "nodes/nodeFuncs.h"
#include "executor/vectorAgg.h"
#include "executor/vectorExpr.h"
#include "miscadmin.h"
#include "parser/parsetree.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "utils/float.h"
#include "utils/numeric.h"
#include "utils/tuplesort.h"
#include "utils/tuplestore.h"
#include "utils/datum.h"

extern Datum int8_numeric(PG_FUNCTION_ARGS);

typedef enum VectorAggBuiltin
{
	VEC_AGG_AVG_INT8 = 2100,
	VEC_AGG_AVG_INT4 = 2101,
	VEC_AGG_AVG_INT2 = 2102,
	VEC_AGG_AVG_NUMERIC = 2103,
	VEC_AGG_SUM_INT8 = 2107,
	VEC_AGG_SUM_INT4 = 2108,
	VEC_AGG_SUM_INT2 = 2109,
	VEC_AGG_SUM_FLOAT4 = 2110,
	VEC_AGG_SUM_FLOAT8 = 2111,
	VEC_AGG_SUM_NUMERIC = 2114,
	VEC_AGG_MAX_INT8 = 2115,
	VEC_AGG_MAX_INT4 = 2116,
	VEC_AGG_MAX_INT2 = 2117,
	VEC_AGG_MAX_FLOAT4 = 2119,
	VEC_AGG_MAX_FLOAT8 = 2120,
	VEC_AGG_MAX_DATE = 2122,
	VEC_AGG_MAX_TIMESTAMP = 2126,
	VEC_AGG_MAX_TIMESTAMPTZ = 2127,
	VEC_AGG_MAX_NUMERIC = 2130,
	VEC_AGG_MIN_INT8 = 2131,
	VEC_AGG_MIN_INT4 = 2132,
	VEC_AGG_MIN_INT2 = 2133,
	VEC_AGG_MIN_FLOAT4 = 2135,
	VEC_AGG_MIN_FLOAT8 = 2136,
	VEC_AGG_MIN_DATE = 2138,
	VEC_AGG_MIN_TIMESTAMP = 2142,
	VEC_AGG_MIN_TIMESTAMPTZ = 2143,
	VEC_AGG_MIN_NUMERIC = 2146,
	VEC_AGG_STDDEV_INT8 = 2154,
	VEC_AGG_STDDEV_INT4 = 2155,
	VEC_AGG_STDDEV_INT2 = 2156,
	VEC_AGG_VAR_SAMP_INT8 = 2641,
	VEC_AGG_VAR_SAMP_INT4 = 2642,
	VEC_AGG_VAR_SAMP_INT2 = 2643,
	VEC_AGG_STDDEV_SAMP_INT8 = 2712,
	VEC_AGG_STDDEV_SAMP_INT4 = 2713,
	VEC_AGG_STDDEV_SAMP_INT2 = 2714,
	VEC_AGG_VAR_POP_INT8 = 2718,
	VEC_AGG_VAR_POP_INT4 = 2719,
	VEC_AGG_VAR_POP_INT2 = 2720,
	VEC_AGG_STDDEV_POP_INT8 = 2724,
	VEC_AGG_STDDEV_POP_INT4 = 2725,
	VEC_AGG_STDDEV_POP_INT2 = 2726,
	VEC_AGG_COUNT_ANY = 2802,
	VEC_AGG_COUNT_STAR = 2803
} VectorAggBuiltin;

struct VectorAggState
{
	MemoryContext context;
	bool		packed_i32;
	AggState   *agg;
	VectorExprVarMap vars;
	VectorExprState ***args;
	VectorExprState **filters;
	bool	   *deferred;		/* transno: arg yields decoded numerics */
	Datum	  **filter_values;	/* per transno: root arrays of the filter */
	uint64	 **filter_nulls;
	Datum	 ***arg_values;		/* per transno/argno: root value arrays */
	int32	  ***arg_i32;		/* packed 32-bit root, when present */
	int16	  ***arg_i16;		/* packed INT2 root, when present */
	uint64	***arg_nulls;
	int32	   *filtered_selection;
	Datum	  **values;
	bool	  **nulls;
	TupleTableSlot *input_slot;	/* full input row, for spilling */
	/* group memo: a row whose keys equal the previous row's skips the hash */
	bool		group_valid;	/* memo holds this batch's last group */
	AggStatePerGroup group_pergroup;
	Datum	   *group_keys;
	bool	   *group_key_nulls;
	/*
	 * Fast open-addressing group table for by-value fixed-width keys; NULL
	 * once rejected or after conversion into the standard hash table.
	 */
	struct VectorAggFastGroups *fast;
	bool		fast_rejected;	/* the keys are not fast-table material */
	int			fast_pos;		/* next fast-table entry during retrieval */
	int			projected_set;
	bool		group_open;
	bool		input_done;
	bool		output_done;
	/* AGG_MIXED: plain (non-hash) grouping sets are emitted before hashed */
	int			mixed_plain_index;
	int			plain_set_index;	/* multi-set AGG_PLAIN output cursor */
	bool		mixed_hashed_started;
	/* multi-phase sorted aggregation: per-phase spools and phase driver */
	bool		multi_phase;
	bool		is_combine;	/* the split is FINAL_DESERIAL: combine transitions */
	VectorBatch *batch;
	TupleTableSlot *sort_slot;	/* minimal-tuple slot for spool reads */
	int			position;
	int			current_phase_idx;
	bool		spool_done;		/* current phase's input is exhausted */
	uint32		spool_gen;		/* batch generation already spooled */
	Tuplesortstate **phase_sorts;	/* per phase >= 2: next-phase sort input */
	Tuplestorestate **phase_stores;	/* plain phases need no ordering */
	/*
	 * Argument values evaluated for (args_batch, args_gen).  The sorted
	 * driver returns after every output row and is re-entered for the next
	 * one; without this key it would re-evaluate the whole input batch on
	 * every call (quadratic for row-wise consumers such as a motion
	 * sender above the region).
	 */
	VectorBatch *args_batch;
	uint32		args_gen;
};

bool
VectorAggPackedI32IsSupported(AggState *agg)
{
	int			transno;

	if ((agg->aggstrategy != AGG_PLAIN && agg->aggstrategy != AGG_HASHED) ||
		agg->numtrans == 0)
		return false;
	if (agg->aggstrategy == AGG_HASHED)
	{
		Agg		   *plan = (Agg *) agg->ss.ps.plan;
		Plan	   *outer = outerPlan(plan);
		int			col;

		for (col = 0; col < plan->numCols; col++)
		{
			TargetEntry *te = get_tle_by_resno(outer->targetlist,
										  plan->grpColIdx[col]);

			if (te == NULL || !IsA(te->expr, Var) ||
				(((Var *) te->expr)->vartype != INT2OID &&
				 ((Var *) te->expr)->vartype != INT4OID &&
				 ((Var *) te->expr)->vartype != DATEOID &&
				 ((Var *) te->expr)->vartype != FLOAT4OID))
				return false;
		}
	}
	for (transno = 0; transno < agg->numtrans; transno++)
	{
		AggStatePerTrans pertrans = &agg->pertrans[transno];
		Oid			fnoid = pertrans->aggref->aggfnoid;

		if (pertrans->numSortCols != 0 || pertrans->aggref->aggfilter != NULL ||
			pertrans->aggref->aggdistinct)
			return false;
		if (fnoid != VEC_AGG_SUM_INT2 && fnoid != VEC_AGG_AVG_INT2 &&
			fnoid != VEC_AGG_MIN_INT2 && fnoid != VEC_AGG_MAX_INT2 &&
			fnoid != VEC_AGG_SUM_INT4 && fnoid != VEC_AGG_AVG_INT4 &&
			fnoid != VEC_AGG_MIN_INT4 && fnoid != VEC_AGG_MAX_INT4 &&
			fnoid != VEC_AGG_MIN_DATE && fnoid != VEC_AGG_MAX_DATE &&
			fnoid != VEC_AGG_SUM_FLOAT4 && fnoid != VEC_AGG_MIN_FLOAT4 &&
			fnoid != VEC_AGG_MAX_FLOAT4)
			return false;
		{
			ListCell   *lc;

			foreach(lc, pertrans->aggref->args)
				if (!VectorExprPackedI32IsSupported(
					((TargetEntry *) lfirst(lc))->expr))
					return false;
		}
	}
	return true;
}

static bool
build_var_map(Agg *agg, MemoryContext context, VectorExprVarMap *vars)
{
	Plan	   *outer = outerPlan(agg);
	SeqScan	   *scan;
	ListCell   *lc;
	int			col = 0;

	if (!IsA(outer, SeqScan) && !IsA(outer, Motion) &&
		!IsA(outer, CustomScan) && !IsA(outer, HashJoin) &&
		agg->aggstrategy != AGG_HASHED && agg->aggstrategy != AGG_SORTED &&
		agg->aggstrategy != AGG_MIXED)
		return false;
	if (IsA(outer, HashJoin))
	{
		/*
		 * The region's join chain consumes the probe side only through
		 * SeqScan, Motion, nested join chains or batch handouts
		 * (VectorRegionCreateJoinChain); a chain ending in anything else
		 * falls back to the row adapter, wrapping a row pipeline in a
		 * region for pure overhead.  Mirror the acceptance at plan time.
		 */
		Plan	   *probe = outerPlan(outer);

		while (IsA(probe, HashJoin))
			probe = outerPlan(probe);
		if (!IsA(probe, SeqScan) && !IsA(probe, Motion) &&
			!IsA(probe, CustomScan))
			return false;
	}

	vars->outer_ncols = list_length(outer->targetlist);
	vars->outer_attnos = MemoryContextAlloc(context,
										 sizeof(AttrNumber) * Max(vars->outer_ncols, 1));
	if (IsA(outer, Append))
	{
		Append	   *append = (Append *) outer;
		SeqScan	   *child;

		if (append->appendplans == NIL ||
			!IsA(linitial(append->appendplans), SeqScan))
			return false;
		child = (SeqScan *) linitial(append->appendplans);
		vars->scanrelid = 0;
		foreach(lc, child->plan.targetlist)
		{
			TargetEntry *te = (TargetEntry *) lfirst(lc);
			Var		   *var;

			if (te->resjunk || !IsA(te->expr, Var))
				return false;
			var = (Var *) te->expr;
			if (var->varattno <= 0 || var->varlevelsup != 0)
				return false;
			vars->outer_attnos[col++] = var->varattno;
		}
		return true;
	}
	if (!IsA(outer, SeqScan))
	{
		/*
		 * A region above a Motion, another row leaf, or a child region
		 * reads its output shape.
		 */
		vars->scanrelid = 0;
		foreach(lc, outer->targetlist)
		{
			TargetEntry *te = (TargetEntry *) lfirst(lc);

			if (te->resjunk)
				return false;
			vars->outer_attnos[col] = col + 1;
			col++;
		}
		return true;
	}

	scan = (SeqScan *) outer;
	vars->scanrelid = scan->scanrelid;

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
		vars->outer_attnos[col++] = var->varattno;
	}
	return true;
}

/*
 * The per-aggregate plan checks for one Aggref inside a finalize
 * targetlist expression: same rules as a direct tlist Aggref.
 */
static bool
aggref_plan_supported(Aggref *aggref, Agg *agg, VectorExprVarMap *vars)
{
	ListCell   *arglc;

	if ((aggref->aggkind != AGGKIND_NORMAL &&
		 !AGGKIND_IS_ORDERED_SET(aggref->aggkind)) ||
		aggref->agglevelsup != 0 ||
		aggref->aggsplit != agg->aggsplit || aggref->aggvariadic ||
		(aggref->aggdirectargs != NIL &&
		 !AGGKIND_IS_ORDERED_SET(aggref->aggkind)) ||
		aggref->aggtype == INTERNALOID ||
		((aggref->aggorder != NIL || aggref->aggdistinct != NIL) &&
		 agg->aggstrategy != AGG_PLAIN && agg->aggstrategy != AGG_SORTED) ||
		(aggref->aggdistinct != NIL && list_length(aggref->args) != 1))
		return false;
	if (aggref->aggfilter != NULL &&
		!VectorExprIsSupported(aggref->aggfilter, vars))
		return false;
	foreach(arglc, aggref->args)
	{
		TargetEntry *arg = (TargetEntry *) lfirst(arglc);

		if (!VectorExprIsSupported(arg->expr, vars))
			return false;
	}
	return true;
}

struct FinalizeTlistContext
{
	Agg			  *agg;
	VectorExprVarMap *vars;
};

/*
 * A finalize targetlist expression must be built from constants, outer
 * Vars and supported Aggrefs: the row machinery projects exactly such
 * expressions after finalize_aggregates.  This admits e.g. the
 * planner's avg(int8) expansion, a division expression over the sum
 * and count aggregates.
 */
static bool
finalize_tlist_walker(Node *node, void *context)
{
	struct FinalizeTlistContext *ctx = (struct FinalizeTlistContext *) context;

	if (node == NULL)
		return true;
	if (IsA(node, Aggref))
		return aggref_plan_supported((Aggref *) node, ctx->agg, ctx->vars);
	if (IsA(node, Var))
		return VectorExprIsSupported((Expr *) node, ctx->vars);
	if (IsA(node, Const) || IsA(node, Param) || IsA(node, CaseTestExpr))
		return true;
	return expression_tree_walker(node, finalize_tlist_walker, (void *) ctx);
}

bool
VectorAggPlanIsSupported(Agg *agg)
{
	VectorExprVarMap vars;
	ListCell   *lc;
	int			col;
	bool		supported = false;

	if ((agg->aggstrategy != AGG_PLAIN && agg->aggstrategy != AGG_HASHED &&
		 agg->aggstrategy != AGG_SORTED && agg->aggstrategy != AGG_MIXED) ||
		(agg->aggsplit != AGGSPLIT_SIMPLE &&
		 agg->aggsplit != AGGSPLIT_INITIAL_SERIAL &&
		 !(agg->aggsplit == AGGSPLIT_FINAL_DESERIAL &&
		   agg->aggstrategy == AGG_HASHED)) ||
		!bms_is_empty(agg->aggParams) || agg->agg_expr_id != 0)
		return false;
	/*
	 * A streaming hash aggregate processes input in rounds without spilling
	 * and may emit a group's partial state more than once; the consumer
	 * (a Finalize aggregate) combines states associatively, so emitting the
	 * whole input in one pass is equivalent.  Other streaming strategies
	 * stay with the standard executor.
	 */
	if (agg->streaming && agg->aggstrategy != AGG_HASHED)
		return false;
	if (agg->aggstrategy == AGG_MIXED)
	{
		ListCell   *clc;

		/*
		 * Only mixed plans whose non-hash phases are all plain (no ordered
		 * input) run in one pass over the batch stream.  Sorted phases would
		 * need the tuplesort machinery, which the region does not have.
		 */
		foreach(clc, agg->chain)
		{
			Agg		   *ch = (Agg *) lfirst(clc);

			/*
			 * The chain holds the other hashed groupings (AGG_HASHED)
			 * followed by plain empty-set phases (AGG_PLAIN).  Sorted phases
			 * need ordered input the region does not provide.
			 */
			if ((ch->aggstrategy != AGG_HASHED && ch->aggstrategy != AGG_PLAIN) ||
				(ch->aggstrategy == AGG_PLAIN && ch->numCols != 0))
				return false;
		}
	}
	else if (agg->aggstrategy == AGG_SORTED && agg->chain != NIL)
	{
		ListCell   *clc;

		/* multi-phase sorted aggregation: phases replay from per-phase
		 * tuplesorts built during the single pass over the input */
		foreach(clc, agg->chain)
		{
			Agg		   *ch = (Agg *) lfirst(clc);

			if ((ch->aggstrategy != AGG_SORTED && ch->aggstrategy != AGG_PLAIN) ||
				(ch->aggstrategy == AGG_PLAIN && ch->numCols != 0))
				return false;
		}
	}
	else if (agg->chain != NIL)
		return false;
	if ((agg->aggstrategy == AGG_PLAIN && agg->numCols != 0) ||
		(agg->aggstrategy != AGG_PLAIN && agg->numCols == 0))
		return false;
	if (!build_var_map(agg, CurrentMemoryContext, &vars))
		return false;
	/* Aggregate arguments and FILTER may execute correlated subplans per row. */
	vars.allow_correlated = true;

	for (col = 0; col < agg->numCols; col++)
	{
		if (agg->grpColIdx[col] <= 0 || agg->grpColIdx[col] > vars.outer_ncols)
			goto done;
	}

	foreach(lc, agg->plan.targetlist)
	{
		TargetEntry *te = (TargetEntry *) lfirst(lc);

		if (te->resjunk)
			goto done;
		if (IsA(te->expr, GroupingSetId) || IsA(te->expr, GroupingFunc))
			continue;	/* evaluated by nodeAgg's projection, not the batch exprs */
		if (IsA(te->expr, Var))
		{
			if (!VectorExprIsSupported((Expr *) te->expr, &vars))
				goto done;
			continue;
		}
		/*
		 * Any targetlist expression over Aggrefs and group-key Vars
		 * projects through the row machinery's finalize path
		 * (finalize_aggregates + project_aggregates): accept it when
		 * every Aggref inside passes the per-aggregate checks and every
		 * Var reads an outer column the batch can supply.
		 */
		{
			struct FinalizeTlistContext fctx;

			fctx.agg = agg;
			fctx.vars = &vars;
			if (!finalize_tlist_walker((Node *) te->expr, &fctx))
				goto done;
		}
	}
	supported = true;

done:
	pfree(vars.outer_attnos);
	return supported;
}

static TupleTableSlot *fill_input_slot(VectorAggState *state,
										   VectorBatch *batch, int row);

/*
 * Build the spool each phase (> 1) will be read back from.  A sorted next
 * phase gets a tuplesort keyed by its sort node; a plain phase needs only
 * insertion order, so it gets a tuplestore.
 */
static void
init_phase_spools(VectorAggState *state)
{
	AggState   *agg = state->agg;
	MemoryContext spoolcontext;
	int			i;

	if (!state->multi_phase)
		return;
	if (state->sort_slot == NULL)
	{
		MemoryContext oldcontext = MemoryContextSwitchTo(
			agg->ss.ps.state->es_query_cxt);

		/* registered in es_tupleTable: must outlive this VectorAggState */
		state->sort_slot = ExecInitExtraTupleSlot(agg->ss.ps.state,
												 agg->ss.ss_ScanTupleSlot->tts_tupleDescriptor,
												 &TTSOpsMinimalTuple);
		MemoryContextSwitchTo(oldcontext);
	}
	if (state->phase_sorts == NULL)
		state->phase_sorts = MemoryContextAllocZero(state->context,
										  sizeof(Tuplesortstate *) * agg->numphases);
	if (state->phase_stores == NULL)
		state->phase_stores = MemoryContextAllocZero(state->context,
										   sizeof(Tuplestorestate *) * agg->numphases);
	spoolcontext = MemoryContextSwitchTo(state->context);
	for (i = 2; i < agg->numphases; i++)
	{
		Sort	   *sortnode = agg->phases[i].sortnode;
		int			memkb = PlanStateOperatorMemKB((PlanState *) agg);

		if (sortnode != NULL)
			state->phase_sorts[i] = tuplesort_begin_heap(
				agg->ss.ss_ScanTupleSlot->tts_tupleDescriptor,
				sortnode->numCols, sortnode->sortColIdx,
				sortnode->sortOperators, sortnode->collations,
				sortnode->nullsFirst, memkb, NULL, false);
		else
			state->phase_stores[i] = tuplestore_begin_heap(true, false, memkb);
	}
	MemoryContextSwitchTo(spoolcontext);
}

/* Spool this batch's rows for the next phase, if there is one. */
static void
spool_next_phase(VectorAggState *state, VectorBatch *batch)
{
	int			next = state->current_phase_idx + 1;
	int			pos;

	if (!state->multi_phase || next >= state->agg->numphases)
		return;
	/* ProcessBatchSorted may be resumed on the same batch at every group
	 * boundary: spool each generation exactly once, or rows multiply. */
	if (state->spool_gen == batch->gen)
		return;
	state->spool_gen = batch->gen;
	for (pos = 0; pos < batch->nselected; pos++)
	{
		int			row = VectorBatchSelectedRow(batch, pos);
		TupleTableSlot *slot = fill_input_slot(state, batch, row);

		if (state->phase_sorts[next] != NULL)
			tuplesort_puttupleslot(state->phase_sorts[next], slot);
		else
			tuplestore_puttupleslot(state->phase_stores[next], slot);
	}
}

/* Fill the batch with the next rows of the current phase's spool. */
static bool
next_spool_batch(VectorAggState *state)
{
	VectorBatch *batch = state->batch;
	int			cur = state->current_phase_idx;
	int			nrows = 0;
	int			col;

	VectorBatchReset(batch);
	while (nrows < batch->capacity)
	{
		TupleTableSlot *slot = state->sort_slot;

		if (state->phase_sorts[cur] != NULL)
		{
			if (!tuplesort_gettupleslot(state->phase_sorts[cur], true, false, slot, NULL))
				break;
		}
		else
		{
			if (!tuplestore_gettupleslot(state->phase_stores[cur], true, false, slot))
				break;
		}
		slot_getallattrs(slot);
		for (col = 0; col < batch->ncols; col++)
		{
			Datum		value;
			bool		isnull;

			value = slot_getattr(slot, col + 1, &isnull);
			value = VectorBatchCopyDatum(batch, col, value, isnull);
			VectorBatchSetValue(batch, col, nrows, value, isnull);
		}
		nrows++;
	}
	VectorBatchSetRowCount(batch, nrows);
	if (nrows == 0)
		return false;
	for (col = 0; col < nrows; col++)
		batch->selection[col] = col;
	batch->nselected = nrows;
	state->position = 0;
	return true;
}

/* Advance to the next phase and make its spool readable. */
static bool
advance_phase(VectorAggState *state)
{
	AggState   *agg = state->agg;
	int			idx = state->current_phase_idx + 1;
	int			old_numsets = Max(agg->phase->numsets, 1);
	MemoryContext oldcontext;
	int			i;

	if (idx >= agg->numphases)
		return false;
	/* The old phase's transition values are no longer needed. */
	for (i = 0; i < old_numsets; i++)
		ReScanExprContext(agg->aggcontexts[i]);
	state->current_phase_idx = idx;
	agg->current_phase = idx;
	agg->phase = &agg->phases[idx];
	if (state->phase_sorts[idx] != NULL)
		tuplesort_performsort(state->phase_sorts[idx]);
	state->projected_set = -1;
	state->group_open = false;
	state->spool_done = false;
	/* No spool batch is active yet: fetch one on the next VectorAggNext. */
	state->position = state->batch->nselected;
	/* each phase reuses the same per-group storage: reinitialize it */
	oldcontext = MemoryContextSwitchTo(agg->ss.ps.state->es_query_cxt);
	VectorAggInitializeSet(agg, Max(agg->phase->numsets, 1));
	MemoryContextSwitchTo(oldcontext);
	return true;
}

VectorAggState *
VectorAggCreate(MemoryContext parent, AggState *agg, VectorBatch *batch)
{
	VectorAggState *state;
	MemoryContext oldcontext;
	int			transno;

	if (!VectorAggPlanIsSupported((Agg *) agg->ss.ps.plan))
		return NULL;

	state = MemoryContextAllocZero(parent, sizeof(VectorAggState));
	state->projected_set = -1;
	state->context = AllocSetContextCreate(parent, "Vector Aggregate",
										 ALLOCSET_DEFAULT_MINSIZE,
										 ALLOCSET_DEFAULT_INITSIZE,
										 ALLOCSET_DEFAULT_MAXSIZE);
	state->agg = agg;
	state->packed_i32 = batch->packed_i32;
	oldcontext = MemoryContextSwitchTo(state->context);
	if (!build_var_map((Agg *) agg->ss.ps.plan, state->context, &state->vars))
	{
		MemoryContextSwitchTo(oldcontext);
		VectorAggDestroy(state);
		return NULL;
	}
	state->vars.estate = agg->ss.ps.state;
	state->vars.parent = &agg->ss.ps;
	state->vars.allow_correlated = true;

	state->args = palloc0(sizeof(VectorExprState **) * agg->numtrans);
	state->filters = palloc0(sizeof(VectorExprState *) * agg->numtrans);
	state->filter_values = palloc0(sizeof(Datum *) * agg->numtrans);
	state->filter_nulls = palloc0(sizeof(uint64 *) * agg->numtrans);
	state->arg_values = palloc0(sizeof(Datum **) * agg->numtrans);
	state->arg_i32 = palloc0(sizeof(int32 **) * agg->numtrans);
	state->arg_i16 = palloc0(sizeof(int16 **) * agg->numtrans);
	state->arg_nulls = palloc0(sizeof(uint64 **) * agg->numtrans);
	state->filtered_selection = palloc(sizeof(int32) * batch->capacity);
	state->values = palloc0(sizeof(Datum *) * agg->numtrans);
	state->nulls = palloc0(sizeof(bool *) * agg->numtrans);
	for (transno = 0; transno < agg->numtrans; transno++)
	{
		AggStatePerTrans pertrans = &agg->pertrans[transno];
		int			nargs = list_length(pertrans->aggref->args);
		int			argno = 0;
		ListCell   *lc;

		state->args[transno] = palloc0(sizeof(VectorExprState *) * Max(nargs, 1));
		state->arg_values[transno] = palloc0(sizeof(Datum *) * Max(nargs, 1));
		state->arg_i32[transno] = palloc0(sizeof(int32 *) * Max(nargs, 1));
		state->arg_i16[transno] = palloc0(sizeof(int16 *) * Max(nargs, 1));
		state->arg_nulls[transno] = palloc0(sizeof(uint64 *) * Max(nargs, 1));
		state->values[transno] = palloc(sizeof(Datum) * Max(nargs, 1));
		state->nulls[transno] = palloc(sizeof(bool) * Max(nargs, 1));
		foreach(lc, pertrans->aggref->args)
		{
			TargetEntry *arg = (TargetEntry *) lfirst(lc);

			state->args[transno][argno] = VectorExprCompile(state->context,
														 arg->expr, batch,
														 &state->vars);
			if (state->args[transno][argno] == NULL)
			{
				MemoryContextSwitchTo(oldcontext);
				VectorAggDestroy(state);
				return NULL;
			}
			argno++;
		}
		if (pertrans->aggref->aggfilter != NULL)
		{
			state->filters[transno] = VectorExprCompile(state->context,
											 pertrans->aggref->aggfilter,
											 batch, &state->vars);
			if (state->filters[transno] == NULL)
			{
				MemoryContextSwitchTo(oldcontext);
				VectorAggDestroy(state);
				return NULL;
			}
		}
		Assert(argno == pertrans->numInputs);
	}
	state->deferred = palloc0(sizeof(bool) * agg->numtrans);
	for (transno = 0; transno < agg->numtrans; transno++)
	{
		AggStatePerTrans pertrans = &agg->pertrans[transno];

		/*
		 * sum()/avg(numeric) over a numeric arithmetic expression consumes
		 * the decoded fast-path result directly, without the Numeric
		 * materialization and re-parse per row.
		 */
		if (pertrans->numSortCols == 0 && pertrans->numTransInputs == 1 &&
			(pertrans->aggref->aggfnoid == VEC_AGG_SUM_NUMERIC ||
			 pertrans->aggref->aggfnoid == VEC_AGG_AVG_NUMERIC))
			state->deferred[transno] =
				VectorExprDeferNumericResult(state->args[transno][0]);
	}
	state->multi_phase = agg->aggstrategy == AGG_SORTED && agg->numphases > 2;
	state->is_combine = agg->aggsplit == AGGSPLIT_FINAL_DESERIAL;
	state->batch = batch;
	state->current_phase_idx = 1;
	if (state->multi_phase)
	{
		MemoryContext oldcontext = MemoryContextSwitchTo(
			agg->ss.ps.state->es_query_cxt);

		/* per-group storage is reused by all phases: initialize it once */
		VectorAggInitializeSet(agg, agg->maxsets);
		MemoryContextSwitchTo(oldcontext);
	}
	init_phase_spools(state);
	if (agg->aggstrategy == AGG_PLAIN)
	{
		/* nodeAgg owns its tuplesorts and frees them after the region state. */
		MemoryContextSwitchTo(agg->ss.ps.state->es_query_cxt);
		VectorAggInitializePlain(agg);
		MemoryContextSwitchTo(state->context);
	}
	if (agg->aggstrategy == AGG_MIXED)
	{
		/*
		 * Non-hash grouping sets live in pergroups[] in phase order, one per
		 * plain phase; initialize them like the standard executor would on
		 * the first input tuple of each phase.
		 */
		MemoryContextSwitchTo(agg->ss.ps.state->es_query_cxt);
		VectorAggInitializeSet(agg, agg->maxsets);
		MemoryContextSwitchTo(state->context);
	}
	MemoryContextSwitchTo(oldcontext);

	/*
	 * Spilled rows are read back by nodeAgg's refill machinery, which reads
	 * group keys and aggregate arguments alike from the input tuple.  Spill
	 * therefore needs the full input row, not just the grouping columns.
	 */
	state->input_slot = ExecInitExtraTupleSlot(agg->ss.ps.state,
											   agg->ss.ss_ScanTupleSlot->tts_tupleDescriptor,
											   &TTSOpsVirtual);
	return state;
}

void
VectorAggDestroy(VectorAggState *state)
{
	int			i;

	if (state == NULL)
		return;
	if (state->multi_phase)
	{
		for (i = 2; i < state->agg->numphases; i++)
		{
			if (state->phase_sorts[i] != NULL)
				tuplesort_end(state->phase_sorts[i]);
			if (state->phase_stores[i] != NULL)
				tuplestore_end(state->phase_stores[i]);
		}
	}
	MemoryContextDelete(state->context);
	pfree(state);
}

static void
eval_arguments(VectorAggState *state, VectorBatch *batch)
{
	AggState   *agg = state->agg;
	int			transno;

	for (transno = 0; transno < agg->numtrans; transno++)
	{
		int			nargs = agg->pertrans[transno].numInputs;
		int32	   *selection = batch->selection;
		int			nselected = batch->nselected;
		int			argno;

		/* A filter runs before the aggregate arguments, as in evaltrans. */
		if (state->filters[transno] != NULL)
		{
			int			pos;

			VectorExprEval(batch, state->filters[transno]);
			VectorExprRootValue(state->filters[transno],
								&state->filter_values[transno],
								&state->filter_nulls[transno]);
			nselected = 0;
			for (pos = 0; pos < batch->nselected; pos++)
			{
				int			row = VectorBatchSelectedRow(batch, pos);
				bool		isnull;
				Datum		value = VectorExprGetValue(state->filters[transno], row,
													 &isnull);

				if (!isnull && DatumGetBool(value))
					state->filtered_selection[nselected++] = row;
			}
			selection = state->filtered_selection;
		}

		for (argno = 0; argno < nargs; argno++)
		{
			VectorExprEvalSelection(batch, state->args[transno][argno],
									selection, nselected);
			VectorExprRootValue(state->args[transno][argno],
								&state->arg_values[transno][argno],
								&state->arg_nulls[transno][argno]);
			state->arg_i32[transno][argno] =
				VectorExprRootInt4(state->args[transno][argno]);
			state->arg_i16[transno][argno] =
				VectorExprRootInt2(state->args[transno][argno]);
		}
	}
}

/* Build the full input row of a batch position in the input layout. */
static TupleTableSlot *
fill_input_slot(VectorAggState *state, VectorBatch *batch, int row)
{
	TupleTableSlot *slot = state->input_slot;
	int			col;

	ExecClearTuple(slot);
	for (col = 0; col < state->vars.outer_ncols; col++)
	{
		AttrNumber	attno = state->vars.outer_attnos[col];

		slot->tts_values[col] = VectorBatchGetDatum(batch, attno - 1, row);
		slot->tts_isnull[col] = VectorBatchIsNull(batch, attno - 1, row);
	}
	ExecStoreVirtualTuple(slot);
	return slot;
}

/*
 * Fast open-addressing group table for hashed aggregates whose grouping
 * keys are all by-value fixed-width columns (at most VFAST_MAXKEYS of
 * 1/2/4/8 bytes).  Each entry packs a null-bit word plus the key datums
 * into VFAST_WORDS words and owns its per-group states in one array, so a
 * lookup is a probe and a word compare: no virtual slot round-trip, no
 * MinimalTuple copy, no per-group palloc.  When the table outgrows the
 * operator's memory limit it converts into the standard hash table - every
 * entry is reinserted and its state pointer becomes the new entry's
 * additional - and the rest of the stream runs through the ordinary spill
 * machinery.
 */
#define VFAST_MAXKEYS	4
#define VFAST_WORDS		(VFAST_MAXKEYS + 1)

/*
 * Plan-time census of a hashed aggregate's grouping keys: true when two
 * or more keys are outside the fast group table's reach (anything not
 * by-value 1/2/4/8 bytes).  Such an aggregate hashes per row through
 * the standard slot machinery: batched over a motion it loses to the
 * row executor, so the caller gates the shape out.
 */
bool
VectorAggWideGroupKeys(Agg *agg)
{
	TargetEntry *te;
	int			col;
	int			slow = 0;

	if (agg->aggstrategy != AGG_HASHED)
		return false;
	for (col = 0; col < agg->numCols; col++)
	{
		int16		typlen;
		bool		byval;

		if (agg->grpColIdx[col] <= 0 ||
			agg->grpColIdx[col] > list_length(outerPlan(agg)->targetlist))
			return false;
		te = (TargetEntry *) list_nth(outerPlan(agg)->targetlist,
									  agg->grpColIdx[col] - 1);
		if (te == NULL || te->resjunk)
			return false;
		get_typlenbyval(exprType((Node *) te->expr), &typlen, &byval);
		if (!byval || (typlen != 1 && typlen != 2 && typlen != 4 && typlen != 8))
			slow++;
	}
	/*
	 * One slow key costs about what it saves; two or more lose to the
	 * row executor consistently (query11's seven text keys, query67's
	 * rollup levels), while single-text-key shapes measured par and
	 * stayed net-positive vectorized (query06).
	 */
	return slow >= 2;
}


typedef struct VectorAggFastGroups
{
	uint64	   *keys;			/* nentries * VFAST_WORDS packed keys */
	AggStatePerGroupData *states;	/* flat: nslots * numtrans states */
	uint32	   *buckets;		/* entry index + 1; 0 = empty */
	uint32		nbuckets;		/* power of two */
	uint32		nentries;
	uint32		nslots;			/* allocated key/state entries */
	int			nkeys;
	int			ntrans;
	int			keycols[VFAST_MAXKEYS];	/* batch column index */
	int16		keylen[VFAST_MAXKEYS];
	Size		mem_used;
	bool		converted;
} VectorAggFastGroups;

static bool
fast_groups_supported(VectorAggState *state, VectorBatch *batch)
{
	AggState   *agg = state->agg;
	AggStatePerHash perhash = &agg->perhash[0];
	int			col;

	if (agg->aggstrategy != AGG_HASHED || agg->num_hashes != 1 ||
		agg->streaming || agg->hash_spill_mode)
		return false;
	if (perhash->numhashGrpCols < 1 ||
		perhash->numhashGrpCols > VFAST_MAXKEYS)
		return false;
	for (col = 0; col < perhash->numhashGrpCols; col++)
	{
		int			inputcol = perhash->hashGrpColIdxInput[col] - 1;
		AttrNumber	attno = state->vars.outer_attnos[inputcol];
		VectorColumn *column;
		int16		typlen;

		if (attno <= 0 || attno > batch->ncols)
			return false;
		column = &batch->columns[attno - 1];
		typlen = column->typlen;
		if (!column->typbyval ||
			(typlen != 1 && typlen != 2 && typlen != 4 && typlen != 8))
			return false;
	}
	return true;
}

static void
fast_groups_init(VectorAggState *state, VectorBatch *batch)
{
	AggState   *agg = state->agg;
	AggStatePerHash perhash = &agg->perhash[0];
	VectorAggFastGroups *t;
	uint32		nbuckets = 1024;
	double		ngroups = perhash->aggnode->numGroups;
	int			col;

	/* pre-size for the estimated group count, bounded for a first alloc */
	while (nbuckets < (1u << 22) && (double) nbuckets < ngroups)
		nbuckets <<= 1;

	t = MemoryContextAllocZero(state->context, sizeof(*t));
	t->nbuckets = nbuckets;
	t->buckets = MemoryContextAllocZero(state->context,
										sizeof(uint32) * nbuckets);
	t->nkeys = perhash->numhashGrpCols;
	t->ntrans = Max(agg->numtrans, 1);
	t->nslots = 1024;
	t->keys = MemoryContextAlloc(state->context,
								 sizeof(uint64) * VFAST_WORDS * t->nslots);
	t->states = MemoryContextAlloc(state->context,
								   sizeof(AggStatePerGroupData) * t->ntrans *
								   t->nslots);
	for (col = 0; col < t->nkeys; col++)
	{
		int			inputcol = perhash->hashGrpColIdxInput[col] - 1;
		AttrNumber	attno = state->vars.outer_attnos[inputcol];

		t->keycols[col] = attno - 1;
		t->keylen[col] = batch->columns[attno - 1].typlen;
	}
	t->mem_used = sizeof(uint32) * nbuckets +
		t->nslots * (sizeof(uint64) * VFAST_WORDS +
					 sizeof(AggStatePerGroupData) * t->ntrans);
	state->fast = t;
}

static uint32
fast_key_hash(const uint64 *key, int nwords)
{
	uint64		h = UINT64CONST(0x9E3779B97F4A7C15);
	int			c;

	for (c = 0; c < nwords; c++)
		h = (h ^ key[c]) * UINT64CONST(0xC2B7AE3D27D4EB4F);
	h ^= h >> 33;
	h *= UINT64CONST(0xFF51AFD7ED5581C3);
	h ^= h >> 33;
	return (uint32) (h >> 32);
}

/* Pack one row's keys and null bits; returns the hash of the packed words. */
static inline uint32
fast_key_pack(VectorAggFastGroups *t, VectorBatch *batch, int row,
			  uint64 *key)
{
	int			c;

	key[0] = 0;
	for (c = 0; c < t->nkeys; c++)
	{
		VectorColumn *column = &batch->columns[t->keycols[c]];
		Datum		d = 0;

		if ((column->nulls[row / 64] & ((uint64) 1 << (row % 64))) == 0)
		{
			d = VectorColumnGetDatum(column, row);
			switch (t->keylen[c])
			{
				case 8:
					key[c + 1] = (uint64) DatumGetInt64(d);
					continue;
				case 4:
					key[c + 1] = (uint64) (uint32) DatumGetInt32(d);
					continue;
				case 2:
					key[c + 1] = (uint64) (uint16) DatumGetInt16(d);
					continue;
				default:
					key[c + 1] = (uint64) (uint8) (uint32) d;
					continue;
			}
		}
		key[0] |= (uint64) 1 << c;
		key[c + 1] = 0;
	}
	return fast_key_hash(key, t->nkeys + 1);
}

static inline bool
fast_key_eq(const VectorAggFastGroups *t, uint32 e, const uint64 *key)
{
	const uint64 *k = &t->keys[(Size) e * VFAST_WORDS];
	int			c;

	for (c = 0; c <= t->nkeys; c++)
		if (k[c] != key[c])
			return false;
	return true;
}

/*
 * Move every entry into the standard hash table.  The states were filled
 * during the pass, so each new entry adopts its state pointer instead of
 * initializing fresh ones.
 */
static void
fast_convert(VectorAggState *state)
{
	VectorAggFastGroups *t = state->fast;
	AggState   *agg = state->agg;
	AggStatePerHash perhash = &agg->perhash[0];
	TupleTableSlot *hashslot = perhash->hashslot;
	uint32		i;
	int			c;

	for (i = 0; i < t->nentries; i++)
	{
		const uint64 *k = &t->keys[(Size) i * VFAST_WORDS];
		TupleHashEntry entry;
		bool		isnew;

		ExecClearTuple(hashslot);
		for (c = 0; c < t->nkeys; c++)
		{
			uint64		w = k[c + 1];

			switch (t->keylen[c])
			{
				case 8:
					hashslot->tts_values[c] = Int64GetDatum((int64) w);
					break;
				case 4:
					hashslot->tts_values[c] = Int32GetDatum((int32) (uint32) w);
					break;
				case 2:
					hashslot->tts_values[c] = Int16GetDatum((int16) (uint16) w);
					break;
				default:
					hashslot->tts_values[c] = Int8GetDatum((int8) (uint8) w);
					break;
			}
			hashslot->tts_isnull[c] = (k[0] >> c) & 1;
		}
		ExecStoreVirtualTuple(hashslot);
		entry = LookupTupleHashEntry(perhash->hashtable, hashslot, &isnew,
									 NULL);
		if (entry != NULL && isnew)
			entry->additional = &t->states[(Size) i * t->ntrans];
	}
	/*
	 * The groups were already counted in hash_ngroups_current when the fast
	 * table created them (VectorAggInitPergroup mirrors
	 * initialize_hash_entry); counting nentries here as well would double
	 * the group count and skew the spill-partition sizing.
	 */
	t->converted = true;
	t->mem_used -= sizeof(uint32) * t->nbuckets;
	pfree(t->buckets);
	t->buckets = NULL;
	state->group_valid = false;
}

/*
 * Append a new entry for key at the free bucket idx.  Growth invalidates
 * the per-row memo (the states array moves), never the caller's held
 * pointer: the entry is appended before lookup_group returns it.
 */
static AggStatePerGroup
fast_lookup_group(VectorAggState *state, VectorAggFastGroups *t,
				  VectorBatch *batch, int row)
{
	AggState   *agg = state->agg;
	uint64		key[VFAST_WORDS];
	uint32		h = fast_key_pack(t, batch, row, key);
	uint32		idx = h & (t->nbuckets - 1);
	uint32		e;

	for (;;)
	{
		e = t->buckets[idx];
		if (e == 0)
			break;
		if (fast_key_eq(t, e - 1, key))
			return &t->states[(Size) (e - 1) * t->ntrans];
		idx = (idx + 1) & (t->nbuckets - 1);
	}

	if (t->mem_used > agg->hash_mem_limit ||
		t->nentries >= agg->hash_ngroups_limit)
	{
		/* hand the groups over and spill the rest of the stream */
		fast_convert(state);
		VectorAggEnterSpillMode(state->agg);
		return NULL;
	}

	if (t->nentries >= t->nslots)
	{
		uint32		nslots = t->nslots * 2;
		Size		entrysz = sizeof(uint64) * VFAST_WORDS +
			sizeof(AggStatePerGroupData) * t->ntrans;

		t->keys = repalloc(t->keys, sizeof(uint64) * VFAST_WORDS * nslots);
		t->states = repalloc(t->states,
							 sizeof(AggStatePerGroupData) * t->ntrans * nslots);
		t->mem_used += (nslots - t->nslots) * entrysz;
		t->nslots = nslots;
		state->group_valid = false;
	}
	if (t->nentries + 1 > t->nbuckets * 3 / 4)
	{
		uint32		nb = t->nbuckets * 2;
		uint32	   *buckets = MemoryContextAllocZero(state->context,
													 sizeof(uint32) * nb);
		uint32		i;

		for (i = 0; i < t->nentries; i++)
		{
			uint32		h2 = fast_key_hash(&t->keys[(Size) i * VFAST_WORDS],
										   t->nkeys + 1);
			uint32		j = h2 & (nb - 1);

			while (buckets[j] != 0)
				j = (j + 1) & (nb - 1);
			buckets[j] = i + 1;
		}
		t->mem_used -= sizeof(uint32) * t->nbuckets;
		t->mem_used += sizeof(uint32) * nb;
		pfree(t->buckets);
		t->buckets = buckets;
		t->nbuckets = nb;
		idx = h & (nb - 1);
		while (t->buckets[idx] != 0)
			idx = (idx + 1) & (nb - 1);
	}

	e = t->nentries++;
	memcpy(&t->keys[(Size) e * VFAST_WORDS], key,
		   sizeof(uint64) * (t->nkeys + 1));
	t->buckets[idx] = e + 1;
	VectorAggInitPergroup(state->agg, &t->states[(Size) e * t->ntrans]);
	return &t->states[(Size) e * t->ntrans];
}

/*
 * Serve one group from the fast table: unpack its packed key into the scan
 * slot the way agg_retrieve_hash_table_in_memory unpacks the representative
 * tuple, and project through the standard per-entry path (which applies the
 * HAVING qual).  Surviving tables retrieve this way; a table that converted
 * mid-stream is served from the standard hash table instead.
 */
static TupleTableSlot *
fast_retrieve_next(VectorAggState *state)
{
	VectorAggFastGroups *t = state->fast;
	AggState   *agg = state->agg;
	AggStatePerHash perhash = &agg->perhash[0];
	TupleTableSlot *firstSlot = agg->ss.ss_ScanTupleSlot;

	while (state->fast_pos < t->nentries)
	{
		uint32		e = state->fast_pos++;
		const uint64 *k = &t->keys[(Size) e * VFAST_WORDS];
		TupleTableSlot *result;
		int			c;

		CHECK_FOR_INTERRUPTS();
		ResetExprContext(agg->ss.ps.ps_ExprContext);

		ExecClearTuple(firstSlot);
		memset(firstSlot->tts_isnull, true,
			   firstSlot->tts_tupleDescriptor->natts * sizeof(bool));
		for (c = 0; c < t->nkeys; c++)
		{
			int			varNumber = perhash->hashGrpColIdxInput[c] - 1;
			uint64		w = k[c + 1];

			if ((k[0] >> c) & 1)
			{
				firstSlot->tts_isnull[varNumber] = true;
				firstSlot->tts_values[varNumber] = (Datum) 0;
				continue;
			}
			firstSlot->tts_isnull[varNumber] = false;
			switch (t->keylen[c])
			{
				case 8:
					firstSlot->tts_values[varNumber] = Int64GetDatum((int64) w);
					break;
				case 4:
					firstSlot->tts_values[varNumber] =
						Int32GetDatum((int32) (uint32) w);
					break;
				case 2:
					firstSlot->tts_values[varNumber] =
						Int16GetDatum((int16) (uint16) w);
					break;
				default:
					firstSlot->tts_values[varNumber] =
						Int8GetDatum((int8) (uint8) w);
					break;
			}
		}
		ExecStoreVirtualTuple(firstSlot);

		result = VectorAggRetrieveHashedEntry(agg,
											  &t->states[(Size) e * t->ntrans]);
		if (result != NULL)
			return result;
	}
	return NULL;
}

static AggStatePerGroup
lookup_group(VectorAggState *state, VectorBatch *batch, int row)
{
	AggState   *agg = state->agg;
	AggStatePerHash perhash = &agg->perhash[0];
	TupleTableSlot *hashslot = perhash->hashslot;
	TupleHashEntry entry;
	bool		isnew = false;
	uint32		hash = 0;
	int			col;

	/*
	 * Grouping keys come in long runs, so a row equal to the previous one
	 * can reuse its group without the hash lookup.  The memo lives for one
	 * batch only: its key datums point into this batch's column arrays, and
	 * resetting it per batch also covers any hash table reset between
	 * batches.  Byte equality implies key equality; a miss just takes the
	 * full lookup.
	 */
	if (state->group_valid)
	{
		bool		hit = true;

		for (col = 0; col < perhash->numhashGrpCols; col++)
		{
			int			inputcol = perhash->hashGrpColIdxInput[col] - 1;
			AttrNumber	attno = state->vars.outer_attnos[inputcol];
			VectorColumn *column = &batch->columns[attno - 1];
			bool		isnull = VectorBatchIsNull(batch, attno - 1, row);

			if (isnull != state->group_key_nulls[col] ||
				(!isnull &&
				 !datumIsEqual(state->group_keys[col], VectorColumnGetDatum(column, row),
							   column->typbyval, column->typlen)))
			{
				hit = false;
				break;
			}
		}
		if (hit)
			return state->group_pergroup;
	}

	/*
	 * By-value fixed-width keys run through the fast table.  When it hits
	 * the operator's memory limit it converts into the standard table and
	 * this row falls through to the ordinary path (in spill mode it spills).
	 */
	if (state->fast != NULL && !state->fast->converted)
	{
		AggStatePerGroup pg = fast_lookup_group(state, state->fast, batch, row);

		if (pg != NULL)
		{
			if (state->group_keys == NULL)
			{
				state->group_keys = MemoryContextAlloc(state->context,
													   sizeof(Datum) * Max(perhash->numhashGrpCols, 1));
				state->group_key_nulls = MemoryContextAlloc(state->context,
															sizeof(bool) * Max(perhash->numhashGrpCols, 1));
			}
			for (col = 0; col < perhash->numhashGrpCols; col++)
			{
				int			inputcol = perhash->hashGrpColIdxInput[col] - 1;
				AttrNumber	attno = state->vars.outer_attnos[inputcol];

				state->group_keys[col] = VectorBatchGetDatum(batch, attno - 1, row);
				state->group_key_nulls[col] = VectorBatchIsNull(batch, attno - 1, row);
			}
			state->group_pergroup = pg;
			state->group_valid = true;
			return pg;
		}
	}

	ExecClearTuple(hashslot);
	for (col = 0; col < perhash->numhashGrpCols; col++)
	{
		int			inputcol = perhash->hashGrpColIdxInput[col] - 1;
		AttrNumber	attno = state->vars.outer_attnos[inputcol];

		hashslot->tts_values[col] = VectorBatchGetDatum(batch, attno - 1, row);
		hashslot->tts_isnull[col] = VectorBatchIsNull(batch, attno - 1, row);
	}
	ExecStoreVirtualTuple(hashslot);

	if (agg->hash_spill_mode)
		entry = LookupTupleHashEntry(perhash->hashtable, hashslot, NULL, &hash);
	else
		entry = LookupTupleHashEntry(perhash->hashtable, hashslot, &isnew, &hash);
	if (entry != NULL)
	{
		if (isnew)
			VectorAggInitializeEntry(agg, entry);

		/* remember this row's group for its successor */
		if (state->group_keys == NULL)
		{
			state->group_keys = MemoryContextAlloc(state->context,
												   sizeof(Datum) * Max(perhash->numhashGrpCols, 1));
			state->group_key_nulls = MemoryContextAlloc(state->context,
														sizeof(bool) * Max(perhash->numhashGrpCols, 1));
		}
		for (col = 0; col < perhash->numhashGrpCols; col++)
		{
			state->group_keys[col] = hashslot->tts_values[col];
			state->group_key_nulls[col] = hashslot->tts_isnull[col];
		}
		state->group_pergroup = (AggStatePerGroup) entry->additional;
		state->group_valid = true;
		return state->group_pergroup;
	}

	/* A new group that no longer fits is spilled and processed on refill. */
	if (agg->hash_spill_mode)
	{
		VectorAggSpillTuple(agg, fill_input_slot(state, batch, row), hash);
		return NULL;
	}
	ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
			 errmsg("vectorized hash aggregate could not create a group")));
	return NULL;
}

/*
 * AGG_MIXED: find the row's group in the hash table of one hashed grouping
 * set.  No memo here: keys are compared per set, and with several sets the
 * previous row's group of this set is rarely the same anyway.
 */
static AggStatePerGroup
lookup_group_mixed(VectorAggState *state, VectorBatch *batch, int row,
				   int setno)
{
	AggState   *agg = state->agg;
	AggStatePerHash perhash = &agg->perhash[setno];
	TupleTableSlot *hashslot = perhash->hashslot;
	TupleHashEntry entry;
	bool		isnew = false;
	uint32		hash = 0;
	int			col;

	ExecClearTuple(hashslot);
	for (col = 0; col < perhash->numhashGrpCols; col++)
	{
		int			inputcol = perhash->hashGrpColIdxInput[col] - 1;
		AttrNumber	attno = state->vars.outer_attnos[inputcol];

		hashslot->tts_values[col] = VectorBatchGetDatum(batch, attno - 1, row);
		hashslot->tts_isnull[col] = VectorBatchIsNull(batch, attno - 1, row);
	}
	ExecStoreVirtualTuple(hashslot);

	if (agg->hash_spill_mode)
		entry = LookupTupleHashEntry(perhash->hashtable, hashslot, NULL, &hash);
	else
		entry = LookupTupleHashEntry(perhash->hashtable, hashslot, &isnew, &hash);
	if (entry != NULL)
	{
		if (isnew)
			VectorAggInitializeEntry(agg, entry);
		return (AggStatePerGroup) entry->additional;
	}

	/* A new group that no longer fits is spilled and processed on refill. */
	if (agg->hash_spill_mode)
	{
		VectorAggSpillTupleSet(agg, fill_input_slot(state, batch, row), hash,
							   setno);
		return NULL;
	}
	ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
			 errmsg("vectorized hash aggregate could not create a group")));
	return NULL;
}

static bool
advance_byval_fast(VectorAggState *state, AggStatePerGroup pergroup,
				   int transno, int row)
{
	AggStatePerTrans pertrans = &state->agg->pertrans[transno];
	AggStatePerGroup pergroupstate = &pergroup[transno];
	Datum		value;
	bool		isnull;

	if (pertrans->numSortCols > 0)
		return false;
	if (pertrans->aggref->aggfnoid == VEC_AGG_COUNT_STAR)
	{
		pergroupstate->transValue++;
		return true;
	}
	if (pertrans->numTransInputs != 1)
		return false;

	value = state->values[transno][0];
	isnull = state->nulls[transno][0];
	if (pertrans->aggref->aggfnoid == VEC_AGG_COUNT_ANY)	/* count(any) */
	{
		if (!isnull)
			pergroupstate->transValue++;
		return true;
	}
	if (isnull)
		return pertrans->transfn.fn_strict;

	switch (pertrans->aggref->aggfnoid)
	{
		case VEC_AGG_AVG_INT4:
		case VEC_AGG_AVG_INT2:
		{
			Int8TransTypeData *transdata;

			if (pergroupstate->transValueIsNull)
				return false;
			transdata = (Int8TransTypeData *)
				ARR_DATA_PTR((ArrayType *) DatumGetPointer(pergroupstate->transValue));
			transdata->count++;
			if (pertrans->aggref->aggfnoid == VEC_AGG_AVG_INT2)
				transdata->sum += DatumGetInt16(value);
			else
				transdata->sum += DatumGetInt32(value);
			return true;
		}
		case VEC_AGG_AVG_NUMERIC:
		case VEC_AGG_SUM_NUMERIC:
		case VEC_AGG_STDDEV_INT8:
		case VEC_AGG_VAR_SAMP_INT8:
		case VEC_AGG_STDDEV_SAMP_INT8:
		case VEC_AGG_VAR_POP_INT8:
		case VEC_AGG_STDDEV_POP_INT8:
		{
			NumericAggState *aggstate;
			MemoryContext oldcontext;
			bool		calcSumX2 = pertrans->aggref->aggfnoid != VEC_AGG_AVG_NUMERIC &&
				pertrans->aggref->aggfnoid != VEC_AGG_SUM_NUMERIC;

			if (pergroupstate->transValueIsNull)
			{
				oldcontext = MemoryContextSwitchTo(
					state->agg->curaggcontext->ecxt_per_tuple_memory);
				aggstate = makeNumericAggStateCurrentContext(calcSumX2);
				MemoryContextSwitchTo(oldcontext);
				pergroupstate->transValue = PointerGetDatum(aggstate);
				pergroupstate->transValueIsNull = false;
				pergroupstate->noTransValue = false;
			}
			else
				aggstate = (NumericAggState *) DatumGetPointer(pergroupstate->transValue);
			if (!calcSumX2 && state->deferred[transno])
			{
				int64		mant;
				int			dscale;

				if (VectorExprGetDeferredNumeric(state->args[transno][0], row,
												 &mant, &dscale))
				{
					/* no tmpcontext allocations on this path */
					do_numeric_accum_fast(aggstate, mant, dscale);
					return true;
				}
				/* the row used the generic build/fmgr path: read its datum */
			}
			oldcontext = MemoryContextSwitchTo(
				state->agg->tmpcontext->ecxt_per_tuple_memory);
			if (calcSumX2)
				do_numeric_accum(aggstate,
								 DatumGetNumeric(DirectFunctionCall1(int8_numeric,
																	 value)));
			else
				/* decodes inline varlenas without a detoast copy */
				do_numeric_accum_datum(aggstate, value);
			MemoryContextSwitchTo(oldcontext);
			return true;
		}
		case VEC_AGG_STDDEV_INT4:
		case VEC_AGG_STDDEV_INT2:
		case VEC_AGG_VAR_SAMP_INT4:
		case VEC_AGG_VAR_SAMP_INT2:
		case VEC_AGG_STDDEV_SAMP_INT4:
		case VEC_AGG_STDDEV_SAMP_INT2:
		case VEC_AGG_VAR_POP_INT4:
		case VEC_AGG_VAR_POP_INT2:
		case VEC_AGG_STDDEV_POP_INT4:
		case VEC_AGG_STDDEV_POP_INT2:
		case VEC_AGG_AVG_INT8:
		case VEC_AGG_SUM_INT8:
#ifdef HAVE_INT128
		{
			Int128AggState *aggstate;
			MemoryContext oldcontext;
			bool		calcSumX2 = pertrans->aggref->aggfnoid != VEC_AGG_AVG_INT8 &&
				pertrans->aggref->aggfnoid != VEC_AGG_SUM_INT8;
			int128		newval;

			if (pergroupstate->transValueIsNull)
			{
				oldcontext = MemoryContextSwitchTo(
					state->agg->curaggcontext->ecxt_per_tuple_memory);
				aggstate = makeInt128AggStateCurrentContext(calcSumX2);
				MemoryContextSwitchTo(oldcontext);
				pergroupstate->transValue = PointerGetDatum(aggstate);
				pergroupstate->transValueIsNull = false;
				pergroupstate->noTransValue = false;
			}
			else
				aggstate = (Int128AggState *) DatumGetPointer(pergroupstate->transValue);
			switch (pertrans->aggref->aggfnoid)
			{
				case VEC_AGG_STDDEV_INT2:
				case VEC_AGG_VAR_SAMP_INT2:
				case VEC_AGG_STDDEV_SAMP_INT2:
				case VEC_AGG_VAR_POP_INT2:
				case VEC_AGG_STDDEV_POP_INT2:
					newval = (int128) DatumGetInt16(value);
					break;
				case VEC_AGG_STDDEV_INT4:
				case VEC_AGG_VAR_SAMP_INT4:
				case VEC_AGG_STDDEV_SAMP_INT4:
				case VEC_AGG_VAR_POP_INT4:
				case VEC_AGG_STDDEV_POP_INT4:
					newval = (int128) DatumGetInt32(value);
					break;
				default:
					newval = (int128) DatumGetInt64(value);
					break;
			}
			aggstate->sumX += newval;
			if (calcSumX2)
				aggstate->sumX2 += newval * newval;
			aggstate->N++;
			return true;
		}
#else
			return false;
#endif
		case VEC_AGG_SUM_INT4:
			pergroupstate->transValue += DatumGetInt32(value);
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_SUM_INT2:
			pergroupstate->transValue += (int64) DatumGetInt16(value);
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_SUM_FLOAT4:
			/*
			 * sum(float4) has transtype float4 and no finalfn: the
			 * accumulation must round to float4 on every step, exactly like
			 * float4pl, and the transition value must be a float4 datum.
			 */
			if (pergroupstate->noTransValue)
				pergroupstate->transValue = Float4GetDatum(DatumGetFloat4(value));
			else
				pergroupstate->transValue = Float4GetDatum(float4_pl(
					DatumGetFloat4(pergroupstate->transValue),
					DatumGetFloat4(value)));
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_SUM_FLOAT8:
			if (pergroupstate->noTransValue)
				pergroupstate->transValue = value;
			else
				pergroupstate->transValue = Float8GetDatum(
					DatumGetFloat8(pergroupstate->transValue) + DatumGetFloat8(value));
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MAX_INT8:
			if (pergroupstate->noTransValue ||
				DatumGetInt64(pergroupstate->transValue) < DatumGetInt64(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MAX_INT4:
			if (pergroupstate->noTransValue ||
				DatumGetInt32(pergroupstate->transValue) < DatumGetInt32(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MAX_INT2:
			if (pergroupstate->noTransValue ||
				DatumGetInt16(pergroupstate->transValue) < DatumGetInt16(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MAX_FLOAT4:
			if (pergroupstate->noTransValue ||
				float4_cmp_internal(DatumGetFloat4(pergroupstate->transValue),
									DatumGetFloat4(value)) < 0)
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MAX_FLOAT8:
			if (pergroupstate->noTransValue ||
				float8_cmp_internal(DatumGetFloat8(pergroupstate->transValue),
									DatumGetFloat8(value)) < 0)
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MAX_DATE:
			if (pergroupstate->noTransValue ||
				DatumGetInt32(pergroupstate->transValue) < DatumGetInt32(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MAX_TIMESTAMP:
		case VEC_AGG_MAX_TIMESTAMPTZ:
			if (pergroupstate->noTransValue ||
				DatumGetInt64(pergroupstate->transValue) < DatumGetInt64(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MIN_INT8:
			if (pergroupstate->noTransValue ||
				DatumGetInt64(pergroupstate->transValue) > DatumGetInt64(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MIN_INT4:
			if (pergroupstate->noTransValue ||
				DatumGetInt32(pergroupstate->transValue) > DatumGetInt32(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MIN_INT2:
			if (pergroupstate->noTransValue ||
				DatumGetInt16(pergroupstate->transValue) > DatumGetInt16(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MIN_FLOAT4:
			if (pergroupstate->noTransValue ||
				float4_cmp_internal(DatumGetFloat4(pergroupstate->transValue),
									DatumGetFloat4(value)) > 0)
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MIN_FLOAT8:
			if (pergroupstate->noTransValue ||
				float8_cmp_internal(DatumGetFloat8(pergroupstate->transValue),
									DatumGetFloat8(value)) > 0)
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MIN_DATE:
			if (pergroupstate->noTransValue ||
				DatumGetInt32(pergroupstate->transValue) > DatumGetInt32(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case VEC_AGG_MIN_TIMESTAMP:
		case VEC_AGG_MIN_TIMESTAMPTZ:
			if (pergroupstate->noTransValue ||
				DatumGetInt64(pergroupstate->transValue) > DatumGetInt64(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		default:
			return false;
	}
}

static inline float4
packed_float4(int32 value)
{
	uint32		bits = (uint32) value;
	float4		result;

	memcpy(&result, &bits, sizeof(result));
	return result;
}

static void
advance_packed_row(VectorAggState *state, AggStatePerGroup pergroup, int row)
{
	AggState   *agg = state->agg;
	int			transno;

	for (transno = 0; transno < agg->numtrans; transno++)
	{
		AggStatePerTrans pertrans = &agg->pertrans[transno];
		AggStatePerGroup pergroupstate = &pergroup[transno];
		Oid			fnoid = pertrans->aggref->aggfnoid;
		bool		is_int2 = fnoid == VEC_AGG_AVG_INT2 ||
			fnoid == VEC_AGG_SUM_INT2 || fnoid == VEC_AGG_MIN_INT2 ||
			fnoid == VEC_AGG_MAX_INT2;
		int32		value;

		if ((state->arg_nulls[transno][0][row / 64] >> (row % 64)) & 1)
			continue;
		value = is_int2 ? state->arg_i16[transno][0][row] :
			state->arg_i32[transno][0][row];
		switch (fnoid)
		{
			case VEC_AGG_AVG_INT2:
			case VEC_AGG_AVG_INT4:
			{
				Int8TransTypeData *transdata;

				Assert(!pergroupstate->transValueIsNull);
				transdata = (Int8TransTypeData *)
					ARR_DATA_PTR((ArrayType *) DatumGetPointer(pergroupstate->transValue));
				transdata->count++;
				transdata->sum += value;
				break;
			}
			case VEC_AGG_SUM_INT2:
			case VEC_AGG_SUM_INT4:
				pergroupstate->transValue += value;
				pergroupstate->noTransValue = false;
				pergroupstate->transValueIsNull = false;
				break;
			case VEC_AGG_MAX_INT2:
			case VEC_AGG_MAX_INT4:
			case VEC_AGG_MAX_DATE:
				if (pergroupstate->noTransValue ||
					DatumGetInt32(pergroupstate->transValue) < value)
					pergroupstate->transValue = Int32GetDatum(value);
				pergroupstate->noTransValue = false;
				pergroupstate->transValueIsNull = false;
				break;
			case VEC_AGG_MIN_INT2:
			case VEC_AGG_MIN_INT4:
			case VEC_AGG_MIN_DATE:
				if (pergroupstate->noTransValue ||
					DatumGetInt32(pergroupstate->transValue) > value)
					pergroupstate->transValue = Int32GetDatum(value);
				pergroupstate->noTransValue = false;
				pergroupstate->transValueIsNull = false;
				break;
			case VEC_AGG_SUM_FLOAT4:
			{
				float4		input = packed_float4(value);

				/* transtype is float4: round on every step like float4pl */
				if (pergroupstate->noTransValue)
					pergroupstate->transValue = Float4GetDatum(input);
				else
					pergroupstate->transValue = Float4GetDatum(float4_pl(
						DatumGetFloat4(pergroupstate->transValue), input));
				pergroupstate->noTransValue = false;
				pergroupstate->transValueIsNull = false;
				break;
			}
			case VEC_AGG_MAX_FLOAT4:
				if (pergroupstate->noTransValue ||
					float4_cmp_internal(DatumGetFloat4(pergroupstate->transValue),
						packed_float4(value)) < 0)
					pergroupstate->transValue = Float4GetDatum(packed_float4(value));
				pergroupstate->noTransValue = false;
				pergroupstate->transValueIsNull = false;
				break;
			case VEC_AGG_MIN_FLOAT4:
				if (pergroupstate->noTransValue ||
					float4_cmp_internal(DatumGetFloat4(pergroupstate->transValue),
						packed_float4(value)) > 0)
					pergroupstate->transValue = Float4GetDatum(packed_float4(value));
				pergroupstate->noTransValue = false;
				pergroupstate->transValueIsNull = false;
				break;
			default:
				pg_unreachable();
		}
	}
}

/*
 * A combine transition consumes the serialized partial state carried by
 * the input column: deserialize it, then run the combine through the
 * regular transition bridge.  Mirrors the row executor's
 * EEOP_AGG_STRICT_DESERIALIZE / EEOP_AGG_DESERIALIZE steps, including
 * the strict-deserialize bailout on a NULL input.
 */
static void
advance_combine_row(AggState *agg, AggStatePerTrans pertrans,
					AggStatePerGroup pergroupstate, int transno,
					Datum input, bool isnull)
{
	Datum		value = input;
	bool		valuenull = isnull;
	Datum		combine_values[1];
	bool		combine_nulls[1];

	if (OidIsValid(pertrans->deserialfn_oid))
	{
		FunctionCallInfo ds_fcinfo = pertrans->deserialfn_fcinfo;
		MemoryContext oldcontext;

		if (pertrans->deserialfn.fn_strict && isnull)
			return;
		ds_fcinfo->args[0].value = input;
		ds_fcinfo->args[0].isnull = isnull;
		ds_fcinfo->args[1].value = (Datum) 0;
		ds_fcinfo->args[1].isnull = false;
		oldcontext = MemoryContextSwitchTo(
			agg->curaggcontext->ecxt_per_tuple_memory);
		ds_fcinfo->isnull = false;
		value = FunctionCallInvoke(ds_fcinfo);
		valuenull = ds_fcinfo->isnull;
		MemoryContextSwitchTo(oldcontext);
	}
	combine_values[0] = value;
	combine_nulls[0] = valuenull;
	VectorAggAdvanceTransition(agg, pergroupstate, transno,
							   combine_values, combine_nulls);
}

static void
advance_row(VectorAggState *state, AggStatePerGroup pergroup, int row)
{
	AggState   *agg = state->agg;
	int			transno;

	if (state->packed_i32)
	{
		advance_packed_row(state, pergroup, row);
		return;
	}

	for (transno = 0; transno < agg->numtrans; transno++)
	{
		AggStatePerTrans pertrans = &agg->pertrans[transno];
		int			nargs = pertrans->numInputs;
		int			argno;

		/* eval_arguments() stashed the root arrays: read rows inline */
		if (state->filters[transno] != NULL)
		{
			if ((state->filter_nulls[transno][row / 64] >> (row % 64)) & 1 ||
				!DatumGetBool(state->filter_values[transno][row]))
				continue;
		}
		for (argno = 0; argno < nargs; argno++)
		{
			state->values[transno][argno] =
				state->arg_values[transno][argno][row];
			state->nulls[transno][argno] =
				(state->arg_nulls[transno][argno][row / 64] >> (row % 64)) & 1;
		}
		if (state->is_combine)
		{
			/*
			 * A combine transition: the input column carries the
			 * (optionally serialized) partial state from below.
			 * Deserialize and run the combine through the regular
			 * transition bridge; the by-value fast paths do not apply.
			 */
			advance_combine_row(agg, pertrans, pergroup, transno,
								state->values[transno][0],
								state->nulls[transno][0]);
			continue;
		}
		if (pertrans->numSortCols == 0 &&
			advance_byval_fast(state, pergroup, transno, row))
			continue;
		if (pertrans->numSortCols > 0)
		{
			Assert(agg->aggstrategy == AGG_PLAIN || agg->aggstrategy == AGG_SORTED);
			if (nargs == 1)
				tuplesort_putdatum(pertrans->sortstates[agg->current_set],
								   state->values[transno][0],
								   state->nulls[transno][0]);
			else
			{
				TupleTableSlot *sortslot = pertrans->sortslot;

				ExecClearTuple(sortslot);
				for (argno = 0; argno < nargs; argno++)
				{
					sortslot->tts_values[argno] = state->values[transno][argno];
					sortslot->tts_isnull[argno] = state->nulls[transno][argno];
				}
				sortslot->tts_nvalid = nargs;
				ExecStoreVirtualTuple(sortslot);
				tuplesort_puttupleslot(pertrans->sortstates[agg->current_set], sortslot);
			}
		}
		else
			VectorAggAdvanceTransition(agg, pergroup, transno,
									   state->values[transno], state->nulls[transno]);
	}
}

static void
advance_sorted_row(VectorAggState *state, int row)
{
	AggState   *agg = state->agg;
	int			numsets = Max(agg->phase->numsets, 1);
	int			setno;

	/* Every input row contributes to every grouping set in this phase. */
	for (setno = 0; setno < numsets; setno++)
	{
		VectorAggSelectSet(agg, setno);
		advance_row(state, agg->pergroups[setno], row);
	}
}

void
VectorAggProcessBatch(VectorAggState *state, VectorBatch *batch)
{
	AggState   *agg = state->agg;
	int			pos;

	if (batch->nselected == 0)
		return;

	if (state->fast == NULL && !state->fast_rejected)
	{
		if (fast_groups_supported(state, batch))
			fast_groups_init(state, batch);
		else
			state->fast_rejected = true;
	}

	/* the group memo's key datums die with the previous batch */
	state->group_valid = false;
	eval_arguments(state, batch);
	for (pos = 0; pos < batch->nselected; pos++)
	{
		int			row = VectorBatchSelectedRow(batch, pos);
		AggStatePerGroup pergroup;

		if ((pos & 0xff) == 0)
			CHECK_FOR_INTERRUPTS();
		if (agg->aggstrategy == AGG_HASHED)
		{
			pergroup = lookup_group(state, batch, row);
			if (pergroup == NULL)
				continue;
		}
		else if (agg->aggstrategy == AGG_MIXED)
		{
			int			setno;

			/* hashed grouping sets: one lookup per set per row */
			for (setno = 0; setno < agg->num_hashes; setno++)
			{
				VectorAggSelectHashSet(agg, setno);
				pergroup = lookup_group_mixed(state, batch, row, setno);
				if (pergroup != NULL)
					advance_row(state, pergroup, row);
				/* a spilled row still feeds the plain grouping sets below */
			}
			/* plain grouping sets: every row advances every one of them */
			for (setno = 0; setno < agg->maxsets; setno++)
			{
				VectorAggSelectSet(agg, setno);
				advance_row(state, agg->pergroups[setno], row);
			}
			continue;
		}
		else if (agg->aggstrategy == AGG_PLAIN && agg->phase->numsets > 1)
		{
			int			setno;

			/*
			 * A plain aggregate can still carry grouping sets (no hashed,
			 * no sorted set): every input row contributes to every set of
			 * the phase, exactly as the sorted multi-set advance does.
			 */
			for (setno = 0; setno < agg->phase->numsets; setno++)
			{
				VectorAggSelectSet(agg, setno);
				advance_row(state, agg->pergroups[setno], row);
			}
			continue;
		}
		else
			pergroup = agg->pergroups[0];

		advance_row(state, pergroup, row);
	}
	/* Transition garbage is bounded by the batch size; reset once. */
	ResetExprContext(agg->tmpcontext);
}

static void
start_sorted_group(VectorAggState *state, VectorBatch *batch, int row)
{
	AggState   *agg = state->agg;
	TupleTableSlot *input;
	HeapTuple	first;
	MemoryContext oldcontext;

	int			numsets = Max(agg->phase->numsets, 1);
	int			numReset = state->projected_set < 0 ? numsets :
		Min(state->projected_set + 1, numsets);
	int			i;

	for (i = 0; i < numReset; i++)
		ReScanExprContext(agg->aggcontexts[i]);
	input = fill_input_slot(state, batch, row);
	first = ExecCopySlotHeapTuple(input);
	ExecForceStoreHeapTuple(first, agg->ss.ss_ScanTupleSlot, true);
	oldcontext = MemoryContextSwitchTo(agg->ss.ps.state->es_query_cxt);
	VectorAggInitializeSet(agg, numReset);
	MemoryContextSwitchTo(oldcontext);
	state->projected_set = 0;
	state->group_open = true;
}

TupleTableSlot *
VectorAggProcessBatchSorted(VectorAggState *state, VectorBatch *batch,
							int *position)
{
	AggState   *agg = state->agg;
	int			numsets = Max(agg->phase->numsets, 1);

	if (batch->nselected == 0)
		return NULL;

	spool_next_phase(state, batch);
	/* Evaluate once per batch content, not once per call: this driver
	 * returns after every output row and is re-entered per row by
	 * row-wise consumers. */
	if (state->args_batch != batch || state->args_gen != batch->gen)
	{
		eval_arguments(state, batch);
		state->args_batch = batch;
		state->args_gen = batch->gen;
	}
	while (*position < batch->nselected)
	{
		int			row = VectorBatchSelectedRow(batch, *position);
		TupleTableSlot *input;
		TupleTableSlot *result;

		if ((*position & 0xff) == 0)
			CHECK_FOR_INTERRUPTS();
		input = fill_input_slot(state, batch, row);

		/*
		 * A pending row can cross boundaries of several enclosing grouping
		 * sets.  Project one set per call, leaving the row pending until all
		 * crossed sets have been emitted.
		 */
		if (!state->group_open && state->projected_set >= 0 &&
			state->projected_set + 1 < numsets)
		{
			int			nextSetSize = agg->phase->gset_lengths[state->projected_set + 1];

			if (nextSetSize > 0 &&
				!VectorAggSameGroup(agg, input, nextSetSize))
			{
				state->projected_set++;
				result = VectorAggRetrieveGroup(agg, state->projected_set);
				ResetExprContext(agg->tmpcontext);
				if (result != NULL)
					return result;
				continue;
			}
		}

		if (state->group_open &&
			!VectorAggSameGroup(agg, input, agg->phase->aggnode->numCols))
		{
			result = VectorAggRetrieveGroup(agg, state->projected_set);
			state->group_open = false;
			ResetExprContext(agg->tmpcontext);
			if (result != NULL)
				return result;	/* resume at this row, the next group's first */
			continue;
		}
		if (!state->group_open)
			start_sorted_group(state, batch, row);
		advance_sorted_row(state, row);
		(*position)++;
	}
	ResetExprContext(agg->tmpcontext);
	return NULL;
}

void
VectorAggFinish(VectorAggState *state)
{
	if (state->input_done)
		return;
	VectorAggFinishInput(state->agg);
	state->input_done = true;
}

static void
initialize_empty_sorted_set(VectorAggState *state)
{
	AggState   *agg = state->agg;
	int			numsets = Max(agg->phase->numsets, 1);
	int			numReset = state->projected_set < 0 ? numsets :
		Min(state->projected_set + 1, numsets);
	MemoryContext oldcontext;
	int			i;

	for (i = 0; i < numReset; i++)
		ReScanExprContext(agg->aggcontexts[i]);
	oldcontext = MemoryContextSwitchTo(agg->ss.ps.state->es_query_cxt);
	VectorAggInitializeSet(agg, numReset);
	MemoryContextSwitchTo(oldcontext);
}

TupleTableSlot *
VectorAggNext(VectorAggState *state)
{
	TupleTableSlot *result;

	Assert(state->input_done);
	if (state->output_done)
		return NULL;

	if (state->multi_phase)
	{
		AggState   *agg = state->agg;

		for (;;)
		{
			if (!state->spool_done)
			{
				if (state->current_phase_idx == 1)
					state->spool_done = true;	/* streaming input is over */
				else
				{
					/* Resume a partially consumed spool batch before fetching more. */
					if (state->position >= state->batch->nselected &&
						!next_spool_batch(state))
						state->spool_done = true;
					if (!state->spool_done)
					{
						if (agg->phase->aggnode->numCols == 0)
						{
							int			pos;

							spool_next_phase(state, state->batch);
							eval_arguments(state, state->batch);
							VectorAggSelectSet(agg, 0);
							for (pos = 0; pos < state->batch->nselected; pos++)
								advance_row(state, agg->pergroups[0],
											VectorBatchSelectedRow(state->batch, pos));
							state->position = state->batch->nselected;
							ResetExprContext(agg->tmpcontext);
							result = NULL;
						}
						else
							result = VectorAggProcessBatchSorted(state, state->batch,
																 &state->position);
						if (result != NULL)
							return result;
						continue;
					}
				}
			}

			/* current phase is exhausted: flush its remaining groups */
			if (agg->phase->aggnode->numCols == 0)
			{
				if (state->projected_set < 0)
				{
					state->projected_set = 0;
					return VectorAggRetrievePlainSet(agg,
												 state->current_phase_idx, 0, 0);
				}
			}
			else
			{
				int			numsets = Max(agg->phase->numsets, 1);
				bool		had_input = state->projected_set >= 0;

				if (state->group_open)
				{
					ResetExprContext(agg->tmpcontext);
					result = VectorAggRetrieveGroup(agg, state->projected_set);
					state->group_open = false;
					if (result != NULL)
						return result;
				}
				while (state->projected_set + 1 < numsets)
				{
					int			nextset = state->projected_set + 1;

					if (!had_input && agg->phase->aggnode->numCols > 0 &&
						(agg->phase->gset_lengths == NULL ||
						 agg->phase->gset_lengths[nextset] > 0))
					{
						state->projected_set = nextset;
						continue;
					}
					state->projected_set = nextset;
					if (!had_input)
						initialize_empty_sorted_set(state);
					result = VectorAggRetrieveGroup(agg, state->projected_set);
					if (result != NULL)
						return result;
				}
			}
			if (!advance_phase(state))
			{
				state->output_done = true;
				return NULL;
			}
		}
	}

	if (state->agg->aggstrategy == AGG_SORTED)
	{
		int			numsets = Max(state->agg->phase->numsets, 1);
		bool		had_input = state->projected_set >= 0;

		if (state->group_open)
		{
			ResetExprContext(state->agg->tmpcontext);
			result = VectorAggRetrieveGroup(state->agg, state->projected_set);
			state->group_open = false;
			if (result != NULL)
				return result;
		}
		while (state->projected_set + 1 < numsets)
		{
			int			nextset = state->projected_set + 1;

			/* Empty input has rows only for grouping sets of size zero. */
			if (!had_input && state->agg->phase->aggnode->numCols > 0 &&
				(state->agg->phase->gset_lengths == NULL ||
				 state->agg->phase->gset_lengths[nextset] > 0))
			{
				state->projected_set = nextset;
				continue;
			}
			state->projected_set = nextset;
			if (!had_input)
				initialize_empty_sorted_set(state);
			result = VectorAggRetrieveGroup(state->agg, state->projected_set);
			if (result != NULL)
				return result;
		}
		state->output_done = true;
		return NULL;
	}

	if (state->agg->aggstrategy == AGG_MIXED)
	{
		AggState   *agg = state->agg;

		/*
		 * Plain grouping sets go first, in phase order, as in nodeAgg.
		 * Every non-hash grouping set lives in exactly one plain phase
		 * (all same-prefix sets share their phase, e.g. all the empty
		 * sets of a GROUPING SETS list), so the walk covers each phase's
		 * own numsets rather than assuming one phase per set - an
		 * assumed-per-set phase index runs off the end of phases[] and
		 * crashes once a phase holds two or more sets.
		 */
		while (!state->mixed_hashed_started &&
			   state->mixed_plain_index < agg->maxsets)
		{
			int			phaseidx;
			int			phaseset;

			for (phaseidx = 1; phaseidx < agg->numphases; phaseidx++)
			{
				int			numsets = Max(agg->phases[phaseidx].numsets, 1);

				/*
				 * Prepare projection and finalize both index the phase's
				 * per-set arrays by the GLOBAL grouping-set number, the
				 * same number pergroups[] and select_current_set use.
				 */
				for (phaseset = 0; phaseset < numsets; phaseset++)
				{
					int			k = state->mixed_plain_index++;

					result = VectorAggRetrievePlainSet(agg, phaseidx,
													   k, k);
					if (result != NULL)
						return result;
				}
			}
		}
		if (!state->mixed_hashed_started)
		{
			agg->phase = &agg->phases[0];
			ResetTupleHashIterator(agg->perhash[0].hashtable,
								   &agg->perhash[0].hashiter);
			VectorAggSelectHashSet(agg, 0);
			state->mixed_hashed_started = true;
		}
		result = VectorAggRetrieveHashed(agg);
		if (result == NULL)
			state->output_done = true;
		return result;
	}

	if (state->agg->aggstrategy == AGG_HASHED)
	{
		/*
		 * A table that survived the whole input serves its groups directly;
		 * one that converted mid-stream lives in the standard hash table.
		 */
		if (state->fast != NULL && !state->fast->converted)
			return fast_retrieve_next(state);
		return VectorAggRetrieveHashed(state->agg);
	}

	/*
	 * An AGG_PLAIN node can still carry grouping sets (GROUPING SETS with
	 * no hashed and no sorted set): each set projects one empty-group row
	 * from its own pergroup array.  VectorAggRetrievePlain only knows the
	 * first set, so walk the phase's sets here; initialize_aggregates
	 * (numReset 0) has already initialized all of them.
	 */
	if (state->agg->aggstrategy == AGG_PLAIN &&
		state->agg->phase->numsets > 1)
	{
		int			setno = state->plain_set_index;
		int			phaseidx = state->agg->phase - state->agg->phases;

		if (setno >= state->agg->phase->numsets)
		{
			state->output_done = true;
			return NULL;
		}
		state->plain_set_index++;
		return VectorAggRetrievePlainSet(state->agg, phaseidx, setno, setno);
	}

	state->output_done = true;
	return VectorAggRetrievePlain(state->agg);
}

void
VectorAggReset(VectorAggState *state)
{
	ExecReScanAgg(state->agg);
	/* the rescanned hash table drops the memoized group */
	state->group_valid = false;
	if (state->fast != NULL)
	{
		VectorAggFastGroups *t = state->fast;

		if (t->buckets != NULL)
			memset(t->buckets, 0, sizeof(uint32) * t->nbuckets);
		t->nentries = 0;
		t->converted = false;
		state->fast_pos = 0;
		t->mem_used = sizeof(uint32) * t->nbuckets +
			t->nslots * (sizeof(uint64) * VFAST_WORDS +
						 sizeof(AggStatePerGroupData) * t->ntrans);
	}
	if (state->agg->aggstrategy == AGG_PLAIN)
	{
		MemoryContext oldcontext = MemoryContextSwitchTo(
			state->agg->ss.ps.state->es_query_cxt);

		VectorAggInitializePlain(state->agg);
		MemoryContextSwitchTo(oldcontext);
	}
	if (state->agg->aggstrategy == AGG_MIXED)
	{
		MemoryContext oldcontext = MemoryContextSwitchTo(
			state->agg->ss.ps.state->es_query_cxt);

		VectorAggInitializeSet(state->agg, state->agg->maxsets);
		MemoryContextSwitchTo(oldcontext);
		state->mixed_plain_index = 0;
		state->mixed_hashed_started = false;
	}
	/* the multi-set plain walk restarts from the first grouping set */
	state->plain_set_index = 0;
	/* A reused hash table must be iterated, not populated a second time. */
	state->input_done = state->agg->aggstrategy == AGG_HASHED &&
		state->agg->table_filled;
	if (state->multi_phase)
	{
		AggState   *agg = state->agg;
		int			i;

		for (i = 2; i < agg->numphases; i++)
		{
			if (state->phase_sorts[i] != NULL)
			{
				tuplesort_end(state->phase_sorts[i]);
				state->phase_sorts[i] = NULL;
			}
			if (state->phase_stores[i] != NULL)
			{
				tuplestore_end(state->phase_stores[i]);
				state->phase_stores[i] = NULL;
			}
		}
		state->current_phase_idx = 1;
		state->spool_done = false;
		agg->current_phase = 1;
		agg->phase = &agg->phases[1];
		init_phase_spools(state);
	}
	state->group_open = false;
	state->projected_set = -1;
	state->output_done = false;
	state->args_batch = NULL;
	state->args_gen = 0;
}
