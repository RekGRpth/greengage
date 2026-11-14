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
#include "executor/vectorAgg.h"
#include "executor/vectorExpr.h"
#include "miscadmin.h"
#include "utils/array.h"
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
	bool		packed_int4;
	AggState   *agg;
	VectorExprVarMap vars;
	VectorExprState ***args;
	VectorExprState **filters;
	bool	   *deferred;		/* transno: arg yields decoded numerics */
	Datum	  **filter_values;	/* per transno: root arrays of the filter */
	uint64	 **filter_nulls;
	Datum	 ***arg_values;		/* per transno/argno: root value arrays */
	int32	  ***arg_i32;		/* packed INT4 root, when present */
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
	int			projected_set;
	bool		group_open;
	bool		input_done;
	bool		output_done;
	/* AGG_MIXED: plain (non-hash) grouping sets are emitted before hashed */
	int			mixed_plain_index;
	bool		mixed_hashed_started;
	/* multi-phase sorted aggregation: per-phase spools and phase driver */
	bool		multi_phase;
	VectorBatch *batch;
	TupleTableSlot *sort_slot;	/* minimal-tuple slot for spool reads */
	int			position;
	int			current_phase_idx;
	bool		spool_done;		/* current phase's input is exhausted */
	uint32		spool_gen;		/* batch generation already spooled */
	Tuplesortstate **phase_sorts;	/* per phase >= 2: next-phase sort input */
	Tuplestorestate **phase_stores;	/* plain phases need no ordering */
};

bool
VectorAggPackedInt4IsSupported(AggState *agg)
{
	int			transno;

	if (agg->aggstrategy != AGG_PLAIN || agg->numtrans == 0)
		return false;
	for (transno = 0; transno < agg->numtrans; transno++)
	{
		AggStatePerTrans pertrans = &agg->pertrans[transno];
		Oid			fnoid = pertrans->aggref->aggfnoid;

		if (pertrans->numSortCols != 0 || pertrans->aggref->aggfilter != NULL ||
			pertrans->aggref->aggdistinct)
			return false;
		if (fnoid != VEC_AGG_SUM_INT4 && fnoid != VEC_AGG_AVG_INT4 &&
			fnoid != VEC_AGG_MIN_INT4 && fnoid != VEC_AGG_MAX_INT4)
			return false;
		{
			ListCell   *lc;

			foreach(lc, pertrans->aggref->args)
				if (!VectorExprPackedInt4IsSupported(
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
		agg->aggstrategy != AGG_HASHED && agg->aggstrategy != AGG_SORTED &&
		agg->aggstrategy != AGG_MIXED)
		return false;

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
		/* A region above a Motion or another row leaf reads its output shape. */
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
		 agg->aggsplit != AGGSPLIT_INITIAL_SERIAL) ||
		!bms_is_empty(agg->aggParams) || agg->agg_expr_id != 0 || agg->streaming)
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
		if (!IsA(te->expr, Aggref))
			goto done;

		{
			Aggref	   *aggref = (Aggref *) te->expr;
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
				goto done;
			if (aggref->aggfilter != NULL &&
				!VectorExprIsSupported(aggref->aggfilter, &vars))
				goto done;
			foreach(arglc, aggref->args)
			{
				TargetEntry *arg = (TargetEntry *) lfirst(arglc);

				if (!VectorExprIsSupported(arg->expr, &vars))
				goto done;
			}
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
	state->packed_int4 = batch->packed_int4;
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
		case VEC_AGG_SUM_FLOAT8:
			if (pergroupstate->noTransValue)
				pergroupstate->transValue = pertrans->aggref->aggfnoid == VEC_AGG_SUM_FLOAT4 ?
					Float8GetDatum((float8) DatumGetFloat4(value)) : value;
			else
				pergroupstate->transValue = Float8GetDatum(
					DatumGetFloat8(pergroupstate->transValue) +
					(pertrans->aggref->aggfnoid == VEC_AGG_SUM_FLOAT4 ?
					 (float8) DatumGetFloat4(value) : DatumGetFloat8(value)));
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

static void
advance_packed_row(VectorAggState *state, AggStatePerGroup pergroup, int row)
{
	AggState   *agg = state->agg;
	int			transno;

	for (transno = 0; transno < agg->numtrans; transno++)
	{
		AggStatePerTrans pertrans = &agg->pertrans[transno];
		AggStatePerGroup pergroupstate = &pergroup[transno];
		int32		value;

		if ((state->arg_nulls[transno][0][row / 64] >> (row % 64)) & 1)
			continue;
		value = state->arg_i32[transno][0][row];
		switch (pertrans->aggref->aggfnoid)
		{
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
			case VEC_AGG_SUM_INT4:
				pergroupstate->transValue += value;
				pergroupstate->noTransValue = false;
				pergroupstate->transValueIsNull = false;
				break;
			case VEC_AGG_MAX_INT4:
				if (pergroupstate->noTransValue ||
					DatumGetInt32(pergroupstate->transValue) < value)
					pergroupstate->transValue = Int32GetDatum(value);
				pergroupstate->noTransValue = false;
				pergroupstate->transValueIsNull = false;
				break;
			case VEC_AGG_MIN_INT4:
				if (pergroupstate->noTransValue ||
					DatumGetInt32(pergroupstate->transValue) > value)
					pergroupstate->transValue = Int32GetDatum(value);
				pergroupstate->noTransValue = false;
				pergroupstate->transValueIsNull = false;
				break;
			default:
				pg_unreachable();
		}
	}
}

static void
advance_row(VectorAggState *state, AggStatePerGroup pergroup, int row)
{
	AggState   *agg = state->agg;
	int			transno;

	if (state->packed_int4)
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
	eval_arguments(state, batch);
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

		/* plain grouping sets go first, in phase order, as in nodeAgg */
		while (!state->mixed_hashed_started &&
			   state->mixed_plain_index < agg->maxsets)
		{
			int			k = state->mixed_plain_index++;

			/* each plain phase contributes exactly one grouping set */
			result = VectorAggRetrievePlainSet(agg, k + 1, 0, k);
			if (result != NULL)
				return result;
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
		return VectorAggRetrieveHashed(state->agg);

	state->output_done = true;
	return VectorAggRetrievePlain(state->agg);
}

void
VectorAggReset(VectorAggState *state)
{
	ExecReScanAgg(state->agg);
	/* the rescanned hash table drops the memoized group */
	state->group_valid = false;
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
}
