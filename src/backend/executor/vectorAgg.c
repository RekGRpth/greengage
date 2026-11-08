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
#include "utils/float.h"
#include "utils/numeric.h"
#include "utils/tuplesort.h"

struct VectorAggState
{
	MemoryContext context;
	AggState   *agg;
	VectorExprVarMap vars;
	VectorExprState ***args;
	VectorExprState **filters;
	int32	   *filtered_selection;
	Datum	  **values;
	bool	  **nulls;
	TupleTableSlot *input_slot;	/* full input row, for spilling */
	int			projected_set;
	bool		group_open;
	bool		input_done;
	bool		output_done;
};

static bool
build_var_map(Agg *agg, MemoryContext context, VectorExprVarMap *vars)
{
	Plan	   *outer = outerPlan(agg);
	SeqScan	   *scan;
	ListCell   *lc;
	int			col = 0;

	if (!IsA(outer, SeqScan) && !IsA(outer, Motion) &&
		agg->aggstrategy != AGG_HASHED && agg->aggstrategy != AGG_SORTED)
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
		 agg->aggstrategy != AGG_SORTED) ||
		(agg->aggsplit != AGGSPLIT_SIMPLE &&
		 agg->aggsplit != AGGSPLIT_INITIAL_SERIAL) ||
		agg->chain != NIL ||
		!bms_is_empty(agg->aggParams) || agg->agg_expr_id != 0 || agg->streaming)
		return false;
	if ((agg->aggstrategy == AGG_PLAIN && agg->numCols != 0) ||
		(agg->aggstrategy != AGG_PLAIN && agg->numCols == 0))
		return false;
	if (!build_var_map(agg, CurrentMemoryContext, &vars))
		return false;

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
		if (IsA(te->expr, GroupingSetId))
			continue;
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
	oldcontext = MemoryContextSwitchTo(state->context);
	if (!build_var_map((Agg *) agg->ss.ps.plan, state->context, &state->vars))
	{
		MemoryContextSwitchTo(oldcontext);
		VectorAggDestroy(state);
		return NULL;
	}
	state->vars.estate = agg->ss.ps.state;

	state->args = palloc0(sizeof(VectorExprState **) * agg->numtrans);
	state->filters = palloc0(sizeof(VectorExprState *) * agg->numtrans);
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
	if (agg->aggstrategy == AGG_PLAIN)
	{
		/* nodeAgg owns its tuplesorts and frees them after the region state. */
		MemoryContextSwitchTo(agg->ss.ps.state->es_query_cxt);
		VectorAggInitializePlain(agg);
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
	if (state == NULL)
		return;
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
			VectorExprEvalSelection(batch, state->args[transno][argno],
									selection, nselected);
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

		slot->tts_values[col] = batch->columns[attno - 1].values[row];
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

	ExecClearTuple(hashslot);
	for (col = 0; col < perhash->numhashGrpCols; col++)
	{
		int			inputcol = perhash->hashGrpColIdxInput[col] - 1;
		AttrNumber	attno = state->vars.outer_attnos[inputcol];

		hashslot->tts_values[col] = batch->columns[attno - 1].values[row];
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
		VectorAggSpillTuple(agg, fill_input_slot(state, batch, row), hash);
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
	if (pertrans->aggref->aggfnoid == 2803)	/* count(*) */
	{
		pergroupstate->transValue++;
		return true;
	}
	if (pertrans->numTransInputs != 1)
		return false;

	value = state->values[transno][0];
	isnull = state->nulls[transno][0];
	if (pertrans->aggref->aggfnoid == 2802)	/* count(any) */
	{
		if (!isnull)
			pergroupstate->transValue++;
		return true;
	}
	if (isnull)
		return pertrans->transfn.fn_strict;

	switch (pertrans->aggref->aggfnoid)
	{
		case 2107:				/* sum(int8) */
#ifdef HAVE_INT128
		{
			Int128AggState *aggstate;
			MemoryContext oldcontext;

			if (pergroupstate->transValueIsNull)
			{
				oldcontext = MemoryContextSwitchTo(
					state->agg->curaggcontext->ecxt_per_tuple_memory);
				aggstate = makeInt128AggStateCurrentContext(false);
				MemoryContextSwitchTo(oldcontext);
				pergroupstate->transValue = PointerGetDatum(aggstate);
				pergroupstate->transValueIsNull = false;
				pergroupstate->noTransValue = false;
			}
			else
				aggstate = (Int128AggState *) DatumGetPointer(pergroupstate->transValue);
			aggstate->sumX += (int128) DatumGetInt64(value);
			aggstate->N++;
			return true;
		}
#else
			return false;
#endif
		case 2108:				/* sum(int4) */
			pergroupstate->transValue += DatumGetInt32(value);
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2109:				/* sum(int2) */
			pergroupstate->transValue += (int64) DatumGetInt16(value);
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2110:				/* sum(float4) */
		case 2111:				/* sum(float8) */
			if (pergroupstate->noTransValue)
				pergroupstate->transValue = pertrans->aggref->aggfnoid == 2110 ?
					Float8GetDatum((float8) DatumGetFloat4(value)) : value;
			else
				pergroupstate->transValue = Float8GetDatum(
					DatumGetFloat8(pergroupstate->transValue) +
					(pertrans->aggref->aggfnoid == 2110 ?
					 (float8) DatumGetFloat4(value) : DatumGetFloat8(value)));
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2115:				/* max(int8) */
			if (pergroupstate->noTransValue ||
				DatumGetInt64(pergroupstate->transValue) < DatumGetInt64(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2116:				/* max(int4) */
			if (pergroupstate->noTransValue ||
				DatumGetInt32(pergroupstate->transValue) < DatumGetInt32(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2117:				/* max(int2) */
			if (pergroupstate->noTransValue ||
				DatumGetInt16(pergroupstate->transValue) < DatumGetInt16(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2119:				/* max(float4) */
			if (pergroupstate->noTransValue ||
				float4_cmp_internal(DatumGetFloat4(pergroupstate->transValue),
									DatumGetFloat4(value)) < 0)
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2120:				/* max(float8) */
			if (pergroupstate->noTransValue ||
				float8_cmp_internal(DatumGetFloat8(pergroupstate->transValue),
									DatumGetFloat8(value)) < 0)
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2122:				/* max(date) */
			if (pergroupstate->noTransValue ||
				DatumGetInt32(pergroupstate->transValue) < DatumGetInt32(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2126:				/* max(timestamp) */
		case 2127:				/* max(timestamptz) */
			if (pergroupstate->noTransValue ||
				DatumGetInt64(pergroupstate->transValue) < DatumGetInt64(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2131:				/* min(int8) */
			if (pergroupstate->noTransValue ||
				DatumGetInt64(pergroupstate->transValue) > DatumGetInt64(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2132:				/* min(int4) */
			if (pergroupstate->noTransValue ||
				DatumGetInt32(pergroupstate->transValue) > DatumGetInt32(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2133:				/* min(int2) */
			if (pergroupstate->noTransValue ||
				DatumGetInt16(pergroupstate->transValue) > DatumGetInt16(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2135:				/* min(float4) */
			if (pergroupstate->noTransValue ||
				float4_cmp_internal(DatumGetFloat4(pergroupstate->transValue),
									DatumGetFloat4(value)) > 0)
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2136:				/* min(float8) */
			if (pergroupstate->noTransValue ||
				float8_cmp_internal(DatumGetFloat8(pergroupstate->transValue),
									DatumGetFloat8(value)) > 0)
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2138:				/* min(date) */
			if (pergroupstate->noTransValue ||
				DatumGetInt32(pergroupstate->transValue) > DatumGetInt32(value))
				pergroupstate->transValue = value;
			pergroupstate->noTransValue = false;
			pergroupstate->transValueIsNull = false;
			return true;
		case 2142:				/* min(timestamp) */
		case 2143:				/* min(timestamptz) */
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
advance_row(VectorAggState *state, AggStatePerGroup pergroup, int row)
{
	AggState   *agg = state->agg;
	int			transno;

	for (transno = 0; transno < agg->numtrans; transno++)
	{
		AggStatePerTrans pertrans = &agg->pertrans[transno];
		int			nargs = pertrans->numInputs;
		int			argno;

		if (state->filters[transno] != NULL)
		{
			bool		isnull;
			Datum		value = VectorExprGetValue(state->filters[transno], row,
												 &isnull);

			if (isnull || !DatumGetBool(value))
				continue;
		}
		for (argno = 0; argno < nargs; argno++)
			state->values[transno][argno] =
				VectorExprGetValue(state->args[transno][argno], row,
								   &state->nulls[transno][argno]);
		if (pertrans->numSortCols == 0 &&
			advance_byval_fast(state, pergroup, transno, row))
			continue;
		if (pertrans->numSortCols > 0)
		{
			Assert(agg->aggstrategy == AGG_PLAIN || agg->aggstrategy == AGG_SORTED);
			if (nargs == 1)
				tuplesort_putdatum(pertrans->sortstates[0],
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
				tuplesort_puttupleslot(pertrans->sortstates[0], sortslot);
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

	if (state->agg->aggstrategy == AGG_SORTED)
	{
		int			numsets = Max(state->agg->phase->numsets, 1);

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
			if (state->projected_set < 0 && state->agg->phase->aggnode->numCols > 0 &&
				(state->agg->phase->gset_lengths == NULL ||
				 state->agg->phase->gset_lengths[nextset] > 0))
			{
				state->projected_set = nextset;
				continue;
			}
			state->projected_set = nextset;
			initialize_empty_sorted_set(state);
			result = VectorAggRetrieveGroup(state->agg, state->projected_set);
			if (result != NULL)
				return result;
		}
		state->output_done = true;
		return NULL;
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
	if (state->agg->aggstrategy == AGG_PLAIN)
	{
		MemoryContext oldcontext = MemoryContextSwitchTo(
			state->agg->ss.ps.state->es_query_cxt);

		VectorAggInitializePlain(state->agg);
		MemoryContextSwitchTo(oldcontext);
	}
	/* A reused hash table must be iterated, not populated a second time. */
	state->input_done = state->agg->aggstrategy == AGG_HASHED &&
		state->agg->table_filled;
	state->group_open = false;
	state->projected_set = -1;
	state->output_done = false;
}
