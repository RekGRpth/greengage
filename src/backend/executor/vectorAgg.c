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

struct VectorAggState
{
	MemoryContext context;
	AggState   *agg;
	VectorExprVarMap vars;
	VectorExprState ***args;
	Datum	  **values;
	bool	  **nulls;
	TupleTableSlot *input_slot;	/* full input row, for spilling */
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
		agg->aggstrategy != AGG_HASHED)
		return false;

	vars->outer_ncols = list_length(outer->targetlist);
	vars->outer_attnos = MemoryContextAlloc(context,
										 sizeof(AttrNumber) * Max(vars->outer_ncols, 1));
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

	if ((agg->aggstrategy != AGG_PLAIN && agg->aggstrategy != AGG_HASHED) ||
		(agg->aggsplit != AGGSPLIT_SIMPLE &&
		 agg->aggsplit != AGGSPLIT_INITIAL_SERIAL) ||
		agg->groupingSets != NIL || agg->chain != NIL ||
		!bms_is_empty(agg->aggParams) || agg->agg_expr_id != 0 || agg->streaming ||
		agg->plan.qual != NIL || agg->plan.initPlan != NIL ||
		agg->plan.extParam != NULL || agg->plan.allParam != NULL)
		return false;
	if ((agg->aggstrategy == AGG_PLAIN && agg->numCols != 0) ||
		(agg->aggstrategy == AGG_HASHED && agg->numCols == 0))
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

			if (aggref->aggkind != AGGKIND_NORMAL || aggref->agglevelsup != 0 ||
				aggref->aggsplit != agg->aggsplit || aggref->aggvariadic ||
				aggref->aggorder != NIL || aggref->aggdistinct != NIL ||
				aggref->aggfilter != NULL || aggref->aggdirectargs != NIL ||
				aggref->aggtype == INTERNALOID)
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

	state->args = palloc0(sizeof(VectorExprState **) * agg->numtrans);
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
		Assert(argno == pertrans->numTransInputs);
	}
	if (agg->aggstrategy == AGG_PLAIN)
		VectorAggInitializePlain(agg);
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
		int			nargs = agg->pertrans[transno].numTransInputs;
		int			argno;

		for (argno = 0; argno < nargs; argno++)
			VectorExprEval(batch, state->args[transno][argno]);
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
		int			transno;

		if ((pos & 0xff) == 0)
			CHECK_FOR_INTERRUPTS();
		if (agg->aggstrategy == AGG_HASHED)
		{
			pergroup = lookup_group(state, batch, row);
			if (pergroup == NULL)
			{
				ResetExprContext(agg->tmpcontext);
				continue;
			}
		}
		else
			pergroup = agg->pergroups[0];

		for (transno = 0; transno < agg->numtrans; transno++)
		{
			int			nargs = agg->pertrans[transno].numTransInputs;
			int			argno;

			for (argno = 0; argno < nargs; argno++)
				state->values[transno][argno] =
					VectorExprGetValue(state->args[transno][argno], row,
									   &state->nulls[transno][argno]);
			VectorAggAdvanceTransition(agg, pergroup, transno,
									   state->values[transno], state->nulls[transno]);
		}
		ResetExprContext(agg->tmpcontext);
	}
}

void
VectorAggFinish(VectorAggState *state)
{
	if (state->input_done)
		return;
	VectorAggFinishInput(state->agg);
	state->input_done = true;
}

TupleTableSlot *
VectorAggNext(VectorAggState *state)
{
	TupleTableSlot *result;

	Assert(state->input_done);
	if (state->output_done)
		return NULL;

	if (state->agg->aggstrategy == AGG_HASHED)
	{
		result = VectorAggRetrieveHashed(state->agg);
		if (result == NULL)
			state->output_done = true;
		return result;
	}

	state->output_done = true;
	return VectorAggRetrievePlain(state->agg);
}

void
VectorAggReset(VectorAggState *state)
{
	ExecReScanAgg(state->agg);
	if (state->agg->aggstrategy == AGG_PLAIN)
		VectorAggInitializePlain(state->agg);
	/* A reused hash table must be iterated, not populated a second time. */
	state->input_done = state->agg->aggstrategy == AGG_HASHED &&
		state->agg->table_filled;
	state->output_done = false;
}
