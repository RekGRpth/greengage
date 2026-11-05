/*-------------------------------------------------------------------------
 *
 * vectorHashJoin.c
 *	  Feed batches of outer rows into the existing hash join implementation.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "executor/vectorHashJoin.h"

struct VectorHashJoinState
{
	MemoryContext context;
	HashJoinState *hj;
	AttrNumber *outer_attnos;
	int			outer_ncols;
	bool		built;
	bool		outer_in_progress;
};

bool
VectorHashJoinPlanIsSupported(HashJoin *join)
{
	Hash	   *hash;

	if (!IsA(innerPlan(join), Hash))
		return false;
	hash = (Hash *) innerPlan(join);
	if (!IsA(outerPlan(hash), SeqScan) || join->hashclauses == NIL ||
		join->hashqualclauses != NIL || join->hashkeys == NIL ||
		join->join.plan.initPlan != NIL || join->join.plan.extParam != NULL ||
		join->join.plan.allParam != NULL)
		return false;

	switch (join->join.jointype)
	{
		case JOIN_INNER:
		case JOIN_LEFT:
		case JOIN_RIGHT:
		case JOIN_FULL:
		case JOIN_SEMI:
		case JOIN_ANTI:
			return true;
		default:
			return false;
	}
}

VectorHashJoinState *
VectorHashJoinCreate(MemoryContext parent, HashJoinState *hj)
{
	VectorHashJoinState *state;
	SeqScan	   *outer = (SeqScan *) outerPlan(hj->js.ps.plan);
	ListCell   *lc;
	int			col = 0;

	state = MemoryContextAllocZero(parent, sizeof(VectorHashJoinState));
	state->context = AllocSetContextCreate(parent, "Vector Hash Join",
										 ALLOCSET_DEFAULT_MINSIZE,
										 ALLOCSET_DEFAULT_INITSIZE,
										 ALLOCSET_DEFAULT_MAXSIZE);
	state->hj = hj;
	state->outer_ncols = list_length(outer->plan.targetlist);
	state->outer_attnos = MemoryContextAlloc(state->context,
										  sizeof(AttrNumber) * Max(state->outer_ncols, 1));
	foreach(lc, outer->plan.targetlist)
	{
		TargetEntry *te = (TargetEntry *) lfirst(lc);
		Var		   *var;

		if (te->resjunk || !IsA(te->expr, Var))
		{
			VectorHashJoinDestroy(state);
			return NULL;
		}
		var = (Var *) te->expr;
		if (var->varno != outer->scanrelid || var->varattno <= 0 ||
			var->varlevelsup != 0)
		{
			VectorHashJoinDestroy(state);
			return NULL;
		}
		state->outer_attnos[col++] = var->varattno;
	}
	return state;
}

void
VectorHashJoinDestroy(VectorHashJoinState *state)
{
	if (state == NULL)
		return;
	MemoryContextDelete(state->context);
	pfree(state);
}

TupleTableSlot *
VectorHashJoinNextRow(VectorHashJoinState *state, VectorBatch *batch, int position)
{
	HashJoinState *hj = state->hj;
	TupleTableSlot *outer = hj->hj_OuterTupleSlot;
	int			row = VectorBatchSelectedRow(batch, position);
	int			col;

	if (!state->built)
	{
		VectorHashJoinBuild(hj);
		state->built = true;
	}

	ExecClearTuple(outer);
	for (col = 0; col < state->outer_ncols; col++)
	{
		AttrNumber	attno = state->outer_attnos[col];

		outer->tts_values[col] = batch->columns[attno - 1].values[row];
		outer->tts_isnull[col] = VectorBatchIsNull(batch, attno - 1, row);
	}
	outer->tts_nvalid = state->outer_ncols;
	ExecStoreVirtualTuple(outer);
	state->outer_in_progress = true;
	{
		TupleTableSlot *result = VectorHashJoinProbe(hj, outer);

		if (VectorHashJoinNeedsNewOuter(hj))
			state->outer_in_progress = false;
		return result;
	}
}

bool
VectorHashJoinInProgress(VectorHashJoinState *state)
{
	return state->outer_in_progress;
}

TupleTableSlot *
VectorHashJoinContinue(VectorHashJoinState *state)
{
	TupleTableSlot *result = VectorHashJoinProbe(state->hj, NULL);

	if (VectorHashJoinNeedsNewOuter(state->hj))
		state->outer_in_progress = false;
	return result;
}

TupleTableSlot *
VectorHashJoinFinish(VectorHashJoinState *state)
{
	if (!state->built)
	{
		VectorHashJoinBuild(state->hj);
		state->built = true;
	}
	return VectorHashJoinProbeFinal(state->hj);
}

void
VectorHashJoinReset(VectorHashJoinState *state)
{
	ExecReScanHashJoin(state->hj);
	state->built = state->hj->hj_HashTable != NULL;
	state->outer_in_progress = false;
}
