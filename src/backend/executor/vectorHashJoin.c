/*-------------------------------------------------------------------------
 *
 * vectorHashJoin.c
 *	  Feed batches of outer rows into the existing hash join implementation.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "executor/executor.h"
#include "executor/nodeHash.h"
#include "executor/vectorHashJoin.h"
#include "miscadmin.h"
#include "utils/datum.h"
#include "utils/faultinjector.h"

/* same state-machine constants as nodeHashjoin.c */
#define HJ_NEED_NEW_OUTER		2

/* Returns true if doing null-fill on inner relation */
#define HJ_FILL_INNER(hjstate)	((hjstate)->hj_NullOuterTupleSlot != NULL)

struct VectorHashJoinState
{
	MemoryContext context;
	MemoryContext outer_values;
	HashJoinState *hj;
	AttrNumber *outer_attnos;
	int			outer_ncols;
	bool		outer_dirty;	/* outer_values holds copies since last reset */
	bool		built;
	bool		outer_in_progress;
	VectorHashJoinSourceNext next_outer;
	void	   *outer_arg;
	VectorHashJoinState *outer_join;
	VectorBatch *outer_batch;
	int			outer_position;
	bool		outer_eof;
	bool		done;
	VectorBatch *output;
	VectorHashJoinSourceNext next_build;	/* batch feed for the hash build */
	void	   *build_arg;
};

bool
VectorHashJoinPlanIsSupported(HashJoin *join)
{
	if (!IsA(innerPlan(join), Hash))
		return false;
	/*
	 * The build side is consumed row-wise through the standard Hash node,
	 * so its input may be any plan; only the probe side feeds on batches
	 * and must be a simple scan.
	 */
	if (join->hashclauses == NIL ||
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
	Plan	   *outer = outerPlan(hj->js.ps.plan);
	ListCell   *lc;
	int			col = 0;

	state = MemoryContextAllocZero(parent, sizeof(VectorHashJoinState));
	state->context = AllocSetContextCreate(parent, "Vector Hash Join",
										 ALLOCSET_DEFAULT_MINSIZE,
										 ALLOCSET_DEFAULT_INITSIZE,
										 ALLOCSET_DEFAULT_MAXSIZE);
	state->outer_values = AllocSetContextCreate(state->context,
										 "Vector Hash Join Outer Values",
										 ALLOCSET_DEFAULT_MINSIZE,
										 ALLOCSET_DEFAULT_INITSIZE,
										 ALLOCSET_DEFAULT_MAXSIZE);
	state->hj = hj;
	state->outer_ncols = list_length(outer->targetlist);
	state->outer_attnos = MemoryContextAlloc(state->context,
										  sizeof(AttrNumber) * Max(state->outer_ncols, 1));
	foreach(lc, outer->targetlist)
	{
		TargetEntry *te = (TargetEntry *) lfirst(lc);
		Var		   *var;

		if (te->resjunk || !IsA(te->expr, Var))
		{
			VectorHashJoinDestroy(state);
			return NULL;
		}
		if (!IsA(outer, SeqScan))
		{
			state->outer_attnos[col] = col + 1;
			col++;
			continue;
		}
		var = (Var *) te->expr;
		if (var->varno != ((SeqScan *) outer)->scanrelid || var->varattno <= 0 ||
			var->varlevelsup != 0)
		{
			VectorHashJoinDestroy(state);
			return NULL;
		}
		state->outer_attnos[col++] = var->varattno;
	}
	return state;
}

VectorHashJoinState *
VectorHashJoinCreateBatch(MemoryContext parent, HashJoinState *hj,
						  VectorHashJoinSourceNext next_outer, void *outer_arg,
							  VectorHashJoinState *outer_join)
{
	VectorHashJoinState *state = VectorHashJoinCreate(parent, hj);

	if (state == NULL)
		return NULL;
	state->next_outer = next_outer;
	state->outer_arg = outer_arg;
	state->outer_join = outer_join;
	state->output = VectorBatchCreate(parent,
									  hj->js.ps.ps_ResultTupleDesc,
									  VECTOR_BATCH_DEFAULT_CAPACITY, false);
	return state;
}

VectorBatch *
VectorHashJoinOutput(VectorHashJoinState *state)
{
	return state->output;
}

void
VectorHashJoinSetBuildSource(VectorHashJoinState *state,
							 VectorHashJoinSourceNext next, void *arg)
{
	state->next_build = next;
	state->build_arg = arg;
}

static void
VectorHashJoinCopyResult(VectorHashJoinState *state, TupleTableSlot *slot, int row)
{
	int			col;

	slot_getallattrs(slot);
	if (slot->tts_nvalid != state->output->ncols)
		elog(ERROR, "hash join output shape changed during execution");
	for (col = 0; col < state->output->ncols; col++)
	{
		Datum		value = slot->tts_values[col];
		bool		isnull = slot->tts_isnull[col];

		value = VectorBatchCopyDatum(state->output, col, value, isnull);
		VectorBatchSetValue(state->output, col, row, value, isnull);
	}
}

bool
VectorHashJoinNextBatch(VectorHashJoinState *state)
{
	int			nrows;

	if (state->done)
		return false;
	do
	{
		nrows = 0;
		VectorBatchReset(state->output);
		while (nrows < state->output->capacity && !state->done)
		{
			TupleTableSlot *result = NULL;

			if (state->outer_in_progress)
				result = VectorHashJoinContinue(state);
			else if (state->outer_batch != NULL &&
					 state->outer_position < state->outer_batch->nselected)
				result = VectorHashJoinNextRow(state, state->outer_batch,
											  state->outer_position++);
			else if (state->outer_eof)
				result = VectorHashJoinFinish(state);
			else
			{
				state->outer_batch = NULL;
				state->outer_position = 0;
				if (state->next_outer(state->outer_arg, &state->outer_batch))
				{
					if (state->outer_batch->nselected == 0)
						continue;
					result = VectorHashJoinNextRow(state, state->outer_batch,
												  state->outer_position++);
				}
				else
				{
					state->outer_eof = true;
					continue;
				}
			}
			if (result == NULL)
			{
				if (state->outer_eof)
					state->done = true;
				else
					state->outer_in_progress = false;
				continue;
			}
			VectorHashJoinCopyResult(state, result, nrows++);
		}
		VectorBatchSetRowCount(state->output, nrows);
	} while (nrows == 0 && !state->done);
	/*
	 * A done chain with rows in the batch still reports true: the caller
	 * consumes the final partial batch before the next call reports EOF.
	 */
	return !state->done || nrows > 0;
}

static void
VectorHashJoinBatchReset(VectorHashJoinState *state)
{
	if (state == NULL)
		return;
	if (state->output != NULL)
		VectorBatchReset(state->output);
	state->outer_batch = NULL;
	state->outer_position = 0;
	state->outer_eof = false;
	state->done = false;
	state->built = state->hj->hj_HashTable != NULL;
	state->outer_in_progress = false;
	MemoryContextReset(state->outer_values);
	VectorHashJoinBatchReset(state->outer_join);
}

void
VectorHashJoinDestroy(VectorHashJoinState *state)
{
	if (state == NULL)
		return;
	if (state->outer_join != NULL)
		VectorHashJoinDestroy(state->outer_join);
	if (state->output != NULL)
		VectorBatchDestroy(state->output);
	MemoryContextDelete(state->context);
	pfree(state);
}

/*
 * Build the hash table from join output batches when the build side of the
 * join is itself a batch pipeline; otherwise MultiExecHash pulls rows the
 * usual way.  The per-row sequence mirrors MultiExecPrivateHash.
 */
static void
VectorHashBuildFromBatches(VectorHashJoinState *state, HashState *hashNode,
						   HashJoinTable hashtable)
{
	ExprContext *econtext = hashNode->ps.ps_ExprContext;
	List	   *hashkeys = hashNode->hashkeys;
	PlanState  *child = outerPlanState(hashNode);
	TupleTableSlot *slot;
	int			ncols;
	int			col;
	int			row;
	bool		quit = false;

	slot = MakeSingleTupleTableSlot(child->ps_ResultTupleDesc, &TTSOpsVirtual);
	ncols = slot->tts_tupleDescriptor->natts;

	while (!quit && VectorHashJoinNextBatch((VectorHashJoinState *) state->build_arg))
	{
		VectorBatch *batch = ((VectorHashJoinState *) state->build_arg)->output;

		for (row = 0; row < batch->nselected; row++)
		{
			bool		hashkeys_null = false;
			uint32		hashvalue;
			Datum		value;
			bool		isnull;

			CHECK_FOR_INTERRUPTS();

			ExecClearTuple(slot);
			for (col = 0; col < ncols; col++)
			{
				value = VectorBatchGetDatum(batch, col, row);
				isnull = VectorBatchIsNull(batch, col, row);
				slot->tts_values[col] = value;
				slot->tts_isnull[col] = isnull;
			}
			slot->tts_nvalid = ncols;
			ExecStoreVirtualTuple(slot);

			econtext->ecxt_outertuple = slot;
			if (ExecHashGetHashValue(hashNode, hashtable, econtext, hashkeys,
									 false, hashtable->keepNulls,
									 &hashvalue, &hashkeys_null))
			{
				int			bucketNumber = ExecHashGetSkewBucket(hashtable, hashvalue);

				if (bucketNumber != INVALID_SKEW_BUCKET_NO)
				{
					ExecHashSkewTableInsert(hashNode, hashtable, slot, hashvalue,
											bucketNumber);
					hashtable->skewTuples += 1;
				}
				else
					ExecHashTableInsert(hashNode, hashtable, slot, hashvalue);
				hashtable->totalTuples += 1;
			}

			if (hashkeys_null)
			{
				hashNode->hs_hashkeys_null = true;
				if (hashNode->hs_quit_if_hashkeys_null)
				{
					quit = true;
					break;
				}
			}
		}
	}

	ExecClearTuple(slot);
	ExecDropSingleTupleTableSlot(slot);

	/* the same finishing steps as MultiExecPrivateHash */
	hashtable->nbatch_outstart = hashtable->nbatch;
	if (hashtable->nbuckets != hashtable->nbuckets_optimal)
		ExecHashIncreaseNumBuckets(hashtable);
	hashtable->spaceUsed += hashtable->nbuckets * sizeof(HashJoinTuple);
	if (hashtable->spaceUsed > hashtable->spacePeak)
		hashtable->spacePeak = hashtable->spaceUsed;
	hashtable->partialTuples = hashtable->totalTuples;
}

void
VectorHashJoinBuild(VectorHashJoinState *state)
{
	HashJoinState *node = state->hj;
	HashState  *hashNode = castNode(HashState, innerPlanState(node));
	HashJoinTable hashtable;

	hashtable = ExecHashTableCreate(hashNode, node, node->hj_HashOperators,
									node->hj_Collations,
									HJ_FILL_INNER(node) || hashNode->hs_keepnull,
									PlanStateOperatorMemKB((PlanState *) hashNode));
	node->hj_HashTable = hashtable;
	hashNode->hashtable = hashtable;

	SIMPLE_FAULT_INJECTOR("multi_exec_hash_large_vmem");

	if (state->next_build != NULL)
		VectorHashBuildFromBatches(state, hashNode, hashtable);
	else
		MultiExecHash(hashNode);

	node->hj_HashTable = hashNode->hashtable;
	node->hj_HashTable->nbatch_outstart = node->hj_HashTable->nbatch;
	node->hj_InnerEmpty = node->hj_HashTable->totalTuples == 0;
	node->hj_OuterNotEmpty = false;
	node->hj_JoinState = HJ_NEED_NEW_OUTER;
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
		VectorHashJoinBuild(state);
		state->built = true;
	}

	if (state->outer_dirty)
	{
		MemoryContextReset(state->outer_values);
		state->outer_dirty = false;
	}
	ExecClearTuple(outer);
	for (col = 0; col < state->outer_ncols; col++)
	{
		AttrNumber	attno = state->outer_attnos[col];

		Datum		value = VectorBatchGetDatum(batch, attno - 1, row);
		bool		isnull = VectorBatchIsNull(batch, attno - 1, row);

		/*
		 * Result slots project by-ref outer values by reference and consumers
		 * may outlive the current batch, so the copy cannot be skipped (an
		 * attempt regressed q14 with invalid-memory errors).
		 */
		if (!isnull && !batch->columns[attno - 1].typbyval)
		{
			MemoryContext oldcontext = MemoryContextSwitchTo(state->outer_values);

			value = datumCopy(value, false, batch->columns[attno - 1].typlen);
			state->outer_dirty = true;
			MemoryContextSwitchTo(oldcontext);
		}
		outer->tts_values[col] = value;
		outer->tts_isnull[col] = isnull;
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
		VectorHashJoinBuild(state);
		state->built = true;
	}
	return VectorHashJoinProbeFinal(state->hj);
}

void
VectorHashJoinReset(VectorHashJoinState *state)
{
	ExecReScanHashJoin(state->hj);
	VectorHashJoinBatchReset(state);
	MemoryContextReset(state->outer_values);
	state->built = state->hj->hj_HashTable != NULL;
	state->outer_in_progress = false;
}
