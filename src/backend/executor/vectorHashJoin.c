/*-------------------------------------------------------------------------
 *
 * vectorHashJoin.c
 *	  Feed batches of outer rows into the existing hash join implementation.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "access/htup_details.h"
#include "executor/executor.h"
#include "executor/hashjoin.h"
#include "executor/nodeHash.h"
#include "executor/vectorHashJoin.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "nodes/nodeFuncs.h"
#include "utils/datum.h"
#include "utils/faultinjector.h"
#include "utils/lsyscache.h"

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
	/*
	 * Batch probe fast path: assemble matched rows straight into the output
	 * batch without the per-row HashJoin state machine.  The outer slot is
	 * only filled with the key columns; output columns are read from the
	 * probe batch and the stored inner tuples directly.
	 */
	bool		fast_probe;		/* plan gate passed */
	bool		fast_checked;	/* post-build gate evaluated */
	int			hash_slot_nvalid;
	TupleTableSlot *hash_slot;
	int			nkeys;
	int		   *outer_key_pos;	/* outer targetlist positions of the keys */
	int		   *inner_key_attno;	/* stored-tuple attnos of the inner keys */
	FmgrInfo   *eq_fns;
	Oid		   *eq_collations;
	int			out_ncols;
	bool	   *out_is_outer;
	int		   *out_src;		/* outer targetlist pos / inner tuple attno */
	int		   *out_batch_col;	/* probe-batch column for outer outputs */
	bool		resume_active;	/* capacity stop inside a bucket walk */
	int			resume_bucket;
	HashJoinTuple resume_cursor;
	uint32		resume_hashvalue;
	int			resume_row;
	bool		fill_outer;		/* LEFT: emit null-filled unmatched rows */
	bool		fill_inner;		/* RIGHT: mark matches for the finish phase */
	bool		row_matched;	/* current probe row already emitted a match */
	bool		semi_join;		/* SEMI: emit the outer row on the first match */
	bool		anti_join;		/* ANTI/LASJ: a matched row is never emitted */
	bool		notin_join;		/* LASJ NOTIN: null probe keys never match */
	bool		hash_direct;	/* single key: hash straight from the batch */
	Datum		probe_key;		/* current probe key (survives a resume) */
	/*
	 * Batched fill_inner finish: a cursor over the hash table's buckets that
	 * assembles unmatched stored tuples directly into output batches.
	 */
	bool		finish_prepped; /* cursor initialized */
	bool		finish_done;	/* all buckets scanned */
	int			finish_bucket;	/* next regular bucket */
	int			finish_skew;	/* next skew bucket index */
	HashJoinTuple finish_tuple; /* last unmatched tuple returned */
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
		case JOIN_LASJ_NOTIN:
			return true;
		default:
			return false;
	}
}

static bool VectorHashProbeFastGate(VectorHashJoinState *state);
static int VectorHashFinishBatch(VectorHashJoinState *state, int start_row);
static int VectorHashProbeBatchFast(VectorHashJoinState *state, int start_row);

static void
VectorHashJoinInitFastProbe(VectorHashJoinState *state)
{
	HashJoinState *hj = state->hj;
	int			nclauses = list_length(((HashJoin *) hj->js.ps.plan)->hashclauses);
	int			natts = hj->js.ps.ps_ResultTupleDesc->natts;

	state->hash_slot = MakeSingleTupleTableSlot(
		ExecGetResultType(outerPlanState(hj)), &TTSOpsVirtual);
	state->outer_key_pos = MemoryContextAlloc(state->context,
									   sizeof(int) * Max(nclauses, 1));
	state->inner_key_attno = MemoryContextAlloc(state->context,
									   sizeof(int) * Max(nclauses, 1));
	state->eq_fns = MemoryContextAlloc(state->context,
									   sizeof(FmgrInfo) * Max(nclauses, 1));
	state->eq_collations = MemoryContextAlloc(state->context,
									   sizeof(Oid) * Max(nclauses, 1));
	state->out_is_outer = MemoryContextAlloc(state->context,
									   sizeof(bool) * Max(natts, 1));
	state->out_src = MemoryContextAlloc(state->context,
									   sizeof(int) * Max(natts, 1));
	state->out_batch_col = MemoryContextAlloc(state->context,
									   sizeof(int) * Max(natts, 1));
	state->fast_probe = VectorHashProbeFastGate(state);
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
	VectorHashJoinInitFastProbe(state);
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

/*
 * Plan-time gate for the batch probe fast path.  Only the plain shape:
 * an inner join with plain equality hash clauses and a plain positional
 * targetlist over outer and inner columns, no extra quals.
 */
static bool
VectorHashProbeFastGate(VectorHashJoinState *state)
{
	HashJoin   *join = (HashJoin *) state->hj->js.ps.plan;
	ListCell   *lc;
	ListCell   *collc;
	int			nkeys = 0;
	int			out_ncols = 0;
	int			maxkey = 0;

	/*
	 * INNER is the plain case.  LEFT fills unmatched probe rows with null
	 * inner columns; RIGHT marks matched stored tuples and lets the normal
	 * finish phase emit the unmatched inner rows.  SEMI emits the outer row
	 * on the first match; ANTI and LASJ NOTIN never emit a matched row and
	 * reuse the LEFT-style fill for unmatched ones.  Residual quals are
	 * evaluated per key-equal candidate.  Anything else stays on the per-row
	 * path.
	 */
	if (join->join.jointype != JOIN_INNER &&
		join->join.jointype != JOIN_LEFT &&
		join->join.jointype != JOIN_RIGHT &&
		join->join.jointype != JOIN_SEMI &&
		join->join.jointype != JOIN_ANTI &&
		join->join.jointype != JOIN_LASJ_NOTIN)
		return false;

	forboth(lc, join->hashclauses, collc, join->hashcollations)
	{
		OpExpr	   *op = lfirst_node(OpExpr, lc);
		Var		   *ovar = NULL;
		Var		   *ivar = NULL;
		ListCell   *arglc;
		if (list_length(op->args) != 2)
			return false;
		foreach(arglc, op->args)
		{
			Node	   *arg = lfirst(arglc);
			Var		   *v;

			if (!IsA(arg, Var))
				return false;
			v = (Var *) arg;
			if (v->varno == OUTER_VAR)
				ovar = v;
			else if (v->varno == INNER_VAR)
				ivar = v;
			else
				return false;
		}
		if (ovar == NULL || ivar == NULL ||
			ovar->varattno <= 0 || ivar->varattno <= 0)
			return false;
		state->outer_key_pos[nkeys] = ovar->varattno;
		state->inner_key_attno[nkeys] = ivar->varattno;
		fmgr_info(get_opcode(op->opno), &state->eq_fns[nkeys]);
		/*
		 * The plan's hashcollations, like the standard executor's
		 * hj_Collations, not the clause's inputcollid: ORCA leaves the
		 * latter invalid on hash clauses, and the key equality would
		 * fail with "could not determine which collation".
		 */
		state->eq_collations[nkeys] = lfirst_oid(collc);
		nkeys++;
		maxkey = Max(maxkey, ovar->varattno);
	}
	if (nkeys == 0)
		return false;
	state->fill_outer = (join->join.jointype == JOIN_LEFT ||
						 join->join.jointype == JOIN_ANTI ||
						 join->join.jointype == JOIN_LASJ_NOTIN);
	state->fill_inner = (join->join.jointype == JOIN_RIGHT);
	state->semi_join = (join->join.jointype == JOIN_SEMI);
	state->anti_join = (join->join.jointype == JOIN_ANTI ||
						join->join.jointype == JOIN_LASJ_NOTIN);
	state->notin_join = (join->join.jointype == JOIN_LASJ_NOTIN);

	foreach(lc, join->join.plan.targetlist)
	{
		TargetEntry *te = lfirst_node(TargetEntry, lc);
		Var		   *v = (Var *) te->expr;

		if (te->resjunk)
			continue;
		if (!IsA(te->expr, Var))
			return false;
		if (v->varno == OUTER_VAR)
		{
			state->out_is_outer[out_ncols] = true;
			state->out_src[out_ncols] = v->varattno;
			state->out_batch_col[out_ncols] =
				state->outer_attnos[v->varattno - 1] - 1;
		}
		else if (v->varno == INNER_VAR)
		{
			state->out_is_outer[out_ncols] = false;
			state->out_src[out_ncols] = v->varattno;
		}
		else
			return false;
		out_ncols++;
	}
	if (out_ncols != state->hj->js.ps.ps_ResultTupleDesc->natts ||
		state->hash_slot->tts_tupleDescriptor->natts < maxkey)
		return false;
	state->nkeys = nkeys;
	/*
	 * A single hash clause hashes straight from the batch column: with one
	 * key the combine is just hashfn(key), so neither the slot nor the
	 * expression interpreter is needed.
	 */
	state->hash_direct = (nkeys == 1);
	state->out_ncols = out_ncols;
	state->hash_slot_nvalid = maxkey;
	return true;
}

/*
 * Batch probe: consume probe rows from state->outer_batch starting at
 * state->outer_position, assemble matched rows directly into the output
 * batch, and return the number of output rows produced.  May stop early on
 * output capacity, leaving a resume cursor inside the current bucket walk.
 */
static int
VectorHashProbeBatchFast(VectorHashJoinState *state, int start_row)
{
	HashJoinState *hj = state->hj;
	HashJoinTable hashtable = hj->hj_HashTable;
	HashState  *hashNode = castNode(HashState, innerPlanState(hj));
	VectorBatch *batch = state->outer_batch;
	VectorBatch *out = state->output;
	ExprContext *econtext = hj->js.ps.ps_ExprContext;
	int			nrows = start_row;
	int			row = 0;
	int			k;

	while (nrows < out->capacity)
	{
		uint32		hashvalue;
		bool		hashkeys_null = false;
		int			bucketno;
		int			batchno;
		HashJoinTuple cursor;
		bool		resume_stopped = false;

		/* resume an unfinished bucket walk for the current probe row */
		if (!state->resume_active)
		{
			if (state->outer_position >= batch->nselected)
				break;
			state->row_matched = false;
			row = VectorBatchSelectedRow(batch, state->outer_position);
			state->outer_position++;

			if (state->hash_direct)
			{
				/*
				 * Single key: with one clause the combine is hashfn(key),
				 * so hash straight from the batch column.  Null semantics
				 * follow ExecHashGetHashValue (strict keys reject the row
				 * unless the table keeps nulls).
				 */
				int			bcol = state->outer_attnos[state->outer_key_pos[0] - 1] - 1;
				Datum		keyval = VectorBatchGetDatum(batch, bcol, row);

				state->probe_key = keyval;
				if (VectorBatchIsNull(batch, bcol, row))
				{
					if (hashtable->hashStrict[0] && !hashtable->keepNulls)
						continue;	/* strict key: never matches */
					if (state->notin_join)
					{
						/*
						 * OPT-3325: a null probe key drops the row when the
						 * table is not empty; an empty table falls through
						 * to the unmatched-row fill below.
						 */
						state->row_matched = (hashtable->totalTuples > 0);
						continue;
					}
					hashvalue = 0;	/* null hashes as zero, like the interp */
				}
				else
					hashvalue = DatumGetUInt32(FunctionCall1Coll(
								 &hashtable->outer_hashfunctions[0],
								 hashtable->collations[0], keyval));
			}
			else
			{
				/* fill the hash slot with key columns only */
				ExecClearTuple(state->hash_slot);
				for (k = 0; k < state->nkeys; k++)
				{
					int			bcol = state->outer_attnos[state->outer_key_pos[k] - 1] - 1;

					state->hash_slot->tts_values[state->outer_key_pos[k] - 1] =
						VectorBatchGetDatum(batch, bcol, row);
					state->hash_slot->tts_isnull[state->outer_key_pos[k] - 1] =
						VectorBatchIsNull(batch, bcol, row);
				}
				state->hash_slot->tts_nvalid = state->hash_slot_nvalid;
				ExecStoreVirtualTuple(state->hash_slot);

				if (state->notin_join)
				{
					/* OPT-3325, same drop rule as the direct path */
					bool		keynull = false;

					for (k = 0; k < state->nkeys; k++)
					{
						if (state->hash_slot->tts_isnull[state->outer_key_pos[k] - 1])
						{
							keynull = true;
							break;
						}
					}
					if (keynull)
					{
						state->row_matched = (hashtable->totalTuples > 0);
						continue;
					}
				}

				econtext->ecxt_outertuple = state->hash_slot;
				if (!ExecHashGetHashValue(hashNode, hashtable, econtext,
										  hj->hj_OuterHashKeys, true,
										  hashtable->keepNulls,
										  &hashvalue, &hashkeys_null))
					continue;	/* null keys never match */
			}
			ExecHashGetBucketAndBatch(hashtable, hashvalue, &bucketno, &batchno);
			if (batchno != 0)
			{
				/* spilled to another batch: fall back to the row path */
				state->fast_probe = false;
				break;
			}
			{
				int			skewno = ExecHashGetSkewBucket(hashtable, hashvalue);

				if (skewno == INVALID_SKEW_BUCKET_NO)
					cursor = hashtable->buckets.unshared[bucketno];
				else
					cursor = hashtable->skewBucket[skewno]->tuples;
			}
		}
		else
		{
			row = state->resume_row;
			hashvalue = state->resume_hashvalue;
			bucketno = state->resume_bucket;
			cursor = state->resume_cursor;
		}

		for (; cursor != NULL; cursor = cursor->next.unshared)
		{
			bool		matched = true;
			int			c;
			TupleTableSlot *hslot = hj->hj_HashTupleSlot;

			if (cursor->hashvalue != hashvalue)
				continue;

			if (nrows >= out->capacity)
			{
				state->resume_active = true;
				state->resume_bucket = bucketno;
				state->resume_cursor = cursor;
				state->resume_hashvalue = hashvalue;
				state->resume_row = row;
				resume_stopped = true;
				break;
			}

			ExecStoreMinimalTuple(HJTUPLE_MINTUPLE(cursor), hslot, false);
			for (k = 0; k < state->nkeys; k++)
			{
				bool		inner_null;
				Datum		ival = slot_getattr(hslot,
											   state->inner_key_attno[k],
											   &inner_null);
				Datum		oval = state->hash_direct
					? state->probe_key
					: state->hash_slot->tts_values[state->outer_key_pos[k] - 1];

				if (inner_null ||
					!DatumGetBool(FunctionCall2Coll(&state->eq_fns[k],
													state->eq_collations[k],
													oval, ival)))
				{
					matched = false;
					break;
				}
			}
			/*
			 * The hashed key equality only locates candidates.  Like the
			 * standard probe, the joinqual decides the match; the other qual
			 * (ps.qual) only gates the emission and never un-matches a row.
			 * The outer slot is filled from the batch without copies: it
			 * only lives for the evaluations below.
			 */
			if (matched && (hj->js.joinqual != NULL || hj->js.ps.qual != NULL))
			{
				TupleTableSlot *oslot = hj->hj_OuterTupleSlot;

				ExecClearTuple(oslot);
				for (k = 0; k < state->outer_ncols; k++)
				{
					AttrNumber	attno = state->outer_attnos[k];

					oslot->tts_values[k] = VectorBatchGetDatum(batch, attno - 1, row);
					oslot->tts_isnull[k] = VectorBatchIsNull(batch, attno - 1, row);
				}
				oslot->tts_nvalid = state->outer_ncols;
				ExecStoreVirtualTuple(oslot);
				slot_getallattrs(hslot);

				ResetExprContext(econtext);
				econtext->ecxt_outertuple = oslot;
				econtext->ecxt_innertuple = hslot;
				if (hj->js.joinqual != NULL &&
					!ExecQual(hj->js.joinqual, econtext))
					matched = false;
				else if (hj->js.ps.qual != NULL &&
						 !ExecQual(hj->js.ps.qual, econtext))
				{
					/*
					 * Matched but filtered from the output.  A single-match
					 * (SEMI) row is done; an INNER/LEFT candidate keeps the
					 * bucket walk going, like the standard probe.
					 */
					state->row_matched = true;
					if (state->semi_join)
						break;
					continue;
				}
			}
			if (!matched)
				continue;
			/* the finish phase emits unmatched stored tuples only */
			if (state->fill_inner)
				HeapTupleHeaderSetMatch(HJTUPLE_MINTUPLE(cursor));

			state->row_matched = true;
			if (state->anti_join)
				break;		/* ANTI/LASJ: a matched row is never emitted */
			for (c = 0; c < state->out_ncols; c++)
			{
				Datum		value;
				bool		isnull;

				if (state->out_is_outer[c])
				{
					value = VectorBatchGetDatum(batch,
										   state->out_batch_col[c], row);
					isnull = VectorBatchIsNull(batch,
											   state->out_batch_col[c], row);
				}
				else
				{
					value = slot_getattr(hslot, state->out_src[c], &isnull);
				}
				value = VectorBatchCopyDatum(out, c, value, isnull);
				VectorBatchSetValue(out, c, nrows, value, isnull);
			}
			nrows++;
			if (state->semi_join)
				break;		/* SEMI: only the first match is emitted */
		}

		if (resume_stopped)
			break;

		/*
		 * LEFT join: a probe row with no matches emits a null-filled row.  If
		 * the output batch is full, resume with an empty walk so the fill
		 * row is emitted first in the next call.
		 */
		if (state->fill_outer && !state->row_matched)
		{
			int			c;

			if (nrows >= out->capacity)
			{
				state->resume_active = true;
				state->resume_bucket = bucketno;
				state->resume_cursor = NULL;
				state->resume_hashvalue = hashvalue;
				state->resume_row = row;
				resume_stopped = true;
				break;
			}
			for (c = 0; c < state->out_ncols; c++)
			{
				Datum		value = 0;
				bool		isnull = true;

				if (state->out_is_outer[c])
				{
					value = VectorBatchGetDatum(batch,
											   state->out_batch_col[c], row);
					isnull = VectorBatchIsNull(batch,
											   state->out_batch_col[c], row);
				}
				value = VectorBatchCopyDatum(out, c, value, isnull);
				VectorBatchSetValue(out, c, nrows, value, isnull);
			}
			nrows++;
		}
		state->resume_active = false;
	}

	return nrows;
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

			if (state->fast_probe && !state->outer_in_progress &&
				!state->outer_eof && !state->done)
			{
				HashJoinTable hashtable = state->hj->hj_HashTable;

				if (!state->fast_checked)
				{
					state->fast_checked = true;
					if (hashtable == NULL)
					{
						VectorHashJoinBuild(state);
						state->built = true;
						hashtable = state->hj->hj_HashTable;
					}
					/*
					 * LASJ NOTIN with a null inner key: the whole join result
					 * is empty (the build stopped at the null), like the
					 * standard executor.
					 */
					if (state->notin_join &&
						castNode(HashState, innerPlanState(state->hj))->hs_hashkeys_null)
					{
						state->done = true;
						break;
					}
					if (hashtable->nbatch != 1 ||
						hashtable->parallel_state != NULL ||
						hashtable->eagerlyReleased)
						state->fast_probe = false;
				}
				if (state->fast_probe)
				{
					if (state->outer_batch == NULL ||
						(state->outer_position >= state->outer_batch->nselected &&
						 !state->resume_active))
					{
						state->outer_batch = NULL;
						state->outer_position = 0;
						if (!state->next_outer(state->outer_arg,
											   &state->outer_batch))
						{
							/*
							 * A source hands back its reusable batch object
							 * even at EOF: drop the pointer, or the machine
							 * path would re-probe the already-consumed last
							 * batch and end the chain before the fill_inner
							 * finish phase runs.
							 */
							state->outer_batch = NULL;
							state->outer_position = 0;
							state->outer_eof = true;
							/* RIGHT hands unmatched stored tuples to the
							 * finish phase; INNER/LEFT have nothing left */
							state->done = !state->fill_inner;
							continue;
						}
					}
					nrows = VectorHashProbeBatchFast(state, nrows);
					continue;
				}
			}

			if (state->outer_in_progress)
				result = VectorHashJoinContinue(state);
			else if (state->outer_batch != NULL &&
					 state->outer_position < state->outer_batch->nselected)
				result = VectorHashJoinNextRow(state, state->outer_batch,
											  state->outer_position++);
			else if (state->outer_eof)
			{
				/*
				 * RIGHT join over a single-batch table: assemble the
				 * unmatched stored tuples directly into output batches
				 * instead of projecting them row by row through the
				 * machine's finish phase.
				 */
				if (state->fill_inner && state->fast_probe &&
					state->hj->hj_HashTable != NULL)
				{
					nrows = VectorHashFinishBatch(state, nrows);
					if (state->finish_done)
						state->done = true;
					continue;
				}
				result = VectorHashJoinFinish(state);
			}
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
					/* drop the stale batch the source handed back */
					state->outer_batch = NULL;
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
	state->fast_checked = false;
	state->resume_active = false;
	state->finish_prepped = false;
	state->finish_done = false;
	state->finish_bucket = 0;
	state->finish_skew = 0;
	state->finish_tuple = NULL;
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
	if (state->hash_slot != NULL)
		ExecDropSingleTupleTableSlot(state->hash_slot);
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

	/* CDB: the spill-reload path expects the stats workarea under EXPLAIN */
	if ((node->js.ps.state->es_instrument & INSTRUMENT_CDB) != 0)
		ExecHashTableExplainInit(hashNode, node, hashtable);

	/*
	 * LASJ NOTIN: stop the build at the first null inner key; the join's
	 * result is empty in that case (same rule as the standard executor).
	 */
	hashNode->hs_quit_if_hashkeys_null =
		(node->js.jointype == JOIN_LASJ_NOTIN);

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

/*
 * Batched fill_inner finish: walk the hash table's buckets (single batch,
 * the only shape the fast probe gate approves), assembling each unmatched
 * stored tuple directly into the output batch - outer columns null-filled,
 * inner columns read positionally.  The cursor persists across calls, so
 * the walk resumes where the previous output batch stopped.  No quals run
 * here: the gate only reaches this shape with a plain targetlist and no
 * quals, which is exactly what the machine's ExecProject would project.
 */
static int
VectorHashFinishBatch(VectorHashJoinState *state, int start_row)
{
	HashJoinState *hj = state->hj;
	HashJoinTable hashtable = hj->hj_HashTable;
	VectorBatch *out = state->output;
	TupleTableSlot *hslot = hj->hj_HashTupleSlot;
	int			nrows = start_row;
	int			c;

	if (!state->finish_prepped)
	{
		state->finish_prepped = true;
		state->finish_bucket = 0;
		state->finish_skew = 0;
		state->finish_tuple = NULL;
		state->finish_done = false;
	}
	while (nrows < out->capacity && !state->finish_done)
	{
		HashJoinTuple tuple = state->finish_tuple;

		/*
		 * Advance the cursor: resuming an emitted tuple walks its chain;
		 * NULL (emitted tuple's chain or a fetched bucket exhausted) makes
		 * the next call fetch the following bucket, whose index was already
		 * advanced below.
		 */
		if (tuple != NULL)
			tuple = tuple->next.unshared;
		else if (state->finish_bucket < hashtable->nbuckets)
		{
			tuple = hashtable->buckets.unshared[state->finish_bucket];
			state->finish_bucket++;
		}
		else if (state->finish_skew < hashtable->nSkewBuckets)
		{
			int			j = hashtable->skewBucketNums[state->finish_skew];

			tuple = hashtable->skewBucket[j]->tuples;
			state->finish_skew++;
		}
		else
		{
			state->finish_done = true;
			break;
		}

		while (tuple != NULL &&
			   HeapTupleHeaderHasMatch(HJTUPLE_MINTUPLE(tuple)))
			tuple = tuple->next.unshared;

		/*
		 * Park the cursor on the tuple this call will emit (or NULL when
		 * the chain ended), so a capacity stop resumes after it and an
		 * exhausted chain moves to the next bucket.
		 */
		state->finish_tuple = tuple;
		if (tuple == NULL)
			continue;

		ExecStoreMinimalTuple(HJTUPLE_MINTUPLE(tuple), hslot, false);
		slot_getallattrs(hslot);
		for (c = 0; c < state->out_ncols; c++)
		{
			Datum		value = 0;
			bool		isnull = true;

			if (!state->out_is_outer[c])
				value = slot_getattr(hslot, state->out_src[c], &isnull);
			value = VectorBatchCopyDatum(out, c, value, isnull);
			VectorBatchSetValue(out, c, nrows, value, isnull);
		}
		nrows++;
	}
	return nrows;
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
