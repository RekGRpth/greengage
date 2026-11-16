/*-------------------------------------------------------------------------
 *
 * vectorHashJoin.h
 *	  Batch-oriented probe side of hash join regions.
 *
 *-------------------------------------------------------------------------
 */
#ifndef VECTORHASHJOIN_H
#define VECTORHASHJOIN_H

#include "executor/nodeHashjoin.h"
#include "executor/vectorBatch.h"

typedef struct VectorHashJoinState VectorHashJoinState;
typedef bool (*VectorHashJoinSourceNext)(void *arg, VectorBatch **batch);

extern bool VectorHashJoinPlanIsSupported(HashJoin *join);
extern VectorHashJoinState *VectorHashJoinCreate(MemoryContext parent,
												 HashJoinState *hj);
extern VectorHashJoinState *VectorHashJoinCreateBatch(MemoryContext parent,
									HashJoinState *hj,
									VectorHashJoinSourceNext next_outer,
									void *outer_arg,
									VectorHashJoinState *outer_join);
extern VectorBatch *VectorHashJoinOutput(VectorHashJoinState *state);
extern bool VectorHashJoinNextBatch(VectorHashJoinState *state);
extern void VectorHashJoinDestroy(VectorHashJoinState *state);
extern TupleTableSlot *VectorHashJoinNextRow(VectorHashJoinState *state,
											 VectorBatch *batch, int position);
extern bool VectorHashJoinInProgress(VectorHashJoinState *state);
extern TupleTableSlot *VectorHashJoinContinue(VectorHashJoinState *state);
extern TupleTableSlot *VectorHashJoinFinish(VectorHashJoinState *state);
extern void VectorHashJoinReset(VectorHashJoinState *state);

/* nodeHashjoin.c entry points into its existing build and probe machinery. */
extern void VectorHashJoinBuild(HashJoinState *node);
extern TupleTableSlot *VectorHashJoinProbe(HashJoinState *node,
										  TupleTableSlot *outer);
extern TupleTableSlot *VectorHashJoinProbeFinal(HashJoinState *node);
extern bool VectorHashJoinNeedsNewOuter(HashJoinState *node);

#endif							/* VECTORHASHJOIN_H */
