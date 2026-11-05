/*-------------------------------------------------------------------------
 *
 * vectorAgg.h
 *	  Batch input processing for PostgreSQL aggregate nodes.
 *
 *-------------------------------------------------------------------------
 */
#ifndef VECTORAGG_H
#define VECTORAGG_H

#include "executor/nodeAgg.h"
#include "executor/vectorBatch.h"

typedef struct VectorAggState VectorAggState;

extern bool VectorAggPlanIsSupported(Agg *agg);
extern VectorAggState *VectorAggCreate(MemoryContext parent, AggState *agg,
									   VectorBatch *batch);
extern void VectorAggDestroy(VectorAggState *state);
extern void VectorAggProcessBatch(VectorAggState *state, VectorBatch *batch);
extern void VectorAggFinish(VectorAggState *state);
extern TupleTableSlot *VectorAggNext(VectorAggState *state);
extern void VectorAggReset(VectorAggState *state);

/* nodeAgg.c helpers sharing its transition and finalization machinery. */
extern void VectorAggAdvanceTransition(AggState *aggstate,
									   AggStatePerGroup pergroup, int transno,
									   Datum *values, bool *isnull);
extern void VectorAggInitializeEntry(AggState *aggstate,
									 TupleHashEntry entry);
extern void VectorAggInitializePlain(AggState *aggstate);
extern void VectorAggFinishInput(AggState *aggstate);
extern TupleTableSlot *VectorAggRetrievePlain(AggState *aggstate);
extern TupleTableSlot *VectorAggRetrieveHashed(AggState *aggstate);

#endif							/* VECTORAGG_H */
