/*-------------------------------------------------------------------------
 *
 * nodeVectorRegion.h
 *	  Built-in CustomScan provider for vectorized plan regions.
 *
 *-------------------------------------------------------------------------
 */
#ifndef NODEVECTORREGION_H
#define NODEVECTORREGION_H

#include "nodes/extensible.h"

extern const CustomScanMethods *GetVectorRegionScanMethods(void);
extern void RegisterVectorRegionScanMethods(void);

/* Batch interconnect: pull the next output batch of a batch-producing
 * vector region (scan region / batch-driven HJ-root).  Returns false
 * when ps is not such a region; *batch == NULL marks EOF. */
struct VectorBatch;
extern bool VectorRegionPullOutputBatch(PlanState *ps, struct VectorBatch **batch);

#endif							/* NODEVECTORREGION_H */
