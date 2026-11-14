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

#endif							/* NODEVECTORREGION_H */
