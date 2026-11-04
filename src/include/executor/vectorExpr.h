/*-------------------------------------------------------------------------
 *
 * vectorExpr.h
 *	  Vectorized evaluation of scalar expressions and scan quals.
 *
 *-------------------------------------------------------------------------
 */
#ifndef VECTOREXPR_H
#define VECTOREXPR_H

#include "executor/vectorBatch.h"
#include "nodes/primnodes.h"

typedef struct VectorExprState VectorExprState;

extern bool VectorExprIsSupported(Expr *expr, Index scanrelid);
extern VectorExprState *VectorExprCompile(MemoryContext parent, Expr *expr,
										  VectorBatch *batch, Index scanrelid);
extern void VectorExprDestroy(VectorExprState *state);
extern void VectorExprFilter(VectorBatch *batch, VectorExprState *state);

#endif							/* VECTOREXPR_H */
