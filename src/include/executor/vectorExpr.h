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

typedef struct VectorExprVarMap
{
	Index		scanrelid;
	AttrNumber *outer_attnos;
	int			outer_ncols;
} VectorExprVarMap;

extern bool VectorExprIsSupported(Expr *expr, const VectorExprVarMap *vars);
extern VectorExprState *VectorExprCompile(MemoryContext parent, Expr *expr,
										  VectorBatch *batch,
										  const VectorExprVarMap *vars);
extern void VectorExprDestroy(VectorExprState *state);
extern void VectorExprEval(VectorBatch *batch, VectorExprState *state);
extern Datum VectorExprGetValue(VectorExprState *state, int row, bool *isnull);
extern void VectorExprFilter(VectorBatch *batch, VectorExprState *state);

#endif							/* VECTOREXPR_H */
