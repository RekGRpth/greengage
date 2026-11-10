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
#include "nodes/execnodes.h"
#include "nodes/primnodes.h"

typedef struct VectorExprState VectorExprState;

typedef struct VectorExprVarMap
{
	Index		scanrelid;
	AttrNumber *outer_attnos;
	int			outer_ncols;
	EState	   *estate;
} VectorExprVarMap;

extern bool VectorExprIsSupported(Expr *expr, const VectorExprVarMap *vars);
extern VectorExprState *VectorExprCompile(MemoryContext parent, Expr *expr,
										  VectorBatch *batch,
										  const VectorExprVarMap *vars);
extern void VectorExprDestroy(VectorExprState *state);
extern void VectorExprEval(VectorBatch *batch, VectorExprState *state);
extern void VectorExprEvalSelection(VectorBatch *batch, VectorExprState *state,
									int32 *selection, int nselected);
extern Datum VectorExprGetValue(VectorExprState *state, int row, bool *isnull);
extern bool VectorExprDeferNumericResult(VectorExprState *state);
extern bool VectorExprGetDeferredNumeric(VectorExprState *state, int row,
										 int64 *mant, int *dscale);
extern void VectorExprFilter(VectorBatch *batch, VectorExprState *state);

#endif							/* VECTOREXPR_H */
