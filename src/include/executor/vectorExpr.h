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

typedef enum VectorJitNodeKind
{
	VECTOR_JIT_VAR,
	VECTOR_JIT_CONST,
	VECTOR_JIT_INT4_ADD,
	VECTOR_JIT_INT4_MUL,
	VECTOR_JIT_NUMERIC_VAR,
	VECTOR_JIT_NUMERIC_CONST,
	VECTOR_JIT_NUMERIC_ADD,
	VECTOR_JIT_NUMERIC_SUB,
	VECTOR_JIT_NUMERIC_MUL
} VectorJitNodeKind;

typedef struct VectorJitNode
{
	VectorJitNodeKind kind;
	struct VectorJitNode *left;
	struct VectorJitNode *right;
	Datum	   *values;
	uint64	   *nulls;
	int32		const_value;
	struct VectorExprStep *step;
	int64		mantissa;
	int16		scale;
	int16		dscale;
} VectorJitNode;

typedef struct VectorExprJitRuntime
{
	VectorExprState *expr;
	VectorBatch *batch;
	int32	   *selection;
	int			nselected;
	uint32		gen;
	Datum	   *out_values;
	uint64	   *out_nulls;
	int64	   *defer_mant;
	int16	   *defer_dscale;
	uint64	   *defer_valid;
} VectorExprJitRuntime;

typedef void (*VectorExprJitFunc)(VectorExprJitRuntime *runtime);

typedef struct VectorExprVarMap
{
	Index		scanrelid;
	AttrNumber *outer_attnos;
	int			outer_ncols;
	EState	   *estate;
	/*
	 * Optional remap for a scan whose physical attnos differ from the
	 * batch layout: self_remap[attno - 1] gives the 1-based batch column,
	 * or 0 when the attribute has no batch column.
	 */
	AttrNumber *self_remap;
	/* Correlated subplans are executed per row through ExecSubPlan: the
	 * region supplies the parent PlanState and a scan-shaped slot. */
	PlanState  *parent;
	bool		allow_correlated;
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
extern void VectorExprRootValue(VectorExprState *state, Datum **values,
								uint64 **null_words);
extern bool VectorExprDeferNumericResult(VectorExprState *state);
extern bool VectorExprGetDeferredNumeric(VectorExprState *state, int row,
										 int64 *mant, int *dscale);
extern void VectorExprFilter(VectorBatch *batch, VectorExprState *state);
extern VectorJitNode *VectorExprJitGetTree(VectorExprState *state);
extern EState *VectorExprJitEState(VectorExprState *state);
extern void VectorExprEvalRowFallback(VectorExprState *state,
									  VectorBatch *batch, int32 row);
extern bool VectorExprNumericArgDecode(struct VectorExprStep *arg, int row,
									   uint32 gen, int64 *mant, int *scale,
									   int *dscale);
extern void VectorExprStoreDeferred(VectorExprState *state, int32 row,
									int64 mantissa, int dscale, bool isnull);

#endif							/* VECTOREXPR_H */
