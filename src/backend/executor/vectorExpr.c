/*-------------------------------------------------------------------------
 *
 * vectorExpr.c
 *	  Vectorized evaluation of scalar expressions and scan quals.
 *
 * Expression nodes are evaluated over a selection vector.  Function calls
 * use PostgreSQL's own fmgr implementations.  AND/OR evaluate each argument
 * only on rows for which preceding arguments did not decide the result, so
 * their short-circuit behaviour matches the row executor.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "access/htup_details.h"
#include "access/tupconvert.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type_d.h"
#include "cdb/cdbvars.h"
#include "commands/sequence.h"
#include "common/int.h"
#include "executor/execExpr.h"
#include "executor/nodeSubplan.h"
#include "executor/vectorExpr.h"
#include "executor/nodeSubplan.h"
#include "fmgr.h"
#include "funcapi.h"
#include "jit/jit.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "nodes/nodeFuncs.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/date.h"
#include "utils/float.h"
#include "utils/fmgroids.h"
#include "utils/lsyscache.h"
#include "utils/numeric.h"
#include "utils/timestamp.h"
#include "utils/typcache.h"

typedef enum VectorExprKind
{
	VEC_VAR,
	VEC_CONST,
	VEC_RELABEL,
	VEC_FUNC,
	VEC_AND,
	VEC_OR,
	VEC_NOT,
	VEC_NULLTEST,
	VEC_COALESCE,
	VEC_CASE,
	VEC_CASETEST,
	VEC_SAOP,
	VEC_BOOLTEST,
	VEC_NULLIF,
	VEC_DISTINCT,
	VEC_MINMAX,
	VEC_ARRAY,
	VEC_SUBSCRIPT,
	VEC_SQLVALUE,
	VEC_NEXTVAL,
	VEC_ARRAYCOERCE,
	VEC_ROW,
	VEC_FIELDSELECT,
	VEC_CONVERTROWTYPE,
	VEC_WHOLEVAR,
	VEC_PARAM,
	VEC_SUBPLAN
} VectorExprKind;

typedef enum VectorCompareOp
{
	VEC_COMPARE_NONE,
	VEC_COMPARE_INT4_EQ,
	VEC_COMPARE_INT4_NE,
	VEC_COMPARE_INT4_LT,
	VEC_COMPARE_INT4_LE,
	VEC_COMPARE_INT4_GT,
	VEC_COMPARE_INT4_GE,
	VEC_COMPARE_INT8_EQ,
	VEC_COMPARE_INT8_NE,
	VEC_COMPARE_INT8_LT,
	VEC_COMPARE_INT8_LE,
	VEC_COMPARE_INT8_GT,
	VEC_COMPARE_INT8_GE,
	VEC_COMPARE_DATE_EQ,
	VEC_COMPARE_DATE_NE,
	VEC_COMPARE_DATE_LT,
	VEC_COMPARE_DATE_LE,
	VEC_COMPARE_DATE_GT,
	VEC_COMPARE_DATE_GE,
	VEC_COMPARE_TIMESTAMP_EQ,
	VEC_COMPARE_TIMESTAMP_NE,
	VEC_COMPARE_TIMESTAMP_LT,
	VEC_COMPARE_TIMESTAMP_LE,
	VEC_COMPARE_TIMESTAMP_GT,
	VEC_COMPARE_TIMESTAMP_GE,
	VEC_COMPARE_NUMERIC_EQ,
	VEC_COMPARE_NUMERIC_NE,
	VEC_COMPARE_NUMERIC_LT,
	VEC_COMPARE_NUMERIC_LE,
	VEC_COMPARE_NUMERIC_GT,
	VEC_COMPARE_NUMERIC_GE,
	VEC_COMPARE_FLOAT4_EQ,
	VEC_COMPARE_FLOAT4_NE,
	VEC_COMPARE_FLOAT4_LT,
	VEC_COMPARE_FLOAT4_LE,
	VEC_COMPARE_FLOAT4_GT,
	VEC_COMPARE_FLOAT4_GE,
	VEC_COMPARE_FLOAT8_EQ,
	VEC_COMPARE_FLOAT8_NE,
	VEC_COMPARE_FLOAT8_LT,
	VEC_COMPARE_FLOAT8_LE,
	VEC_COMPARE_FLOAT8_GT,
	VEC_COMPARE_FLOAT8_GE
} VectorCompareOp;

typedef enum VectorArithmeticOp
{
	VEC_ARITHMETIC_NONE,
	VEC_ARITHMETIC_INT4_ADD,
	VEC_ARITHMETIC_INT4_SUB,
	VEC_ARITHMETIC_INT4_MUL,
	VEC_ARITHMETIC_INT4_DIV,
	VEC_ARITHMETIC_INT4_MOD,
	VEC_ARITHMETIC_INT8_ADD,
	VEC_ARITHMETIC_INT8_SUB,
	VEC_ARITHMETIC_INT8_MUL,
	VEC_ARITHMETIC_INT8_DIV,
	VEC_ARITHMETIC_INT8_MOD,
	VEC_ARITHMETIC_NUMERIC_ADD,
	VEC_ARITHMETIC_NUMERIC_SUB,
	VEC_ARITHMETIC_NUMERIC_MUL,
	VEC_ARITHMETIC_FLOAT8_ADD,
	VEC_ARITHMETIC_FLOAT8_SUB,
	VEC_ARITHMETIC_FLOAT8_MUL,
	VEC_ARITHMETIC_FLOAT8_DIV,
	VEC_ARITHMETIC_FLOAT4_ADD,
	VEC_ARITHMETIC_FLOAT4_SUB,
	VEC_ARITHMETIC_FLOAT4_MUL,
	VEC_ARITHMETIC_FLOAT4_DIV
} VectorArithmeticOp;

typedef struct VectorValue
{
	Oid			typid;
	int32		typmod;
	int16		typlen;
	bool		typbyval;
	Datum	   *values;
	uint64	   *nulls;
} VectorValue;

typedef struct VectorExprStep
{
	VectorExprKind kind;
	VectorValue value;
	struct VectorExprStep **args;
	int			nargs;
	VectorColumn *column_cache;	/* VEC_VAR: the aliased batch column */

	/*
	 * Deferred numeric results (numeric arithmetic fast path): an aggregate
	 * consumer asks for (mantissa, dscale) instead of a built Numeric, so
	 * the transition skips make_result() and the decode of the result.
	 */
	bool		defer_build;
	int64	   *defer_mant;
	int16	   *defer_dscale;
	uint64	   *defer_valid;
	int32	   *scratch[2];
	struct VectorExprStep *case_operand;
	bool	   *array_nulls;
	union
	{
		struct
		{
			FunctionCallInfo fcinfo;
			FmgrInfo	fn;
			bool		strict;
			VectorCompareOp compare;
			VectorArithmeticOp arithmetic;
		}			func;
		NullTestType nulltesttype;
		BoolTestType booltesttype;
		struct
		{
			/* SAOP needs the function machinery and the elements together,
			 * so it cannot share the func member. */
			FunctionCallInfo fcinfo;
			FmgrInfo	fn;
			bool		strict;
			Datum	   *elems;
			bool	   *elem_nulls;
			int			nelems;
			int16		typlen;
			bool		typbyval;
			char		typalign;
			bool		const_array;
			bool		useOr;		/* = ANY when set, <> ALL otherwise */
		}			saop;
		struct
		{
			FunctionCallInfo fcinfo;
			FmgrInfo	fn;
			MinMaxOp	op;
		}			minmax;
		Datum	   *array_elems;
		struct
		{
			int			attrlen;
			int16		elmlen;
			bool		elembyval;
			char		elemalign;
			int			nupper;
			int			nlower;
			bool	   *upperprovided;
			bool	   *lowerprovided;
			int		   *upperindex;
			int		   *lowerindex;
		}			subscript;
		SQLValueFunctionOp sqlvalueop;
		struct
		{
			Oid			seqid;
			Oid			seqtypid;
		}			nextval;
		struct
		{
			Oid			resultelemtype;
			int			attrlen;
			int16		elmlen;
			bool		elembyval;
			char		elemalign;
			int16		resultlen;
			bool		resultbyval;
			char		resultalign;
			struct VectorExprStep *case_step;
			struct VectorExprStep *elemstep;
		}			arraycoerce;
		struct
		{
			TupleDesc	tupdesc;
			Datum	   *values;
			bool	   *nulls;
			int			nelems;
			int		   *argattr;
		}			row;
		struct
		{
			ExprEvalStep *op;
			Datum		value;
			bool		isnull;
		}			fieldselect;
		struct
		{
			ExprEvalStep *op;
			ExprContext *econtext;
			Datum		value;
			bool		isnull;
		}			convertrowtype;
		struct
		{
			TupleDesc	tupdesc;
			Datum	   *values;
			bool	   *nulls;
		}			wholevar;
		struct
		{
			EState	   *estate;
			int			paramid;
		}			param;
		struct
		{
			EState	   *estate;
			int			plan_id;
			bool		is_initplan;
			SubPlanState *sstate;	/* correlated: per-row execution state */
			ExprContext *econtext;
			TupleTableSlot *slot;	/* scan-shaped row of the current batch */
			AttrNumber *outer_attnos;	/* Agg outer tlist -> batch columns */
			int			outer_ncols;
		}			subplan;
	}			u;
} VectorExprStep;

struct VectorExprState
{
	MemoryContext context;
	VectorExprStep *root;
	int			capacity;
	EState	   *estate;
	VectorJitNode *jit_tree;
	VectorExprJitRuntime jit_rt;
	void	   *jit_func;
};

static VectorExprStep *build_step(Expr *expr, VectorBatch *batch,
								  const VectorExprVarMap *vars,
								  VectorExprStep *case_test);
static void defer_arith_children(VectorExprStep *step, int capacity,
								 MemoryContext context);
static void eval_step(VectorExprStep *step, VectorBatch *batch,
					  int32 *selection, int nselected);

static inline bool
value_is_null(VectorValue *value, int row)
{
	return (value->nulls[row / 64] & ((uint64) 1 << (row % 64))) != 0;
}

static inline void
value_set(VectorValue *value, int row, Datum datum, bool isnull)
{
	uint64		mask = (uint64) 1 << (row % 64);

	value->values[row] = datum;
	if (isnull)
		value->nulls[row / 64] |= mask;
	else
		value->nulls[row / 64] &= ~mask;
}

static void
init_value(VectorValue *value, Oid typid, int32 typmod, int capacity,
		   bool allocate)
{
	value->typid = typid;
	value->typmod = typmod;
	get_typlenbyval(typid, &value->typlen, &value->typbyval);
	if (allocate)
	{
		value->values = palloc0(sizeof(Datum) * capacity);
		value->nulls = palloc0(sizeof(uint64) * VECTOR_BATCH_NULLWORDS(capacity));
	}
}

static AttrNumber
map_var_attno(Var *var, const VectorExprVarMap *vars)
{
	if (var->varlevelsup != 0)
		return 0;
	if (var->varattno == 0)
		return var->varno == vars->scanrelid ? -1 : 0;
	if (var->varattno < 0)
		return 0;
	if (var->varno == vars->scanrelid)
	{
		if (vars->self_remap != NULL)
			return vars->self_remap[var->varattno - 1];
		return var->varattno;
	}
	if (var->varno == OUTER_VAR && vars->outer_attnos != NULL &&
		var->varattno <= vars->outer_ncols)
		return vars->outer_attnos[var->varattno - 1];
	return 0;
}

bool
VectorExprIsSupported(Expr *expr, const VectorExprVarMap *vars)
{
	ListCell   *lc;

	if (expr == NULL)
		return true;

	switch (nodeTag(expr))
	{
		case T_Var:
			{
				Var		   *var = (Var *) expr;

				return map_var_attno(var, vars) != 0;
			}
		case T_Const:
			return true;
		case T_RelabelType:
			return VectorExprIsSupported(((RelabelType *) expr)->arg, vars);
		case T_NullTest:
			{
				NullTest   *test = (NullTest *) expr;

				return !test->argisrow &&
					VectorExprIsSupported(test->arg, vars);
			}
		case T_BoolExpr:
			foreach(lc, ((BoolExpr *) expr)->args)
			{
				if (!VectorExprIsSupported((Expr *) lfirst(lc), vars))
					return false;
			}
			return true;
		case T_List:
			foreach(lc, (List *) expr)
			{
				if (!VectorExprIsSupported((Expr *) lfirst(lc), vars))
					return false;
			}
			return true;
		case T_DistinctExpr:
		case T_NullIfExpr:
		case T_OpExpr:
			{
				OpExpr	   *op = (OpExpr *) expr;

				if (!OidIsValid(op->opfuncid) ||
					func_volatile(op->opfuncid) != PROVOLATILE_IMMUTABLE)
					return false;
				foreach(lc, op->args)
				{
					if (!VectorExprIsSupported((Expr *) lfirst(lc), vars))
						return false;
				}
				return true;
			}
		case T_FuncExpr:
			{
				FuncExpr   *func = (FuncExpr *) expr;

				if (func->funcvariadic ||
					func_volatile(func->funcid) != PROVOLATILE_IMMUTABLE)
					return false;
				foreach(lc, func->args)
				{
					if (!VectorExprIsSupported((Expr *) lfirst(lc), vars))
						return false;
				}
				return true;
			}
		case T_MinMaxExpr:
			{
				MinMaxExpr *mm = (MinMaxExpr *) expr;
				TypeCacheEntry *typentry = lookup_type_cache(mm->minmaxtype,
													TYPECACHE_CMP_PROC);

				if (!OidIsValid(typentry->cmp_proc) ||
					func_volatile(typentry->cmp_proc) != PROVOLATILE_IMMUTABLE)
					return false;
				foreach(lc, mm->args)
				{
					if (!VectorExprIsSupported((Expr *) lfirst(lc), vars))
						return false;
				}
				return true;
			}
		case T_CoalesceExpr:
			foreach(lc, ((CoalesceExpr *) expr)->args)
			{
				if (!VectorExprIsSupported((Expr *) lfirst(lc), vars))
					return false;
			}
			return true;
		case T_CaseExpr:
			{
				CaseExpr   *ce = (CaseExpr *) expr;

				if (ce->arg != NULL &&
					!VectorExprIsSupported(ce->arg, vars))
					return false;
				foreach(lc, ce->args)
				{
					CaseWhen   *w = (CaseWhen *) lfirst(lc);

					if (!VectorExprIsSupported(w->expr, vars) ||
						!VectorExprIsSupported(w->result, vars))
						return false;
				}
				return ce->defresult == NULL ||
					VectorExprIsSupported(ce->defresult, vars);
			}
		case T_CaseTestExpr:
			return true;
		case T_ArrayExpr:
			{
				ArrayExpr  *ae = (ArrayExpr *) expr;

				if (ae->multidims)
					return false;
				foreach(lc, ae->elements)
				{
					if (!VectorExprIsSupported((Expr *) lfirst(lc), vars))
						return false;
				}
				return true;
			}
		case T_ScalarArrayOpExpr:
			{
				ScalarArrayOpExpr *sa = (ScalarArrayOpExpr *) expr;
				Expr	   *rexpr;

				if (!OidIsValid(sa->opfuncid) ||
					func_volatile(sa->opfuncid) != PROVOLATILE_IMMUTABLE)
					return false;
				if (list_length(sa->args) != 2)
					return false;
				rexpr = lsecond(sa->args);
				if (!OidIsValid(get_element_type(exprType((Node *) rexpr))) ||
					!VectorExprIsSupported(rexpr, vars))
					return false;
				return VectorExprIsSupported(linitial(sa->args), vars);
			}
		case T_SubscriptingRef:
			{
				SubscriptingRef *sr = (SubscriptingRef *) expr;
				ListCell   *idx;

				if (sr->refassgnexpr != NULL)
					return false;
				foreach(idx, sr->refupperindexpr)
				{
					Expr	   *e = (Expr *) lfirst(idx);

					if (e != NULL && !VectorExprIsSupported(e, vars))
						return false;
				}
				foreach(idx, sr->reflowerindexpr)
				{
					Expr	   *e = (Expr *) lfirst(idx);

					if (e != NULL && !VectorExprIsSupported(e, vars))
						return false;
				}
				return VectorExprIsSupported(sr->refexpr, vars);
			}
		case T_ConvertRowtypeExpr:
			return VectorExprIsSupported(((ConvertRowtypeExpr *) expr)->arg, vars);
		case T_RowExpr:
			foreach(lc, ((RowExpr *) expr)->args)
			{
				if (!VectorExprIsSupported((Expr *) lfirst(lc), vars))
					return false;
			}
			return true;
		case T_FieldSelect:
			return ((FieldSelect *) expr)->fieldnum > 0 &&
				VectorExprIsSupported(((FieldSelect *) expr)->arg, vars);
		case T_ArrayCoerceExpr:
			{
				ArrayCoerceExpr *ac = (ArrayCoerceExpr *) expr;

				return VectorExprIsSupported(ac->arg, vars) &&
					(ac->elemexpr == NULL ||
					 VectorExprIsSupported(ac->elemexpr, vars));
			}
		case T_SQLValueFunction:
			return true;
		case T_NextValueExpr:
			{
				NextValueExpr *nve = (NextValueExpr *) expr;

				return nve->typeId == INT2OID || nve->typeId == INT4OID ||
					nve->typeId == INT8OID;
			}
		case T_BooleanTest:
			return VectorExprIsSupported(((BooleanTest *) expr)->arg, vars);
		case T_Param:
			return ((Param *) expr)->paramkind == PARAM_EXEC;
		case T_SubPlan:
			{
				SubPlan    *sp = (SubPlan *) expr;

				/*
				 * Same restriction as in build_step(): only InitPlans and
				 * uncorrelated setParam subplans read a materialized exec
				 * param; correlated subplans must run per row and are left
				 * to the standard executor.
				 */
				return (sp->parParam == NIL &&
						(sp->is_initplan || sp->setParam != NIL)) ||
					(vars->allow_correlated && sp->parParam != NIL &&
					 sp->setParam == NIL);
			}
		default:
			return false;
	}
}

static VectorExprStep *
alloc_step(VectorExprKind kind, int nargs, int capacity)
{
	VectorExprStep *step = palloc0(sizeof(VectorExprStep));

	step->kind = kind;
	step->nargs = nargs;
	if (nargs > 0)
		step->args = palloc0(sizeof(VectorExprStep *) * nargs);
	if (kind == VEC_AND || kind == VEC_OR || kind == VEC_COALESCE ||
		kind == VEC_CASE)
	{
		step->scratch[0] = palloc(sizeof(int32) * capacity);
		step->scratch[1] = palloc(sizeof(int32) * capacity);
	}
	return step;
}

static VectorCompareOp
vector_compare_op(Oid functionid)
{
	switch (functionid)
	{
		case F_INT4EQ:
			return VEC_COMPARE_INT4_EQ;
		case F_INT4NE:
			return VEC_COMPARE_INT4_NE;
		case F_INT4LT:
			return VEC_COMPARE_INT4_LT;
		case F_INT4LE:
			return VEC_COMPARE_INT4_LE;
		case F_INT4GT:
			return VEC_COMPARE_INT4_GT;
		case F_INT4GE:
			return VEC_COMPARE_INT4_GE;
		case F_INT8EQ:
			return VEC_COMPARE_INT8_EQ;
		case F_INT8NE:
			return VEC_COMPARE_INT8_NE;
		case F_INT8LT:
			return VEC_COMPARE_INT8_LT;
		case F_INT8LE:
			return VEC_COMPARE_INT8_LE;
		case F_INT8GT:
			return VEC_COMPARE_INT8_GT;
		case F_INT8GE:
			return VEC_COMPARE_INT8_GE;
		case F_DATE_EQ:
			return VEC_COMPARE_DATE_EQ;
		case F_DATE_NE:
			return VEC_COMPARE_DATE_NE;
		case F_DATE_LT:
			return VEC_COMPARE_DATE_LT;
		case F_DATE_LE:
			return VEC_COMPARE_DATE_LE;
		case F_DATE_GT:
			return VEC_COMPARE_DATE_GT;
		case F_DATE_GE:
			return VEC_COMPARE_DATE_GE;
		case F_TIMESTAMP_EQ:	/* timestamptz_eq */
			return VEC_COMPARE_TIMESTAMP_EQ;
		case F_TIMESTAMP_NE:	/* timestamptz_ne */
			return VEC_COMPARE_TIMESTAMP_NE;
		case F_TIMESTAMP_LT:	/* timestamptz_lt */
			return VEC_COMPARE_TIMESTAMP_LT;
		case F_TIMESTAMP_LE:	/* timestamptz_le */
			return VEC_COMPARE_TIMESTAMP_LE;
		case F_TIMESTAMP_GT:	/* timestamptz_gt */
			return VEC_COMPARE_TIMESTAMP_GT;
		case F_TIMESTAMP_GE:	/* timestamptz_ge */
			return VEC_COMPARE_TIMESTAMP_GE;
		case 2052:				/* timestamp_eq */
			return VEC_COMPARE_TIMESTAMP_EQ;
		case 2053:				/* timestamp_ne */
			return VEC_COMPARE_TIMESTAMP_NE;
		case 2054:				/* timestamp_lt */
			return VEC_COMPARE_TIMESTAMP_LT;
		case 2055:				/* timestamp_le */
			return VEC_COMPARE_TIMESTAMP_LE;
		case 2057:				/* timestamp_gt */
			return VEC_COMPARE_TIMESTAMP_GT;
		case 2056:				/* timestamp_ge */
			return VEC_COMPARE_TIMESTAMP_GE;
		case F_NUMERIC_EQ:
			return VEC_COMPARE_NUMERIC_EQ;
		case F_NUMERIC_NE:
			return VEC_COMPARE_NUMERIC_NE;
		case F_NUMERIC_LT:
			return VEC_COMPARE_NUMERIC_LT;
		case F_NUMERIC_LE:
			return VEC_COMPARE_NUMERIC_LE;
		case F_NUMERIC_GT:
			return VEC_COMPARE_NUMERIC_GT;
		case F_NUMERIC_GE:
			return VEC_COMPARE_NUMERIC_GE;
		case F_FLOAT4EQ:
			return VEC_COMPARE_FLOAT4_EQ;
		case F_FLOAT4NE:
			return VEC_COMPARE_FLOAT4_NE;
		case F_FLOAT4LT:
			return VEC_COMPARE_FLOAT4_LT;
		case F_FLOAT4LE:
			return VEC_COMPARE_FLOAT4_LE;
		case F_FLOAT4GT:
			return VEC_COMPARE_FLOAT4_GT;
		case F_FLOAT4GE:
			return VEC_COMPARE_FLOAT4_GE;
		case F_FLOAT8EQ:
			return VEC_COMPARE_FLOAT8_EQ;
		case F_FLOAT8NE:
			return VEC_COMPARE_FLOAT8_NE;
		case F_FLOAT8LT:
			return VEC_COMPARE_FLOAT8_LT;
		case F_FLOAT8LE:
			return VEC_COMPARE_FLOAT8_LE;
		case F_FLOAT8GT:
			return VEC_COMPARE_FLOAT8_GT;
		case F_FLOAT8GE:
			return VEC_COMPARE_FLOAT8_GE;
		default:
			return VEC_COMPARE_NONE;
	}
}

static VectorArithmeticOp
vector_arithmetic_op(Oid functionid)
{
	switch (functionid)
	{
		case F_INT4PL:
			return VEC_ARITHMETIC_INT4_ADD;
		case F_INT4MI:
			return VEC_ARITHMETIC_INT4_SUB;
		case F_INT4MUL:
			return VEC_ARITHMETIC_INT4_MUL;
		case F_INT4DIV:
			return VEC_ARITHMETIC_INT4_DIV;
		case F_INT4MOD:
			return VEC_ARITHMETIC_INT4_MOD;
		case F_INT8PL:
			return VEC_ARITHMETIC_INT8_ADD;
		case F_INT8MI:
			return VEC_ARITHMETIC_INT8_SUB;
		case F_INT8MUL:
			return VEC_ARITHMETIC_INT8_MUL;
		case F_INT8DIV:
			return VEC_ARITHMETIC_INT8_DIV;
		case F_INT8MOD:
			return VEC_ARITHMETIC_INT8_MOD;
		case F_NUMERIC_ADD:
			return VEC_ARITHMETIC_NUMERIC_ADD;
		case F_NUMERIC_SUB:
			return VEC_ARITHMETIC_NUMERIC_SUB;
		case F_NUMERIC_MUL:
			return VEC_ARITHMETIC_NUMERIC_MUL;
		case F_FLOAT8PL:
			return VEC_ARITHMETIC_FLOAT8_ADD;
		case F_FLOAT8MI:
			return VEC_ARITHMETIC_FLOAT8_SUB;
		case F_FLOAT8MUL:
			return VEC_ARITHMETIC_FLOAT8_MUL;
		case F_FLOAT8DIV:
			return VEC_ARITHMETIC_FLOAT8_DIV;
		case F_FLOAT4PL:
			return VEC_ARITHMETIC_FLOAT4_ADD;
		case F_FLOAT4MI:
			return VEC_ARITHMETIC_FLOAT4_SUB;
		case F_FLOAT4MUL:
			return VEC_ARITHMETIC_FLOAT4_MUL;
		case F_FLOAT4DIV:
			return VEC_ARITHMETIC_FLOAT4_DIV;
		default:
			return VEC_ARITHMETIC_NONE;
	}
}

static VectorExprStep *
build_func_step(Oid functionid, Oid resulttype, int32 resulttypmod,
				Oid collation, List *args, VectorBatch *batch,
				const VectorExprVarMap *vars, VectorExprStep *case_test)
{
	VectorExprStep *step;
	ListCell   *lc;
	int			argno = 0;

	step = alloc_step(VEC_FUNC, list_length(args), batch->capacity);
	init_value(&step->value, resulttype, resulttypmod, batch->capacity, true);
	fmgr_info(functionid, &step->u.func.fn);
	step->u.func.fcinfo = palloc0(SizeForFunctionCallInfo(list_length(args)));
	InitFunctionCallInfoData(*step->u.func.fcinfo, &step->u.func.fn,
							 list_length(args), collation, NULL, NULL);
	step->u.func.strict = step->u.func.fn.fn_strict;
	step->u.func.compare = vector_compare_op(functionid);
	step->u.func.arithmetic = vector_arithmetic_op(functionid);

	foreach(lc, args)
	{
		step->args[argno] = build_step((Expr *) lfirst(lc), batch, vars, case_test);
		if (step->args[argno] == NULL)
			return NULL;
		argno++;
	}
	return step;
}

static VectorExprStep *
build_step(Expr *expr, VectorBatch *batch, const VectorExprVarMap *vars,
		   VectorExprStep *case_test)
{
	VectorExprStep *step;
	ListCell   *lc;
	int			argno;

	switch (nodeTag(expr))
	{
		case T_Var:
			{
				Var		   *var = (Var *) expr;
				VectorColumn *column;
				AttrNumber	attno = map_var_attno(var, vars);

				if (attno == -1)
				{
					step = alloc_step(VEC_WHOLEVAR, 0, batch->capacity);
					init_value(&step->value, var->vartype, var->vartypmod,
							   batch->capacity, true);
					step->u.wholevar.tupdesc = batch->tupdesc;
					step->u.wholevar.values = palloc(sizeof(Datum) * batch->ncols);
					step->u.wholevar.nulls = palloc(sizeof(bool) * batch->ncols);
					return step;
				}
				if (attno <= 0 || attno > batch->ncols)
					return NULL;
				step = alloc_step(VEC_VAR, 0, batch->capacity);
				column = &batch->columns[attno - 1];
				step->column_cache = column;
				step->value.typid = column->typid;
				step->value.typmod = column->typmod;
				step->value.typlen = column->typlen;
				step->value.typbyval = column->typbyval;
				step->value.values = column->values;
				step->value.nulls = column->nulls;
				return step;
			}
		case T_Const:
			{
				Const	   *c = (Const *) expr;
				int			row;

				step = alloc_step(VEC_CONST, 0, batch->capacity);
				init_value(&step->value, c->consttype, c->consttypmod,
						   batch->capacity, true);
				for (row = 0; row < batch->capacity; row++)
					value_set(&step->value, row, c->constvalue, c->constisnull);
				return step;
			}
		case T_RelabelType:
			{
				RelabelType *relabel = (RelabelType *) expr;

				step = alloc_step(VEC_RELABEL, 1, batch->capacity);
				step->args[0] = build_step(relabel->arg, batch, vars, case_test);
				if (step->args[0] == NULL)
					return NULL;
				step->value = step->args[0]->value;
				step->value.typid = relabel->resulttype;
				step->value.typmod = relabel->resulttypmod;
				return step;
			}
		case T_DistinctExpr:
		case T_NullIfExpr:
		case T_OpExpr:
			{
				OpExpr	   *op = (OpExpr *) expr;
				Oid			resulttype = IsA(expr, NullIfExpr) ?
					exprType((Node *) expr) : op->opresulttype;
				int32		resulttypmod = IsA(expr, NullIfExpr) ?
					exprTypmod((Node *) expr) : -1;

				step = build_func_step(op->opfuncid, resulttype, resulttypmod,
									   op->inputcollid, op->args, batch, vars, case_test);
				if (step != NULL && IsA(expr, NullIfExpr))
					step->kind = VEC_NULLIF;
				else if (step != NULL && IsA(expr, DistinctExpr))
					step->kind = VEC_DISTINCT;
				return step;
			}
		case T_FuncExpr:
			{
				FuncExpr   *func = (FuncExpr *) expr;

				if (func->funcvariadic)
					return NULL;
				return build_func_step(func->funcid, func->funcresulttype, -1,
									   func->inputcollid, func->args, batch, vars, case_test);
			}
		case T_MinMaxExpr:
			{
				MinMaxExpr *mm = (MinMaxExpr *) expr;
				TypeCacheEntry *typentry;

				step = alloc_step(VEC_MINMAX, list_length(mm->args),
								  batch->capacity);
				init_value(&step->value, mm->minmaxtype, -1,
						   batch->capacity, true);
				typentry = lookup_type_cache(mm->minmaxtype, TYPECACHE_CMP_PROC);
				if (!OidIsValid(typentry->cmp_proc))
					return NULL;
				fmgr_info(typentry->cmp_proc, &step->u.minmax.fn);
				fmgr_info_set_expr((Node *) expr, &step->u.minmax.fn);
				step->u.minmax.fcinfo = palloc0(SizeForFunctionCallInfo(2));
				InitFunctionCallInfoData(*step->u.minmax.fcinfo, &step->u.minmax.fn,
										 2, mm->inputcollid, NULL, NULL);
				step->u.minmax.op = mm->op;
				argno = 0;
				foreach(lc, mm->args)
				{
					step->args[argno] = build_step((Expr *) lfirst(lc), batch, vars,
												   case_test);
					if (step->args[argno] == NULL)
						return NULL;
					argno++;
				}
				return step;
			}
		case T_NullTest:
			{
				NullTest   *test = (NullTest *) expr;

				if (test->argisrow)
					return NULL;
				step = alloc_step(VEC_NULLTEST, 1, batch->capacity);
				init_value(&step->value, BOOLOID, -1, batch->capacity, true);
				step->u.nulltesttype = test->nulltesttype;
				step->args[0] = build_step(test->arg, batch, vars, case_test);
				return step->args[0] ? step : NULL;
			}
		case T_CoalesceExpr:
			{
				CoalesceExpr *ce = (CoalesceExpr *) expr;

				step = alloc_step(VEC_COALESCE, list_length(ce->args),
								  batch->capacity);
				init_value(&step->value, ce->coalescetype, -1,
						   batch->capacity, true);
				argno = 0;
				foreach(lc, ce->args)
				{
					step->args[argno] = build_step((Expr *) lfirst(lc), batch, vars, case_test);
					if (step->args[argno] == NULL)
						return NULL;
					argno++;
				}
				return step;
			}
		case T_CaseExpr:
			{
				CaseExpr   *ce = (CaseExpr *) expr;
				VectorExprStep *when_case = case_test;
				int			nwhens = list_length(ce->args);
				int			slot = 0;

				step = alloc_step(VEC_CASE, 2 * nwhens + 1, batch->capacity);
				init_value(&step->value, ce->casetype, -1, batch->capacity, true);
				if (ce->arg != NULL)
				{
					step->case_operand = build_step(ce->arg, batch, vars, case_test);
					if (step->case_operand == NULL)
						return NULL;
					when_case = step->case_operand;
				}
				foreach(lc, ce->args)
				{
					CaseWhen   *w = (CaseWhen *) lfirst(lc);

					step->args[slot++] = build_step(w->expr, batch, vars, when_case);
					if (step->args[slot - 1] == NULL)
						return NULL;
					step->args[slot++] = build_step(w->result, batch, vars, case_test);
					if (step->args[slot - 1] == NULL)
						return NULL;
				}
				/* the ELSE slot stays NULL when the CASE has no ELSE */
				step->args[slot] = ce->defresult ?
					build_step(ce->defresult, batch, vars, case_test) : NULL;
				if (ce->defresult && step->args[slot] == NULL)
					return NULL;
				return step;
			}
		case T_CaseTestExpr:
			{
				if (case_test == NULL)
					return NULL;
				step = alloc_step(VEC_CASETEST, 0, batch->capacity);
				step->value = case_test->value;
				return step;
			}
		case T_ArrayExpr:
			{
				ArrayExpr  *ae = (ArrayExpr *) expr;
				int			nargs = list_length(ae->elements);

				if (ae->multidims)
					return NULL;
				step = alloc_step(VEC_ARRAY, nargs, batch->capacity);
				init_value(&step->value, ae->array_typeid, -1,
						   batch->capacity, true);
				step->u.array_elems = palloc(sizeof(Datum) * Max(nargs, 1));
				step->array_nulls = palloc(sizeof(bool) * Max(nargs, 1));
				argno = 0;
				foreach(lc, ae->elements)
				{
					step->args[argno] = build_step((Expr *) lfirst(lc), batch, vars,
												   case_test);
					if (step->args[argno] == NULL)
						return NULL;
					argno++;
				}
				return step;
			}
		case T_ScalarArrayOpExpr:
			{
				ScalarArrayOpExpr *sa = (ScalarArrayOpExpr *) expr;
				Expr	   *array_expr = (Expr *) lsecond(sa->args);
				Oid			elemtype = get_element_type(exprType((Node *) array_expr));
				int16		elmlen;
				bool		elembyval;
				char		elemalign;

				if (!OidIsValid(elemtype))
					return NULL;
				step = alloc_step(VEC_SAOP, 2, batch->capacity);
				init_value(&step->value, BOOLOID, -1, batch->capacity, true);
				get_typlenbyvalalign(elemtype, &elmlen, &elembyval, &elemalign);
				step->u.saop.typlen = elmlen;
				step->u.saop.typbyval = elembyval;
				step->u.saop.typalign = elemalign;
				step->u.saop.const_array = IsA(array_expr, Const);
				if (IsA(array_expr, Const) && !((Const *) array_expr)->constisnull)
					deconstruct_array(DatumGetArrayTypeP(((Const *) array_expr)->constvalue),
									  elemtype, elmlen, elembyval, elemalign,
									  &step->u.saop.elems, &step->u.saop.elem_nulls,
									  &step->u.saop.nelems);
				step->u.saop.useOr = sa->useOr;
				fmgr_info(sa->opfuncid, &step->u.saop.fn);
				step->u.saop.fcinfo = palloc0(SizeForFunctionCallInfo(2));
				InitFunctionCallInfoData(*step->u.saop.fcinfo, &step->u.saop.fn,
										 2, sa->inputcollid, NULL, NULL);
				step->u.saop.strict = step->u.saop.fn.fn_strict;
				step->args[0] = build_step((Expr *) linitial(sa->args), batch, vars, case_test);
				step->args[1] = build_step(array_expr, batch, vars, case_test);
				return step->args[0] && step->args[1] ? step : NULL;
			}
		case T_SubscriptingRef:
			{
				SubscriptingRef *sr = (SubscriptingRef *) expr;
				int			nupper = list_length(sr->refupperindexpr);
				int			nlower = list_length(sr->reflowerindexpr);
				int			i = 0;

				if (sr->refassgnexpr != NULL)
					return NULL;
				step = alloc_step(VEC_SUBSCRIPT, 1 + nupper + nlower,
								  batch->capacity);
				init_value(&step->value, exprType((Node *) expr), sr->reftypmod,
						   batch->capacity, true);
				step->u.subscript.attrlen = get_typlen(sr->refcontainertype);
				get_typlenbyvalalign(sr->refelemtype,
									 &step->u.subscript.elmlen,
									 &step->u.subscript.elembyval,
									 &step->u.subscript.elemalign);
				step->u.subscript.nupper = nupper;
				step->u.subscript.nlower = nlower;
				step->u.subscript.upperprovided = palloc0(sizeof(bool) * Max(nupper, 1));
				step->u.subscript.lowerprovided = palloc0(sizeof(bool) * Max(nlower, 1));
				step->u.subscript.upperindex = palloc0(sizeof(int) * Max(nupper, 1));
				step->u.subscript.lowerindex = palloc0(sizeof(int) * Max(nlower, 1));
				step->args[0] = build_step(sr->refexpr, batch, vars, case_test);
				if (step->args[0] == NULL)
					return NULL;
				foreach(lc, sr->refupperindexpr)
				{
					Expr	   *e = (Expr *) lfirst(lc);

					i++;
					if (e == NULL)
						continue;
					step->u.subscript.upperprovided[i - 1] = true;
					step->args[i] = build_step(e, batch, vars, case_test);
					if (step->args[i] == NULL)
						return NULL;
				}
				i = 0;
				foreach(lc, sr->reflowerindexpr)
				{
					Expr	   *e = (Expr *) lfirst(lc);

					i++;
					if (e == NULL)
						continue;
					step->u.subscript.lowerprovided[i - 1] = true;
					step->args[nupper + i] = build_step(e, batch, vars, case_test);
					if (step->args[nupper + i] == NULL)
						return NULL;
				}
				return step;
			}
		case T_ConvertRowtypeExpr:
			{
				ConvertRowtypeExpr *ce = (ConvertRowtypeExpr *) expr;

				step = alloc_step(VEC_CONVERTROWTYPE, 1, batch->capacity);
				init_value(&step->value, ce->resulttype, -1, batch->capacity, true);
				step->u.convertrowtype.op = palloc0(sizeof(ExprEvalStep));
				step->u.convertrowtype.op->resvalue = &step->u.convertrowtype.value;
				step->u.convertrowtype.op->resnull = &step->u.convertrowtype.isnull;
				step->u.convertrowtype.op->d.convert_rowtype.inputtype = exprType((Node *) ce->arg);
				step->u.convertrowtype.op->d.convert_rowtype.outputtype = ce->resulttype;
				step->u.convertrowtype.econtext = palloc0(sizeof(ExprContext));
				step->u.convertrowtype.econtext->ecxt_per_query_memory = CurrentMemoryContext;
				step->args[0] = build_step(ce->arg, batch, vars, case_test);
				return step->args[0] ? step : NULL;
			}
		case T_RowExpr:
			{
				RowExpr    *re = (RowExpr *) expr;
				TupleDesc	tupdesc;
				int			nargs = list_length(re->args);
				int			i = 0;

				if (re->row_typeid == RECORDOID)
				{
					tupdesc = ExecTypeFromExprList(re->args);
					ExecTypeSetColNames(tupdesc, re->colnames);
					BlessTupleDesc(tupdesc);
				}
				else
					tupdesc = lookup_rowtype_tupdesc_copy(re->row_typeid, -1);
				Assert(nargs <= tupdesc->natts);
				step = alloc_step(VEC_ROW, nargs, batch->capacity);
				init_value(&step->value, re->row_typeid, -1, batch->capacity, true);
				step->u.row.tupdesc = tupdesc;
				step->u.row.nelems = Max(nargs, tupdesc->natts);
				step->u.row.values = palloc(sizeof(Datum) * step->u.row.nelems);
				step->u.row.nulls = palloc(sizeof(bool) * step->u.row.nelems);
				step->u.row.argattr = palloc(sizeof(int) * Max(nargs, 1));
				argno = 0;
				foreach(lc, re->args)
				{
					Form_pg_attribute attr = TupleDescAttr(tupdesc, i);
					Expr	   *e = (Expr *) lfirst(lc);

					if (!attr->attisdropped)
					{
						if (exprType((Node *) e) != attr->atttypid)
							ereport(ERROR,
									(errcode(ERRCODE_DATATYPE_MISMATCH),
									 errmsg("ROW() column has type %s instead of type %s",
											format_type_be(exprType((Node *) e)),
											format_type_be(attr->atttypid))));
						step->args[argno] = build_step(e, batch, vars, case_test);
						if (step->args[argno] == NULL)
							return NULL;
						step->u.row.argattr[argno] = i;
						argno++;
					}
					i++;
				}
				return step;
			}
		case T_FieldSelect:
			{
				FieldSelect *fs = (FieldSelect *) expr;

				step = alloc_step(VEC_FIELDSELECT, 1, batch->capacity);
				init_value(&step->value, exprType((Node *) expr), -1,
						   batch->capacity, true);
				step->u.fieldselect.op = palloc0(sizeof(ExprEvalStep));
				step->u.fieldselect.op->resvalue = &step->u.fieldselect.value;
				step->u.fieldselect.op->resnull = &step->u.fieldselect.isnull;
				step->u.fieldselect.op->d.fieldselect.fieldnum = fs->fieldnum;
				step->u.fieldselect.op->d.fieldselect.resulttype = exprType((Node *) expr);
				step->args[0] = build_step(fs->arg, batch, vars, case_test);
				return step->args[0] ? step : NULL;
			}
		case T_ArrayCoerceExpr:
			{
				ArrayCoerceExpr *ac = (ArrayCoerceExpr *) expr;
				Oid			sourceelemtype = get_element_type(exprType((Node *) ac->arg));

				step = alloc_step(VEC_ARRAYCOERCE, 1, batch->capacity);
				init_value(&step->value, ac->resulttype, ac->resulttypmod,
						   batch->capacity, true);
				step->u.arraycoerce.resultelemtype = get_element_type(ac->resulttype);
				step->u.arraycoerce.attrlen = get_typlen(ac->resulttype);
				get_typlenbyvalalign(sourceelemtype,
									 &step->u.arraycoerce.elmlen,
									 &step->u.arraycoerce.elembyval,
									 &step->u.arraycoerce.elemalign);
				get_typlenbyvalalign(step->u.arraycoerce.resultelemtype,
									 &step->u.arraycoerce.resultlen,
									 &step->u.arraycoerce.resultbyval,
									 &step->u.arraycoerce.resultalign);
				if (ac->elemexpr != NULL)
				{
					step->u.arraycoerce.case_step = alloc_step(VEC_CASETEST, 0,
															 batch->capacity);
					init_value(&step->u.arraycoerce.case_step->value,
							   sourceelemtype, -1, batch->capacity, true);
					step->u.arraycoerce.elemstep = build_step(ac->elemexpr, batch, vars,
													 step->u.arraycoerce.case_step);
					if (step->u.arraycoerce.elemstep == NULL)
						return NULL;
				}
				step->args[0] = build_step(ac->arg, batch, vars, case_test);
				return step->args[0] ? step : NULL;
			}
		case T_SQLValueFunction:
			{
				SQLValueFunction *svf = (SQLValueFunction *) expr;
				Datum		value = (Datum) 0;
				bool		isnull = false;
				int			row;

				switch (svf->op)
				{
					case SVFOP_CURRENT_DATE:
						value = DateADTGetDatum(GetSQLCurrentDate());
						break;
					case SVFOP_CURRENT_TIME:
					case SVFOP_CURRENT_TIME_N:
						value = TimeTzADTPGetDatum(GetSQLCurrentTime(svf->typmod));
						break;
					case SVFOP_CURRENT_TIMESTAMP:
					case SVFOP_CURRENT_TIMESTAMP_N:
						value = TimestampTzGetDatum(GetSQLCurrentTimestamp(svf->typmod));
						break;
					case SVFOP_LOCALTIME:
					case SVFOP_LOCALTIME_N:
						value = TimeADTGetDatum(GetSQLLocalTime(svf->typmod));
						break;
					case SVFOP_LOCALTIMESTAMP:
					case SVFOP_LOCALTIMESTAMP_N:
						value = TimestampGetDatum(GetSQLLocalTimestamp(svf->typmod));
						break;
					case SVFOP_CURRENT_ROLE:
					case SVFOP_CURRENT_USER:
					case SVFOP_USER:
						{
							LOCAL_FCINFO(fcinfo, 0);

							InitFunctionCallInfoData(*fcinfo, NULL, 0, InvalidOid, NULL, NULL);
							value = current_user(fcinfo);
							isnull = fcinfo->isnull;
							break;
						}
					case SVFOP_SESSION_USER:
						{
							LOCAL_FCINFO(fcinfo, 0);

							InitFunctionCallInfoData(*fcinfo, NULL, 0, InvalidOid, NULL, NULL);
							value = session_user(fcinfo);
							isnull = fcinfo->isnull;
							break;
						}
					case SVFOP_CURRENT_CATALOG:
						{
							LOCAL_FCINFO(fcinfo, 0);

							InitFunctionCallInfoData(*fcinfo, NULL, 0, InvalidOid, NULL, NULL);
							value = current_database(fcinfo);
							isnull = fcinfo->isnull;
							break;
						}
					case SVFOP_CURRENT_SCHEMA:
						{
							LOCAL_FCINFO(fcinfo, 0);

							InitFunctionCallInfoData(*fcinfo, NULL, 0, InvalidOid, NULL, NULL);
							value = current_schema(fcinfo);
							isnull = fcinfo->isnull;
							break;
						}
				}
				step = alloc_step(VEC_SQLVALUE, 0, batch->capacity);
				init_value(&step->value, svf->type, svf->typmod,
						   batch->capacity, true);
				for (row = 0; row < batch->capacity; row++)
					value_set(&step->value, row, value, isnull);
				return step;
			}
		case T_NextValueExpr:
			{
				NextValueExpr *nve = (NextValueExpr *) expr;

				step = alloc_step(VEC_NEXTVAL, 0, batch->capacity);
				init_value(&step->value, nve->typeId, -1, batch->capacity, true);
				step->u.nextval.seqid = nve->seqid;
				step->u.nextval.seqtypid = nve->typeId;
				return step;
			}
		case T_BooleanTest:
			{
				BooleanTest *bt = (BooleanTest *) expr;

				step = alloc_step(VEC_BOOLTEST, 1, batch->capacity);
				init_value(&step->value, BOOLOID, -1, batch->capacity, true);
				step->u.booltesttype = bt->booltesttype;
				step->args[0] = build_step(bt->arg, batch, vars, case_test);
				return step->args[0] ? step : NULL;
			}
		case T_List:
			{
				List	   *clauses = (List *) expr;

				step = alloc_step(VEC_AND, list_length(clauses), batch->capacity);
				init_value(&step->value, BOOLOID, -1, batch->capacity, true);
				argno = 0;
				foreach(lc, clauses)
				{
					step->args[argno] = build_step((Expr *) lfirst(lc), batch, vars, case_test);
					if (step->args[argno] == NULL)
						return NULL;
					argno++;
				}
				return step;
			}
		case T_BoolExpr:
			{
				BoolExpr   *b = (BoolExpr *) expr;
				VectorExprKind kind;

				if (b->boolop == AND_EXPR)
					kind = VEC_AND;
				else if (b->boolop == OR_EXPR)
					kind = VEC_OR;
				else if (b->boolop == NOT_EXPR)
					kind = VEC_NOT;
				else
					return NULL;

				step = alloc_step(kind, list_length(b->args), batch->capacity);
				init_value(&step->value, BOOLOID, -1, batch->capacity, true);
				argno = 0;
				foreach(lc, b->args)
				{
					step->args[argno] = build_step((Expr *) lfirst(lc), batch, vars, case_test);
					if (step->args[argno] == NULL)
						return NULL;
					argno++;
				}
				return step;
			}
		case T_Param:
			{
				Param	   *param = (Param *) expr;

				if (param->paramkind != PARAM_EXEC || vars->estate == NULL)
					return NULL;
				step = alloc_step(VEC_PARAM, 0, batch->capacity);
				init_value(&step->value, param->paramtype, param->paramtypmod,
						   batch->capacity, true);
				step->u.param.estate = vars->estate;
				step->u.param.paramid = param->paramid;
				return step;
			}
		case T_SubPlan:
			{
				SubPlan    *sp = (SubPlan *) expr;

				if (vars->estate == NULL || sp->plan_id <= 0)
					return NULL;
				/*
				 * Only subplans whose result is materialized into exec params
				 * can be read here: InitPlans and uncorrelated setParam
				 * subplans (EXISTS, NOT IN).  A correlated subplan
				 * (parParam) must be re-executed per row and a subplan
				 * without output params has no value to read; both stay on
				 * the standard executor.
				 */
				if (sp->parParam != NIL && sp->setParam == NIL &&
					vars->allow_correlated && vars->parent != NULL)
				{
					/*
					 * A correlated subplan runs once per batch row through the
					 * standard ExecSubPlan machinery: its parParam expressions
					 * are evaluated in our ExprContext against a slot holding the
					 * current row of the batch.
					 */
					step = alloc_step(VEC_SUBPLAN, 0, batch->capacity);
					init_value(&step->value, exprType((Node *) sp), -1,
							   batch->capacity, true);
				{
					MemoryContext oldcontext = MemoryContextSwitchTo(
						vars->estate->es_query_cxt);

					/* the slot is registered in es_tupleTable and must outlive
					 * this expression state's context */
					step->u.subplan.sstate = ExecInitSubPlan(sp, vars->parent);
					step->u.subplan.econtext = CreateExprContext(vars->estate);
					step->u.subplan.econtext->ecxt_param_exec_vals =
						vars->estate->es_param_exec_vals;
					TupleDesc	slotdesc = batch->tupdesc;

					/* Aggregate correlation Vars reference the outer plan's
					 * targetlist, not the physical batch column order. */
					if (IsA(vars->parent, AggState) && vars->outer_attnos != NULL)
					{
						slotdesc = ((AggState *) vars->parent)->ss.ss_ScanTupleSlot->tts_tupleDescriptor;
						step->u.subplan.outer_attnos = vars->outer_attnos;
						step->u.subplan.outer_ncols = vars->outer_ncols;
					}
					else if (IsA(vars->parent, SeqScanState) &&
							 vars->self_remap != NULL)
					{
						slotdesc = ((SeqScanState *) vars->parent)->ss.ss_ScanTupleSlot->tts_tupleDescriptor;
						step->u.subplan.outer_attnos = vars->self_remap;
						step->u.subplan.outer_ncols = slotdesc->natts;
					}
					step->u.subplan.slot = ExecInitExtraTupleSlot(vars->estate,
															 slotdesc, &TTSOpsVirtual);
					MemoryContextSwitchTo(oldcontext);
				}
					step->u.subplan.is_initplan = false;
					return step;
				}
				if (sp->parParam != NIL ||
					(!sp->is_initplan && sp->setParam == NIL))
					return NULL;
				step = alloc_step(VEC_SUBPLAN, 0, batch->capacity);
				init_value(&step->value, exprType((Node *) sp), -1,
						   batch->capacity, true);
				step->u.subplan.estate = vars->estate;
				step->u.subplan.plan_id = sp->plan_id;
				step->u.subplan.is_initplan = true;
				if (sp->is_initplan)
				{
					ParamExecData *prm = &vars->estate->es_param_exec_vals[sp->plan_id];
					int			row;

					for (row = 0; row < batch->capacity; row++)
						value_set(&step->value, row, prm->value, prm->isnull);
				}
				return step;
			}
		default:
			return NULL;
	}
}

static VectorJitNode *
jit_build_node(VectorExprStep *step, MemoryContext context)
{
	VectorJitNode *node;

	if (step == NULL)
		return NULL;
	node = MemoryContextAllocZero(context, sizeof(VectorJitNode));
	switch (step->kind)
	{
		case VEC_VAR:
			node->kind = VECTOR_JIT_VAR;
			node->values = step->value.values;
			node->nulls = step->value.nulls;
			return node;
		case VEC_CONST:
			if (value_is_null(&step->value, 0))
				return NULL;
			node->kind = VECTOR_JIT_CONST;
			node->const_value = DatumGetInt32(step->value.values[0]);
			return node;
		case VEC_FUNC:
			if (step->nargs != 2)
				return NULL;
			switch (step->u.func.arithmetic)
			{
				case VEC_ARITHMETIC_INT4_ADD:
					node->kind = VECTOR_JIT_INT4_ADD;
					break;
				case VEC_ARITHMETIC_INT4_MUL:
					node->kind = VECTOR_JIT_INT4_MUL;
					break;
				default:
					return NULL;
			}
			node->left = jit_build_node(step->args[0], context);
			node->right = jit_build_node(step->args[1], context);
			if (node->left == NULL || node->right == NULL)
				return NULL;
			return node;
		default:
			return NULL;
	}
}

VectorJitNode *
VectorExprJitGetTree(VectorExprState *state)
{
	return jit_build_node(state->root, state->context);
}

EState *
VectorExprJitEState(VectorExprState *state)
{
	return state->estate;
}

void
VectorExprEvalRowFallback(VectorExprState *state, VectorBatch *batch, int32 row)
{
	eval_step(state->root, batch, &row, 1);
}

VectorExprState *
VectorExprCompile(MemoryContext parent, Expr *expr, VectorBatch *batch,
				  const VectorExprVarMap *vars)
{
	VectorExprState *state;
	MemoryContext oldcontext;

	if (!VectorExprIsSupported(expr, vars))
		return NULL;

	state = MemoryContextAllocZero(parent, sizeof(VectorExprState));
	state->context = AllocSetContextCreate(parent, "Vector Expression",
										 ALLOCSET_DEFAULT_MINSIZE,
										 ALLOCSET_DEFAULT_INITSIZE,
										 ALLOCSET_DEFAULT_MAXSIZE);
	state->capacity = batch->capacity;
	oldcontext = MemoryContextSwitchTo(state->context);
	state->root = build_step(expr, batch, vars, NULL);
	if (state->root != NULL)
		defer_arith_children(state->root, batch->capacity, state->context);
	MemoryContextSwitchTo(oldcontext);
	if (state->root == NULL)
	{
		VectorExprDestroy(state);
		return NULL;
	}
	state->estate = vars->estate;
	state->jit_tree = VectorExprJitGetTree(state);
	if (state->jit_tree != NULL && state->estate != NULL &&
		(state->estate->es_jit_flags & PGJIT_PERFORM) &&
		(state->estate->es_jit_flags & PGJIT_EXPR))
		state->jit_func = jit_compile_vector_expr(state);
	return state;
}

void
VectorExprDestroy(VectorExprState *state)
{
	if (state == NULL)
		return;
	MemoryContextDelete(state->context);
	pfree(state);
}

static int
numeric_compare_datums(Datum left, Datum right)
{
	return cmp_numeric_datums(left, right);
}

static void
eval_compare(VectorExprStep *step, int32 *selection, int nselected)
{
	VectorValue *left = &step->args[0]->value;
	VectorValue *right = &step->args[1]->value;
	int			pos;

	for (pos = 0; pos < nselected; pos++)
	{
		int			row = selection[pos];
		bool		result;

		if (value_is_null(left, row) || value_is_null(right, row))
		{
			value_set(&step->value, row, (Datum) 0, true);
			continue;
		}

		switch (step->u.func.compare)
		{
			case VEC_COMPARE_INT4_EQ:
				result = DatumGetInt32(left->values[row]) == DatumGetInt32(right->values[row]);
				break;
			case VEC_COMPARE_INT4_NE:
				result = DatumGetInt32(left->values[row]) != DatumGetInt32(right->values[row]);
				break;
			case VEC_COMPARE_INT4_LT:
				result = DatumGetInt32(left->values[row]) < DatumGetInt32(right->values[row]);
				break;
			case VEC_COMPARE_INT4_LE:
				result = DatumGetInt32(left->values[row]) <= DatumGetInt32(right->values[row]);
				break;
			case VEC_COMPARE_INT4_GT:
				result = DatumGetInt32(left->values[row]) > DatumGetInt32(right->values[row]);
				break;
			case VEC_COMPARE_INT4_GE:
				result = DatumGetInt32(left->values[row]) >= DatumGetInt32(right->values[row]);
				break;
			case VEC_COMPARE_INT8_EQ:
				result = DatumGetInt64(left->values[row]) == DatumGetInt64(right->values[row]);
				break;
			case VEC_COMPARE_INT8_NE:
				result = DatumGetInt64(left->values[row]) != DatumGetInt64(right->values[row]);
				break;
			case VEC_COMPARE_INT8_LT:
				result = DatumGetInt64(left->values[row]) < DatumGetInt64(right->values[row]);
				break;
			case VEC_COMPARE_INT8_LE:
				result = DatumGetInt64(left->values[row]) <= DatumGetInt64(right->values[row]);
				break;
			case VEC_COMPARE_INT8_GT:
				result = DatumGetInt64(left->values[row]) > DatumGetInt64(right->values[row]);
				break;
			case VEC_COMPARE_INT8_GE:
				result = DatumGetInt64(left->values[row]) >= DatumGetInt64(right->values[row]);
				break;
			case VEC_COMPARE_DATE_EQ:
				result = DatumGetDateADT(left->values[row]) == DatumGetDateADT(right->values[row]);
				break;
			case VEC_COMPARE_DATE_NE:
				result = DatumGetDateADT(left->values[row]) != DatumGetDateADT(right->values[row]);
				break;
			case VEC_COMPARE_DATE_LT:
				result = DatumGetDateADT(left->values[row]) < DatumGetDateADT(right->values[row]);
				break;
			case VEC_COMPARE_DATE_LE:
				result = DatumGetDateADT(left->values[row]) <= DatumGetDateADT(right->values[row]);
				break;
			case VEC_COMPARE_DATE_GT:
				result = DatumGetDateADT(left->values[row]) > DatumGetDateADT(right->values[row]);
				break;
			case VEC_COMPARE_DATE_GE:
				result = DatumGetDateADT(left->values[row]) >= DatumGetDateADT(right->values[row]);
				break;
			case VEC_COMPARE_TIMESTAMP_EQ:
				result = DatumGetTimestamp(left->values[row]) == DatumGetTimestamp(right->values[row]);
				break;
			case VEC_COMPARE_TIMESTAMP_NE:
				result = DatumGetTimestamp(left->values[row]) != DatumGetTimestamp(right->values[row]);
				break;
			case VEC_COMPARE_TIMESTAMP_LT:
				result = DatumGetTimestamp(left->values[row]) < DatumGetTimestamp(right->values[row]);
				break;
			case VEC_COMPARE_TIMESTAMP_LE:
				result = DatumGetTimestamp(left->values[row]) <= DatumGetTimestamp(right->values[row]);
				break;
			case VEC_COMPARE_TIMESTAMP_GT:
				result = DatumGetTimestamp(left->values[row]) > DatumGetTimestamp(right->values[row]);
				break;
			case VEC_COMPARE_TIMESTAMP_GE:
				result = DatumGetTimestamp(left->values[row]) >= DatumGetTimestamp(right->values[row]);
				break;
			case VEC_COMPARE_NUMERIC_EQ:
				result = numeric_compare_datums(left->values[row], right->values[row]) == 0;
				break;
			case VEC_COMPARE_NUMERIC_NE:
				result = numeric_compare_datums(left->values[row], right->values[row]) != 0;
				break;
			case VEC_COMPARE_NUMERIC_LT:
				result = numeric_compare_datums(left->values[row], right->values[row]) < 0;
				break;
			case VEC_COMPARE_NUMERIC_LE:
				result = numeric_compare_datums(left->values[row], right->values[row]) <= 0;
				break;
			case VEC_COMPARE_NUMERIC_GT:
				result = numeric_compare_datums(left->values[row], right->values[row]) > 0;
				break;
			case VEC_COMPARE_NUMERIC_GE:
				result = numeric_compare_datums(left->values[row], right->values[row]) >= 0;
				break;
			case VEC_COMPARE_FLOAT4_EQ:
				result = float4_cmp_internal(DatumGetFloat4(left->values[row]),
										 DatumGetFloat4(right->values[row])) == 0;
				break;
			case VEC_COMPARE_FLOAT4_NE:
				result = float4_cmp_internal(DatumGetFloat4(left->values[row]),
										 DatumGetFloat4(right->values[row])) != 0;
				break;
			case VEC_COMPARE_FLOAT4_LT:
				result = float4_cmp_internal(DatumGetFloat4(left->values[row]),
										 DatumGetFloat4(right->values[row])) < 0;
				break;
			case VEC_COMPARE_FLOAT4_LE:
				result = float4_cmp_internal(DatumGetFloat4(left->values[row]),
										 DatumGetFloat4(right->values[row])) <= 0;
				break;
			case VEC_COMPARE_FLOAT4_GT:
				result = float4_cmp_internal(DatumGetFloat4(left->values[row]),
										 DatumGetFloat4(right->values[row])) > 0;
				break;
			case VEC_COMPARE_FLOAT4_GE:
				result = float4_cmp_internal(DatumGetFloat4(left->values[row]),
										 DatumGetFloat4(right->values[row])) >= 0;
				break;
			case VEC_COMPARE_FLOAT8_EQ:
				result = float8_cmp_internal(DatumGetFloat8(left->values[row]),
										 DatumGetFloat8(right->values[row])) == 0;
				break;
			case VEC_COMPARE_FLOAT8_NE:
				result = float8_cmp_internal(DatumGetFloat8(left->values[row]),
										 DatumGetFloat8(right->values[row])) != 0;
				break;
			case VEC_COMPARE_FLOAT8_LT:
				result = float8_cmp_internal(DatumGetFloat8(left->values[row]),
										 DatumGetFloat8(right->values[row])) < 0;
				break;
			case VEC_COMPARE_FLOAT8_LE:
				result = float8_cmp_internal(DatumGetFloat8(left->values[row]),
										 DatumGetFloat8(right->values[row])) <= 0;
				break;
			case VEC_COMPARE_FLOAT8_GT:
				result = float8_cmp_internal(DatumGetFloat8(left->values[row]),
										 DatumGetFloat8(right->values[row])) > 0;
				break;
			case VEC_COMPARE_FLOAT8_GE:
				result = float8_cmp_internal(DatumGetFloat8(left->values[row]),
										 DatumGetFloat8(right->values[row])) >= 0;
				break;
			default:
				pg_unreachable();
		}
		value_set(&step->value, row, BoolGetDatum(result), false);
	}
}

/*
 * Decode a numeric operand with the column cache: every VEC_VAR step that
 * reads the same column shares the per-batch decoded entries, so a column
 * value is decoded once per batch no matter how many expressions use it.
 */
static bool
numeric_arg_decode(VectorExprStep *arg, int row, uint32 gen,
				   int64 *mant, int *scale, int *dscale)
{
	VectorColumn *col = arg->column_cache;

	/*
	 * A deferred arithmetic child left its fast-path result decoded: the
	 * value is mant * 10^-dscale at its display scale, exactly what the
	 * parent needs; no Numeric was built and there is nothing to decode.
	 */
	if (arg->defer_valid != NULL &&
		(arg->defer_valid[row / 64] & ((uint64) 1 << (row % 64))) != 0)
	{
		*mant = arg->defer_mant[row];
		*scale = *dscale = arg->defer_dscale[row];
		return true;
	}

	if (col != NULL && col->cache_mant != NULL)
	{
		if (col->cache_gen != gen)
		{
			/*
			 * A new batch rewrites the column: entries from the previous
			 * generation are all stale, even rows this batch has not
			 * decoded yet.
			 */
			memset(col->cache_valid, 0,
				   sizeof(uint64) * col->cache_nwords);
			col->cache_gen = gen;
		}
		if ((col->cache_valid[row / 64] & ((uint64) 1 << (row % 64))) != 0)
		{
			*mant = col->cache_mant[row];
			*scale = col->cache_scale[row];
			*dscale = col->cache_dscale[row];
			return true;
		}
		/*
		 * The datum decode handles the 1-byte-header varlenas that AOCS
		 * blocks store compactly, without a detoast copy.
		 */
		if (numeric_decode_fast_datum(arg->value.values[row],
									  mant, scale, dscale))
		{
			col->cache_gen = gen;
			col->cache_mant[row] = *mant;
			col->cache_scale[row] = *scale;
			col->cache_dscale[row] = *dscale;
			col->cache_valid[row / 64] |= (uint64) 1 << (row % 64);
			return true;
		}
		return false;
	}
	return numeric_decode_fast_datum(arg->value.values[row],
									  mant, scale, dscale);
}

/*
 * One numeric fast-path row: compute in int64 when both operands decode
 * exactly and the result matches what the calculator functions would emit
 * (same display scale); otherwise fall back to the fmgr call for this row.
 */
static Datum
arith_arg_datum(VectorExprStep *arg, int row, Datum stored)
{
	/*
	 * A deferred arithmetic child has no datum on its fast-path rows;
	 * materialize it for the parent's fmgr fallback.  The /1000 guard taken
	 * when the child deferred the value makes numeric_build_fast() succeed.
	 */
	if (arg->defer_valid != NULL &&
		(arg->defer_valid[row / 64] & ((uint64) 1 << (row % 64))) != 0)
		return PointerGetDatum(numeric_build_fast(arg->defer_mant[row],
												  arg->defer_dscale[row]));
	return stored;
}

static void
eval_numeric_arith_row(VectorExprStep *step, VectorBatch *batch, int row)
{
	VectorValue *left = &step->args[0]->value;
	VectorValue *right = &step->args[1]->value;
	int64		ma,
				mb;
	int			sa,
				sb,
				da,
				db,
				dres;
	Datum		ldatum = left->values[row];
	Datum		rdatum = right->values[row];

	if (step->defer_valid != NULL)
		step->defer_valid[row / 64] &= ~((uint64) 1 << (row % 64));

	if (numeric_arg_decode(step->args[0], row, batch->gen, &ma, &sa, &da) &&
		numeric_arg_decode(step->args[1], row, batch->gen, &mb, &sb, &db))
	{
		int64		mant;
		bool		ok = true;

		switch (step->u.func.arithmetic)
		{
			case VEC_ARITHMETIC_NUMERIC_ADD:
			case VEC_ARITHMETIC_NUMERIC_SUB:
				{
					int			smax = Max(sa, sb);

					/* Add exactly at the finer scale, then round once. */
					dres = Max(da, db);
					ok = numeric_scale_fast(&ma, sa, smax) &&
						numeric_scale_fast(&mb, sb, smax);
					if (ok)
					{
						int64		sum;

						if (step->u.func.arithmetic == VEC_ARITHMETIC_NUMERIC_ADD)
						{
							if ((mb > 0 && ma > PG_INT64_MAX - mb) ||
								(mb < 0 && ma < PG_INT64_MIN - mb))
								ok = false;
							else
								sum = ma + mb;
						}
						else
						{
							if ((mb < 0 && ma > PG_INT64_MAX + mb) ||
								(mb > 0 && ma < PG_INT64_MIN + mb))
								ok = false;
							else
								sum = ma - mb;
						}
						if (ok)
							ok = numeric_round_scale_fast(&sum, smax, dres);
						mant = sum;
					}
				}
				break;
			case VEC_ARITHMETIC_NUMERIC_MUL:
				{
					dres = da + db;
					if (dres > NUMERIC_MAX_DISPLAY_SCALE)
						ok = false;
					else
					{
#ifdef HAVE_INT128
						int128		prod = (int128) ma * (int128) mb;

						if (prod > PG_INT64_MAX || prod < PG_INT64_MIN)
							ok = false;
						else
						{
							mant = (int64) prod;
							ok = numeric_round_scale_fast(&mant, sa + sb, dres);
						}
#else
						ok = false;
#endif
					}
				}
				break;
			default:
				ok = false;
				mant = 0;
				break;
		}

		if (ok)
		{
			/*
			 * An aggregate consumer takes the decoded result directly.
			 * The |mant|/1000 bound guarantees the display-scale padding
			 * in do_numeric_accum_fast() cannot overflow, matching the
			 * cases numeric_build_fast() would accept.
			 */
			if (step->defer_build &&
				mant <= PG_INT64_MAX / 1000 && mant >= PG_INT64_MIN / 1000)
			{
				step->defer_mant[row] = mant;
				step->defer_dscale[row] = (int16) dres;
				step->defer_valid[row / 64] |= (uint64) 1 << (row % 64);
				value_set(&step->value, row, (Datum) 0, false);
				return;
			}
			{
				Numeric		result = numeric_build_fast(mant, dres);

				if (result != NULL)
				{
					value_set(&step->value, row, PointerGetDatum(result), false);
					return;
				}
			}
		}
	}

	/* Fallback: exactly one fmgr call, as the generic path would do it. */
	{
		FunctionCallInfo fcinfo = step->u.func.fcinfo;
		Datum		result;

		fcinfo->args[0].value = arith_arg_datum(step->args[0], row, ldatum);
		fcinfo->args[0].isnull = false;
		fcinfo->args[1].value = arith_arg_datum(step->args[1], row, rdatum);
		fcinfo->args[1].isnull = false;
		fcinfo->isnull = false;
		result = step->u.func.fn.fn_addr(fcinfo);
		value_set(&step->value, row, result, fcinfo->isnull);
	}
}

static void
eval_arithmetic(VectorExprStep *step, VectorBatch *batch, int32 *selection, int nselected)
{
	VectorValue *left = &step->args[0]->value;
	VectorValue *right = &step->args[1]->value;
	int			pos;

	for (pos = 0; pos < nselected; pos++)
	{
		int			row = selection[pos];

		if (value_is_null(left, row) || value_is_null(right, row))
		{
			value_set(&step->value, row, (Datum) 0, true);
			continue;
		}

		switch (step->u.func.arithmetic)
		{
			case VEC_ARITHMETIC_NUMERIC_ADD:
			case VEC_ARITHMETIC_NUMERIC_SUB:
			case VEC_ARITHMETIC_NUMERIC_MUL:
				eval_numeric_arith_row(step, batch, row);
				continue;
			case VEC_ARITHMETIC_INT4_ADD:
				{
					int32		result;

					if (unlikely(pg_add_s32_overflow(DatumGetInt32(left->values[row]),
													 DatumGetInt32(right->values[row]), &result)))
						ereport(ERROR,
								(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
								 errmsg("integer out of range")));
					value_set(&step->value, row, Int32GetDatum(result), false);
					break;
				}
			case VEC_ARITHMETIC_INT4_SUB:
				{
					int32		result;

					if (unlikely(pg_sub_s32_overflow(DatumGetInt32(left->values[row]),
													 DatumGetInt32(right->values[row]), &result)))
						ereport(ERROR,
								(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
								 errmsg("integer out of range")));
					value_set(&step->value, row, Int32GetDatum(result), false);
					break;
				}
			case VEC_ARITHMETIC_INT4_MUL:
				{
					int32		result;

					if (unlikely(pg_mul_s32_overflow(DatumGetInt32(left->values[row]),
													 DatumGetInt32(right->values[row]), &result)))
						ereport(ERROR,
								(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
								 errmsg("integer out of range")));
					value_set(&step->value, row, Int32GetDatum(result), false);
					break;
				}
			case VEC_ARITHMETIC_INT4_DIV:
				{
					int32		arg1 = DatumGetInt32(left->values[row]);
					int32		arg2 = DatumGetInt32(right->values[row]);
					int32		result;

					if (unlikely(arg2 == 0))
						ereport(ERROR,
								(errcode(ERRCODE_DIVISION_BY_ZERO),
								 errmsg("division by zero")));
					if (arg2 == -1)
					{
						if (unlikely(arg1 == PG_INT32_MIN))
							ereport(ERROR,
									(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
									 errmsg("integer out of range")));
						result = -arg1;
					}
					else
						result = arg1 / arg2;
					value_set(&step->value, row, Int32GetDatum(result), false);
					break;
				}
			case VEC_ARITHMETIC_INT4_MOD:
				{
					int32		arg1 = DatumGetInt32(left->values[row]);
					int32		arg2 = DatumGetInt32(right->values[row]);
					int32		result;

					if (unlikely(arg2 == 0))
						ereport(ERROR,
								(errcode(ERRCODE_DIVISION_BY_ZERO),
								 errmsg("division by zero")));
					result = arg2 == -1 ? 0 : arg1 % arg2;
					value_set(&step->value, row, Int32GetDatum(result), false);
					break;
				}
			case VEC_ARITHMETIC_INT8_ADD:
				{
					int64		result;

					if (unlikely(pg_add_s64_overflow(DatumGetInt64(left->values[row]),
													 DatumGetInt64(right->values[row]), &result)))
						ereport(ERROR,
								(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
								 errmsg("bigint out of range")));
					value_set(&step->value, row, Int64GetDatum(result), false);
					break;
				}
			case VEC_ARITHMETIC_INT8_SUB:
				{
					int64		result;

					if (unlikely(pg_sub_s64_overflow(DatumGetInt64(left->values[row]),
													 DatumGetInt64(right->values[row]), &result)))
						ereport(ERROR,
								(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
								 errmsg("bigint out of range")));
					value_set(&step->value, row, Int64GetDatum(result), false);
					break;
				}
			case VEC_ARITHMETIC_INT8_MUL:
				{
					int64		result;

					if (unlikely(pg_mul_s64_overflow(DatumGetInt64(left->values[row]),
													 DatumGetInt64(right->values[row]), &result)))
						ereport(ERROR,
								(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
								 errmsg("bigint out of range")));
					value_set(&step->value, row, Int64GetDatum(result), false);
					break;
				}
			case VEC_ARITHMETIC_INT8_DIV:
				{
					int64		arg1 = DatumGetInt64(left->values[row]);
					int64		arg2 = DatumGetInt64(right->values[row]);
					int64		result;

					if (unlikely(arg2 == 0))
						ereport(ERROR,
								(errcode(ERRCODE_DIVISION_BY_ZERO),
								 errmsg("division by zero")));
					if (arg2 == -1)
					{
						if (unlikely(arg1 == PG_INT64_MIN))
							ereport(ERROR,
									(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
									 errmsg("bigint out of range")));
						result = -arg1;
					}
					else
						result = arg1 / arg2;
					value_set(&step->value, row, Int64GetDatum(result), false);
					break;
				}
			case VEC_ARITHMETIC_INT8_MOD:
				{
					int64		arg1 = DatumGetInt64(left->values[row]);
					int64		arg2 = DatumGetInt64(right->values[row]);
					int64		result;

					if (unlikely(arg2 == 0))
						ereport(ERROR,
								(errcode(ERRCODE_DIVISION_BY_ZERO),
								 errmsg("division by zero")));
					result = arg2 == -1 ? 0 : arg1 % arg2;
					value_set(&step->value, row, Int64GetDatum(result), false);
					break;
				}
				/*
				 * The float helpers are static inline copies of the SQL
				 * functions' bodies, so error texts and NaN/Inf handling
				 * match float8pl/float8div and company exactly.
				 */
			case VEC_ARITHMETIC_FLOAT8_ADD:
				value_set(&step->value, row,
						  Float8GetDatum(float8_pl(DatumGetFloat8(left->values[row]),
											DatumGetFloat8(right->values[row]))),
						  false);
				break;
			case VEC_ARITHMETIC_FLOAT8_SUB:
				value_set(&step->value, row,
						  Float8GetDatum(float8_mi(DatumGetFloat8(left->values[row]),
											DatumGetFloat8(right->values[row]))),
						  false);
				break;
			case VEC_ARITHMETIC_FLOAT8_MUL:
				value_set(&step->value, row,
						  Float8GetDatum(float8_mul(DatumGetFloat8(left->values[row]),
											DatumGetFloat8(right->values[row]))),
						  false);
				break;
			case VEC_ARITHMETIC_FLOAT8_DIV:
				value_set(&step->value, row,
						  Float8GetDatum(float8_div(DatumGetFloat8(left->values[row]),
											DatumGetFloat8(right->values[row]))),
						  false);
				break;
			case VEC_ARITHMETIC_FLOAT4_ADD:
				value_set(&step->value, row,
						  Float4GetDatum(float4_pl(DatumGetFloat4(left->values[row]),
											DatumGetFloat4(right->values[row]))),
						  false);
				break;
			case VEC_ARITHMETIC_FLOAT4_SUB:
				value_set(&step->value, row,
						  Float4GetDatum(float4_mi(DatumGetFloat4(left->values[row]),
											DatumGetFloat4(right->values[row]))),
						  false);
				break;
			case VEC_ARITHMETIC_FLOAT4_MUL:
				value_set(&step->value, row,
						  Float4GetDatum(float4_mul(DatumGetFloat4(left->values[row]),
											DatumGetFloat4(right->values[row]))),
						  false);
				break;
			case VEC_ARITHMETIC_FLOAT4_DIV:
				value_set(&step->value, row,
						  Float4GetDatum(float4_div(DatumGetFloat4(left->values[row]),
											DatumGetFloat4(right->values[row]))),
						  false);
				break;
			default:
				pg_unreachable();
		}
	}
}

static void
eval_func(VectorExprStep *step, VectorBatch *batch, int32 *selection, int nselected)
{
	int			argno;
	int			pos;

	for (argno = 0; argno < step->nargs; argno++)
		eval_step(step->args[argno], batch, selection, nselected);

	if (step->u.func.compare != VEC_COMPARE_NONE)
	{
		eval_compare(step, selection, nselected);
		return;
	}
	if (step->u.func.arithmetic != VEC_ARITHMETIC_NONE)
	{
		eval_arithmetic(step, batch, selection, nselected);
		return;
	}

	for (pos = 0; pos < nselected; pos++)
	{
		int			row = selection[pos];
		bool		hasnull = false;
		Datum		result;

		for (argno = 0; argno < step->nargs; argno++)
		{
			VectorValue *arg = &step->args[argno]->value;
			bool		isnull = value_is_null(arg, row);

			step->u.func.fcinfo->args[argno].value = arg->values[row];
			step->u.func.fcinfo->args[argno].isnull = isnull;
			hasnull |= isnull;
		}

		if (step->u.func.strict && hasnull)
		{
			value_set(&step->value, row, (Datum) 0, true);
			continue;
		}

		step->u.func.fcinfo->isnull = false;
		result = step->u.func.fn.fn_addr(step->u.func.fcinfo);
		value_set(&step->value, row, result, step->u.func.fcinfo->isnull);
	}
}

static void
eval_bool(VectorExprStep *step, VectorBatch *batch, int32 *selection, int nselected)
{
	int32	   *current = step->scratch[0];
	int32	   *next = step->scratch[1];
	int			ncurrent = nselected;
	int			argno;

	memcpy(current, selection, sizeof(int32) * nselected);
	for (argno = 0; argno < step->nargs && ncurrent > 0; argno++)
	{
		VectorExprStep *arg = step->args[argno];
		int			nnext = 0;
		int			pos;
		bool		last = (argno == step->nargs - 1);

		eval_step(arg, batch, current, ncurrent);
		for (pos = 0; pos < ncurrent; pos++)
		{
			int			row = current[pos];
			bool		isnull = value_is_null(&arg->value, row);
			bool		value = DatumGetBool(arg->value.values[row]);

			if (last)
			{
				value_set(&step->value, row, arg->value.values[row], isnull);
				continue;
			}

			if (step->kind == VEC_AND && !isnull && !value)
				value_set(&step->value, row, BoolGetDatum(false), false);
			else if (step->kind == VEC_OR && !isnull && value)
				value_set(&step->value, row, BoolGetDatum(true), false);
			else
				next[nnext++] = row;
		}
		{
			int32	   *swap = current;

			current = next;
			next = swap;
			ncurrent = nnext;
		}
	}
}

static void
eval_step(VectorExprStep *step, VectorBatch *batch, int32 *selection, int nselected)
{
	int			pos;

	switch (step->kind)
	{
		case VEC_VAR:
		case VEC_CONST:
		case VEC_CASETEST:
		case VEC_SQLVALUE:
			return;
		case VEC_RELABEL:
			eval_step(step->args[0], batch, selection, nselected);
			return;
		case VEC_FUNC:
			eval_func(step, batch, selection, nselected);
			return;
		case VEC_AND:
		case VEC_OR:
			eval_bool(step, batch, selection, nselected);
			return;
		case VEC_NOT:
			eval_step(step->args[0], batch, selection, nselected);
			for (pos = 0; pos < nselected; pos++)
			{
				int			row = selection[pos];
				bool		isnull = value_is_null(&step->args[0]->value, row);

				value_set(&step->value, row,
						  isnull ? (Datum) 0 :
						  BoolGetDatum(!DatumGetBool(step->args[0]->value.values[row])),
						  isnull);
			}
			return;
		case VEC_NULLTEST:
			eval_step(step->args[0], batch, selection, nselected);
			for (pos = 0; pos < nselected; pos++)
			{
				int			row = selection[pos];
				bool		isnull = value_is_null(&step->args[0]->value, row);
				bool		result = step->u.nulltesttype == IS_NULL ? isnull : !isnull;

				value_set(&step->value, row, BoolGetDatum(result), false);
			}
			return;
		case VEC_COALESCE:
			{
				int32	   *pending = step->scratch[0];
				int			npending = nselected;
				int			argno;

				memcpy(pending, selection, sizeof(int32) * nselected);
				for (argno = 0; argno < step->nargs && npending > 0; argno++)
				{
					VectorExprStep *arg = step->args[argno];
					int			write = 0;
					int			p;

					eval_step(arg, batch, pending, npending);
					/* in-place: rows kept never overtake the read position */
					for (p = 0; p < npending; p++)
					{
						int			row = pending[p];

						if (value_is_null(&arg->value, row))
							pending[write++] = row;
						else
							value_set(&step->value, row, arg->value.values[row],
									  false);
					}
					npending = write;
				}
				for (pos = 0; pos < npending; pos++)
					value_set(&step->value, pending[pos], (Datum) 0, true);
			}
			return;
		case VEC_CASE:
			{
				int32	   *pending = step->scratch[0];
				int32	   *taken = step->scratch[1];
				int			npending = nselected;
				int			nwhens = (step->nargs - 1) / 2;
				int			w;

				if (step->case_operand != NULL)
					eval_step(step->case_operand, batch, selection, nselected);
				memcpy(pending, selection, sizeof(int32) * nselected);
				for (w = 0; w < nwhens && npending > 0; w++)
				{
					VectorExprStep *when = step->args[2 * w];
					VectorExprStep *then = step->args[2 * w + 1];
					int			ntaken = 0;
					int			keep = 0;
					int			p;

					eval_step(when, batch, pending, npending);
					for (p = 0; p < npending; p++)
					{
						int			row = pending[p];

						if (!value_is_null(&when->value, row) &&
							DatumGetBool(when->value.values[row]))
							taken[ntaken++] = row;
						else
							pending[keep++] = row;
					}
					npending = keep;

					if (ntaken > 0)
					{
						int			t;

						eval_step(then, batch, taken, ntaken);
						for (t = 0; t < ntaken; t++)
						{
							int			row = taken[t];

							value_set(&step->value, row, then->value.values[row],
									  value_is_null(&then->value, row));
						}
					}
				}
				if (npending > 0)
				{
					VectorExprStep *els = step->args[2 * nwhens];
					int			p;

					if (els != NULL)
					{
						eval_step(els, batch, pending, npending);
						for (p = 0; p < npending; p++)
						{
							int			row = pending[p];

							value_set(&step->value, row, els->value.values[row],
									  value_is_null(&els->value, row));
						}
					}
					else
					{
						for (p = 0; p < npending; p++)
							value_set(&step->value, pending[p], (Datum) 0, true);
					}
				}
			}
			return;
		case VEC_ARRAY:
			{
				Oid			elemtype = get_element_type(step->value.typid);
				int16		elmlen;
				bool		elembyval;
				char		elemalign;
				int			argno;

				get_typlenbyvalalign(elemtype, &elmlen, &elembyval, &elemalign);
				for (argno = 0; argno < step->nargs; argno++)
					eval_step(step->args[argno], batch, selection, nselected);
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];
					ArrayType  *array;

					for (argno = 0; argno < step->nargs; argno++)
					{
						step->u.array_elems[argno] = step->args[argno]->value.values[row];
						step->array_nulls[argno] = value_is_null(&step->args[argno]->value, row);
					}
					{
						int			dims[1] = {step->nargs};
						int			lbs[1] = {1};

						array = construct_md_array(step->u.array_elems,
												   step->array_nulls, 1, dims, lbs,
												   elemtype, elmlen, elembyval, elemalign);
					}
					value_set(&step->value, row, PointerGetDatum(array), false);
				}
				return;
			}
		case VEC_SAOP:
			{
				VectorExprStep *arg = step->args[0];
				VectorExprStep *array = step->args[1];
				FunctionCallInfo fcinfo = step->u.saop.fcinfo;

				eval_step(arg, batch, selection, nselected);
				eval_step(array, batch, selection, nselected);
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];
					bool		anynull = false;
					int			decided = -1;
					int			e;

					if (value_is_null(&arg->value, row) ||
						value_is_null(&array->value, row))
					{
						value_set(&step->value, row, (Datum) 0, true);
						continue;
					}
					if (!step->u.saop.const_array)
						deconstruct_array(DatumGetArrayTypeP(array->value.values[row]),
										  get_element_type(array->value.typid),
										  step->u.saop.typlen, step->u.saop.typbyval,
										  step->u.saop.typalign,
										  &step->u.saop.elems,
										  &step->u.saop.elem_nulls,
										  &step->u.saop.nelems);
					fcinfo->args[0].value = arg->value.values[row];
					fcinfo->args[0].isnull = false;
					for (e = 0; e < step->u.saop.nelems && decided < 0; e++)
					{
						Datum		r;
						bool		rnull;

						if (step->u.saop.elem_nulls[e])
						{
							anynull = true;
							continue;
						}
						fcinfo->args[1].value = step->u.saop.elems[e];
						fcinfo->args[1].isnull = false;
						fcinfo->isnull = false;
						r = FunctionCallInvoke(fcinfo);
						rnull = fcinfo->isnull;
						if (!rnull && DatumGetBool(r))
						{
							if (step->u.saop.useOr)
								decided = 1;	/* = ANY saw a match */
						}
						else if (!rnull)
						{
							if (!step->u.saop.useOr)
								decided = 0;	/* <> ALL saw a mismatch */
						}
						else
							anynull = true;
					}
					if (decided == 1)
						value_set(&step->value, row, BoolGetDatum(true), false);
					else if (decided == 0)
						value_set(&step->value, row, BoolGetDatum(false), false);
					else if (anynull)
						value_set(&step->value, row, (Datum) 0, true);
					else
						value_set(&step->value, row,
								  BoolGetDatum(!step->u.saop.useOr), false);
				}
			}
			return;
		case VEC_NULLIF:
			{
				FunctionCallInfo fcinfo = step->u.func.fcinfo;

				for (pos = 0; pos < step->nargs; pos++)
					eval_step(step->args[pos], batch, selection, nselected);
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];
					VectorValue *left = &step->args[0]->value;
					VectorValue *right = &step->args[1]->value;
					bool		leftnull = value_is_null(left, row);
					bool		rightnull = value_is_null(right, row);

					fcinfo->args[0].value = left->values[row];
					fcinfo->args[0].isnull = leftnull;
					fcinfo->args[1].value = right->values[row];
					fcinfo->args[1].isnull = rightnull;
					if (!leftnull && !rightnull)
					{
						Datum		equal;

						fcinfo->isnull = false;
						equal = step->u.func.fn.fn_addr(fcinfo);
						if (!fcinfo->isnull && DatumGetBool(equal))
						{
							value_set(&step->value, row, (Datum) 0, true);
							continue;
						}
					}
					value_set(&step->value, row, left->values[row], leftnull);
				}
				return;
			}
		case VEC_DISTINCT:
			{
				FunctionCallInfo fcinfo = step->u.func.fcinfo;

				for (pos = 0; pos < step->nargs; pos++)
					eval_step(step->args[pos], batch, selection, nselected);
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];
					VectorValue *left = &step->args[0]->value;
					VectorValue *right = &step->args[1]->value;
					bool		leftnull = value_is_null(left, row);
					bool		rightnull = value_is_null(right, row);

					if (leftnull || rightnull)
					{
						value_set(&step->value, row,
								  BoolGetDatum(leftnull != rightnull), false);
						continue;
					}
					fcinfo->args[0].value = left->values[row];
					fcinfo->args[0].isnull = false;
					fcinfo->args[1].value = right->values[row];
					fcinfo->args[1].isnull = false;
					fcinfo->isnull = false;
					{
						Datum		equal = step->u.func.fn.fn_addr(fcinfo);

						value_set(&step->value, row,
								  BoolGetDatum(!DatumGetBool(equal)), fcinfo->isnull);
					}
				}
				return;
			}
		case VEC_MINMAX:
			{
				FunctionCallInfo fcinfo = step->u.minmax.fcinfo;
				int			argno;

				for (argno = 0; argno < step->nargs; argno++)
					eval_step(step->args[argno], batch, selection, nselected);
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];
					Datum		result = (Datum) 0;
					bool		isnull = true;

					for (argno = 0; argno < step->nargs; argno++)
					{
						VectorValue *arg = &step->args[argno]->value;
						int			cmpresult;

						if (value_is_null(arg, row))
							continue;
						if (isnull)
						{
							result = arg->values[row];
							isnull = false;
							continue;
						}
						fcinfo->args[0].value = result;
						fcinfo->args[1].value = arg->values[row];
						fcinfo->isnull = false;
						cmpresult = DatumGetInt32(FunctionCallInvoke(fcinfo));
						if (fcinfo->isnull)
							continue;
						if ((cmpresult > 0 && step->u.minmax.op == IS_LEAST) ||
							(cmpresult < 0 && step->u.minmax.op == IS_GREATEST))
							result = arg->values[row];
					}
					value_set(&step->value, row, result, isnull);
				}
				return;
			}
		case VEC_SUBSCRIPT:
			{
				int			nupper = step->u.subscript.nupper;
				int			nlower = step->u.subscript.nlower;
				int			i;

				for (i = 0; i < step->nargs; i++)
					if (step->args[i] != NULL)
						eval_step(step->args[i], batch, selection, nselected);
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];
					Datum		container = step->args[0]->value.values[row];
					bool		isnull = value_is_null(&step->args[0]->value, row);

					for (i = 0; i < nupper && !isnull; i++)
					{
						if (!step->u.subscript.upperprovided[i])
							continue;
						if (value_is_null(&step->args[1 + i]->value, row))
							isnull = true;
						else
							step->u.subscript.upperindex[i] =
								DatumGetInt32(step->args[1 + i]->value.values[row]);
					}
					for (i = 0; i < nlower && !isnull; i++)
					{
						if (!step->u.subscript.lowerprovided[i])
							continue;
						if (value_is_null(&step->args[1 + nupper + i]->value, row))
							isnull = true;
						else
							step->u.subscript.lowerindex[i] =
								DatumGetInt32(step->args[1 + nupper + i]->value.values[row]);
					}
					if (isnull)
					{
						value_set(&step->value, row, (Datum) 0, true);
						continue;
					}
					if (nlower == 0)
						container = array_get_element(container, nupper,
											 step->u.subscript.upperindex,
											 step->u.subscript.attrlen,
											 step->u.subscript.elmlen,
											 step->u.subscript.elembyval,
											 step->u.subscript.elemalign,
											 &isnull);
					else
						container = array_get_slice(container, nupper,
											step->u.subscript.upperindex,
											step->u.subscript.lowerindex,
											step->u.subscript.upperprovided,
											step->u.subscript.lowerprovided,
											step->u.subscript.attrlen,
											step->u.subscript.elmlen,
											step->u.subscript.elembyval,
											step->u.subscript.elemalign);
					value_set(&step->value, row, container, isnull);
				}
				return;
			}
		case VEC_NEXTVAL:
			for (pos = 0; pos < nselected; pos++)
			{
				int			row = selection[pos];
				int64		newval = nextval_internal(step->u.nextval.seqid, false,
												 Gp_role == GP_ROLE_DISPATCH);
				Datum		value;

				switch (step->u.nextval.seqtypid)
				{
					case INT2OID:
						value = Int16GetDatum((int16) newval);
						break;
					case INT4OID:
						value = Int32GetDatum((int32) newval);
						break;
					case INT8OID:
						value = Int64GetDatum(newval);
						break;
					default:
						elog(ERROR, "unsupported sequence type %u",
							 step->u.nextval.seqtypid);
				}
				value_set(&step->value, row, value, false);
			}
			return;
		case VEC_CONVERTROWTYPE:
			{
				ExprEvalStep *op = step->u.convertrowtype.op;

				eval_step(step->args[0], batch, selection, nselected);
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];

					*op->resvalue = step->args[0]->value.values[row];
					*op->resnull = value_is_null(&step->args[0]->value, row);
					ExecEvalConvertRowtype(NULL, op, step->u.convertrowtype.econtext);
					value_set(&step->value, row, *op->resvalue, *op->resnull);
				}
				return;
			}
		case VEC_WHOLEVAR:
			for (pos = 0; pos < nselected; pos++)
			{
				int			row = selection[pos];
				int			col;
				HeapTuple	tuple;

				for (col = 0; col < batch->ncols; col++)
				{
					step->u.wholevar.values[col] = batch->columns[col].values[row];
					step->u.wholevar.nulls[col] =
						TupleDescAttr(step->u.wholevar.tupdesc, col)->attisdropped ||
						VectorBatchIsNull(batch, col, row);
				}
				tuple = heap_form_tuple(step->u.wholevar.tupdesc,
									  step->u.wholevar.values, step->u.wholevar.nulls);
				value_set(&step->value, row, HeapTupleGetDatum(tuple), false);
			}
			return;
		case VEC_ROW:
			{
				int			argno;

				for (argno = 0; argno < step->nargs; argno++)
					eval_step(step->args[argno], batch, selection, nselected);
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];
					HeapTuple	tuple;

					memset(step->u.row.nulls, true, step->u.row.nelems);
					for (argno = 0; argno < step->nargs; argno++)
					{
						int			att = step->u.row.argattr[argno];

						step->u.row.values[att] = step->args[argno]->value.values[row];
						step->u.row.nulls[att] = value_is_null(&step->args[argno]->value, row);
					}
					tuple = heap_form_tuple(step->u.row.tupdesc,
										  step->u.row.values, step->u.row.nulls);
					value_set(&step->value, row, HeapTupleGetDatum(tuple), false);
				}
				return;
			}
		case VEC_FIELDSELECT:
			{
				ExprEvalStep *op = step->u.fieldselect.op;

				eval_step(step->args[0], batch, selection, nselected);
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];

					*op->resvalue = step->args[0]->value.values[row];
					*op->resnull = value_is_null(&step->args[0]->value, row);
					ExecEvalFieldSelect(NULL, op, NULL);
					value_set(&step->value, row, *op->resvalue, *op->resnull);
				}
				return;
			}
		case VEC_ARRAYCOERCE:
			{
				VectorExprStep *arg = step->args[0];
				VectorExprStep *elemstep = step->u.arraycoerce.elemstep;

				eval_step(arg, batch, selection, nselected);
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];
					ArrayType  *array;

					if (value_is_null(&arg->value, row))
					{
						value_set(&step->value, row, (Datum) 0, true);
						continue;
					}
					if (elemstep == NULL)
					{
						array = DatumGetArrayTypePCopy(arg->value.values[row]);
						ARR_ELEMTYPE(array) = step->u.arraycoerce.resultelemtype;
						value_set(&step->value, row, PointerGetDatum(array), false);
						continue;
					}
					{
						Datum	   *elems;
						bool	   *elem_nulls;
						int			nelems;
						Datum	   *result_elems;
						bool	   *result_nulls;
						int			dims[1];
						int			lbs[1] = {1};
						int			e;

						deconstruct_array(DatumGetArrayTypeP(arg->value.values[row]),
										  get_element_type(arg->value.typid),
										  step->u.arraycoerce.elmlen,
										  step->u.arraycoerce.elembyval,
										  step->u.arraycoerce.elemalign,
										  &elems, &elem_nulls, &nelems);
						result_elems = palloc(sizeof(Datum) * Max(nelems, 1));
						result_nulls = palloc(sizeof(bool) * Max(nelems, 1));
						for (e = 0; e < nelems; e++)
						{
							value_set(&step->u.arraycoerce.case_step->value, row,
									  elems[e], elem_nulls[e]);
							eval_step(elemstep, batch, &row, 1);
							result_elems[e] = elemstep->value.values[row];
							result_nulls[e] = value_is_null(&elemstep->value, row);
						}
						dims[0] = nelems;
						array = construct_md_array(result_elems, result_nulls, 1, dims, lbs,
												   step->u.arraycoerce.resultelemtype,
												   step->u.arraycoerce.resultlen,
												   step->u.arraycoerce.resultbyval,
												   step->u.arraycoerce.resultalign);
						value_set(&step->value, row, PointerGetDatum(array), false);
					}
				}
				return;
			}
		case VEC_PARAM:
			for (pos = 0; pos < nselected; pos++)
			{
				int			row = selection[pos];
				ParamExecData *prm = &step->u.param.estate->es_param_exec_vals[
					step->u.param.paramid];

				value_set(&step->value, row, prm->value, prm->isnull);
			}
			return;
		case VEC_SUBPLAN:
			if (!step->u.subplan.is_initplan)
			{
				/* correlated: re-execute the subplan per row, as ExecSubPlan */
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];
					TupleTableSlot *slot = step->u.subplan.slot;
					Datum		result;
					bool		isnull;
					int			col;

					ExecClearTuple(slot);
					for (col = 0;
						 col < (step->u.subplan.outer_attnos != NULL ?
								step->u.subplan.outer_ncols : batch->ncols);
						 col++)
					{
						int			attno = step->u.subplan.outer_attnos != NULL ?
							step->u.subplan.outer_attnos[col] : col + 1;

						if (attno > 0)
						{
							slot->tts_values[col] = batch->columns[attno - 1].values[row];
							slot->tts_isnull[col] = VectorBatchIsNull(batch, attno - 1, row);
						}
						else
						{
							/* This child attribute is absent from the Append tlist. */
							slot->tts_values[col] = (Datum) 0;
							slot->tts_isnull[col] = true;
						}
					}
					ExecStoreVirtualTuple(slot);
					step->u.subplan.econtext->ecxt_scantuple = slot;
					ResetExprContext(step->u.subplan.econtext);
					step->u.subplan.econtext->ecxt_outertuple = slot;
					result = ExecSubPlan(step->u.subplan.sstate,
										 step->u.subplan.econtext, &isnull);
					value_set(&step->value, row, result, isnull);
				}
				return;
			}
			for (pos = 0; pos < nselected; pos++)
			{
				int			row = selection[pos];
				ParamExecData *prm = &step->u.subplan.estate->es_param_exec_vals[
					step->u.subplan.plan_id];

				value_set(&step->value, row, prm->value, prm->isnull);
			}
			return;
		case VEC_BOOLTEST:
			eval_step(step->args[0], batch, selection, nselected);
			for (pos = 0; pos < nselected; pos++)
			{
				int			row = selection[pos];
				bool		isnull = value_is_null(&step->args[0]->value, row);
				bool		value = !isnull && DatumGetBool(step->args[0]->value.values[row]);
				bool		result = false;

				switch (step->u.booltesttype)
				{
					case IS_TRUE:
						result = value;
						break;
					case IS_NOT_TRUE:
						result = !value;
						break;
					case IS_FALSE:
						result = !isnull && !value;
						break;
					case IS_NOT_FALSE:
						result = isnull || value;
						break;
					case IS_UNKNOWN:
						result = isnull;
						break;
					case IS_NOT_UNKNOWN:
						result = !isnull;
						break;
				}
				value_set(&step->value, row, BoolGetDatum(result), false);
			}
			return;
	}
}

void
VectorExprEval(VectorBatch *batch, VectorExprState *state)
{
	VectorExprEvalSelection(batch, state, batch->selection, batch->nselected);
}

void
VectorExprEvalSelection(VectorBatch *batch, VectorExprState *state,
						int32 *selection, int nselected)
{
	MemoryContext oldcontext;

	if (state == NULL)
		return;

	CHECK_FOR_INTERRUPTS();
	oldcontext = MemoryContextSwitchTo(batch->varlena_context);
	if (state->jit_func != NULL)
	{
		state->jit_rt.expr = state;
		state->jit_rt.batch = batch;
		state->jit_rt.selection = selection;
		state->jit_rt.nselected = nselected;
		state->jit_rt.out_values = state->root->value.values;
		state->jit_rt.out_nulls = state->root->value.nulls;
		((VectorExprJitFunc) state->jit_func)(&state->jit_rt);
	}
	else
		eval_step(state->root, batch, selection, nselected);
	MemoryContextSwitchTo(oldcontext);
}

Datum
VectorExprGetValue(VectorExprState *state, int row, bool *isnull)
{
	*isnull = value_is_null(&state->root->value, row);
	return state->root->value.values[row];
}

/*
 * The root step's value array and null bitmap, so per-row consumers can
 * read rows inline instead of calling VectorExprGetValue() per row.
 */
void
VectorExprRootValue(VectorExprState *state, Datum **values, uint64 **null_words)
{
	*values = state->root->value.values;
	*null_words = state->root->value.nulls;
}

/*
 * Ask a numeric arithmetic expression to leave its fast-path results in
 * decoded form for an aggregate consumer instead of building Numerics.
 * Only the + - * fast path qualifies; anything else keeps building datums.
 */
static bool
step_is_numeric_arith(VectorExprStep *step)
{
	if (step->kind != VEC_FUNC)
		return false;
	switch (step->u.func.arithmetic)
	{
		case VEC_ARITHMETIC_NUMERIC_ADD:
		case VEC_ARITHMETIC_NUMERIC_SUB:
		case VEC_ARITHMETIC_NUMERIC_MUL:
			return true;
		default:
			return false;
	}
}

static void
defer_step_alloc(VectorExprStep *step, int capacity, MemoryContext context)
{
	MemoryContext oldcontext;

	if (step->defer_mant == NULL)
	{
		oldcontext = MemoryContextSwitchTo(context);
		step->defer_mant = palloc(sizeof(int64) * capacity);
		step->defer_dscale = palloc(sizeof(int16) * capacity);
		step->defer_valid = palloc0(sizeof(uint64) * ((capacity + 63) / 64));
		MemoryContextSwitchTo(oldcontext);
	}
	step->defer_build = true;
}

/*
 * Arithmetic parents consume their children's decoded fast-path results
 * directly, so arithmetic children never need to materialize a Numeric
 * except on the rare fmgr fallback.  Roots themselves are deferred only
 * when an aggregate consumer asks (VectorExprDeferNumericResult).
 */
static void
defer_arith_children(VectorExprStep *step, int capacity, MemoryContext context)
{
	int			i;

	if (step_is_numeric_arith(step))
	{
		for (i = 0; i < step->nargs; i++)
			if (step_is_numeric_arith(step->args[i]))
				defer_step_alloc(step->args[i], capacity, context);
	}
	for (i = 0; i < step->nargs; i++)
		defer_arith_children(step->args[i], capacity, context);
}

bool
VectorExprDeferNumericResult(VectorExprState *state)
{
	if (!step_is_numeric_arith(state->root))
		return false;
	defer_step_alloc(state->root, state->capacity, state->context);
	return true;
}

/*
 * The decoded fast-path result of a deferred expression, or false when the
 * row went through the generic build/fmgr path and must be read as a datum.
 */
bool
VectorExprGetDeferredNumeric(VectorExprState *state, int row, int64 *mant,
							 int *dscale)
{
	VectorExprStep *root = state->root;

	if (root->defer_valid == NULL ||
		(root->defer_valid[row / 64] & ((uint64) 1 << (row % 64))) == 0)
		return false;
	*mant = root->defer_mant[row];
	*dscale = root->defer_dscale[row];
	return true;
}

void
VectorExprFilter(VectorBatch *batch, VectorExprState *state)
{
	int			write = 0;
	int			pos;

	if (state == NULL)
		return;

	VectorExprEval(batch, state);
	for (pos = 0; pos < batch->nselected; pos++)
	{
		int			row = batch->selection[pos];

		if (!value_is_null(&state->root->value, row) &&
			DatumGetBool(state->root->value.values[row]))
			batch->selection[write++] = row;
	}
	batch->nselected = write;
}
