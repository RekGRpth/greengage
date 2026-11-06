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

#include "catalog/pg_proc.h"
#include "catalog/pg_type_d.h"
#include "executor/vectorExpr.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "nodes/nodeFuncs.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"

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
	VEC_BOOLTEST
} VectorExprKind;

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
	int32	   *scratch[2];
	struct VectorExprStep *case_operand;
	union
	{
		struct
		{
			FunctionCallInfo fcinfo;
			FmgrInfo	fn;
			bool		strict;
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
			bool		useOr;		/* = ANY when set, <> ALL otherwise */
		}			saop;
	}			u;
} VectorExprStep;

struct VectorExprState
{
	MemoryContext context;
	VectorExprStep *root;
	int			capacity;
};

static VectorExprStep *build_step(Expr *expr, VectorBatch *batch,
								  const VectorExprVarMap *vars,
								  VectorExprStep *case_test);
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
	if (var->varlevelsup != 0 || var->varattno <= 0)
		return 0;
	if (var->varno == vars->scanrelid)
		return var->varattno;
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

				return map_var_attno(var, vars) > 0;
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
				if (!IsA(rexpr, Const) || ((Const *) rexpr)->constisnull)
					return false;
				return VectorExprIsSupported(linitial(sa->args), vars);
			}
		case T_BooleanTest:
			return VectorExprIsSupported(((BooleanTest *) expr)->arg, vars);
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

				if (attno <= 0 || attno > batch->ncols)
					return NULL;
				step = alloc_step(VEC_VAR, 0, batch->capacity);
				column = &batch->columns[attno - 1];
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
		case T_OpExpr:
			{
				OpExpr	   *op = (OpExpr *) expr;

				return build_func_step(op->opfuncid, op->opresulttype, -1,
									   op->inputcollid, op->args, batch, vars, case_test);
			}
		case T_FuncExpr:
			{
				FuncExpr   *func = (FuncExpr *) expr;

				if (func->funcvariadic)
					return NULL;
				return build_func_step(func->funcid, func->funcresulttype, -1,
									   func->inputcollid, func->args, batch, vars, case_test);
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
		case T_ScalarArrayOpExpr:
			{
				ScalarArrayOpExpr *sa = (ScalarArrayOpExpr *) expr;
				Const	   *arr = (Const *) lsecond(sa->args);
				Oid			elemtype = get_element_type(arr->consttype);
				int16		elmlen;
				bool		elembyval;
				char		elemalign;

				if (!OidIsValid(elemtype))
					return NULL;
				step = alloc_step(VEC_SAOP, 1, batch->capacity);
				init_value(&step->value, BOOLOID, -1, batch->capacity, true);
				get_typlenbyvalalign(elemtype, &elmlen, &elembyval, &elemalign);
				deconstruct_array(DatumGetArrayTypeP(arr->constvalue),
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
				return step->args[0] ? step : NULL;
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
		default:
			return NULL;
	}
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
	MemoryContextSwitchTo(oldcontext);
	if (state->root == NULL)
	{
		VectorExprDestroy(state);
		return NULL;
	}
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

static void
eval_func(VectorExprStep *step, VectorBatch *batch, int32 *selection, int nselected)
{
	int			argno;
	int			pos;

	for (argno = 0; argno < step->nargs; argno++)
		eval_step(step->args[argno], batch, selection, nselected);

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
		case VEC_SAOP:
			{
				VectorExprStep *arg = step->args[0];
				FunctionCallInfo fcinfo = step->u.saop.fcinfo;

				eval_step(arg, batch, selection, nselected);
				for (pos = 0; pos < nselected; pos++)
				{
					int			row = selection[pos];
					bool		anynull = false;
					int			decided = -1;
					int			e;

					if (value_is_null(&arg->value, row))
					{
						value_set(&step->value, row, (Datum) 0, true);
						continue;
					}
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
	MemoryContext oldcontext;

	if (state == NULL)
		return;

	CHECK_FOR_INTERRUPTS();
	oldcontext = MemoryContextSwitchTo(batch->varlena_context);
	eval_step(state->root, batch, batch->selection, batch->nselected);
	MemoryContextSwitchTo(oldcontext);
}

Datum
VectorExprGetValue(VectorExprState *state, int row, bool *isnull)
{
	*isnull = value_is_null(&state->root->value, row);
	return state->root->value.values[row];
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
