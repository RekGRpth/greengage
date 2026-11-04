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
	VEC_NULLTEST
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
	union
	{
		struct
		{
			FunctionCallInfo fcinfo;
			FmgrInfo	fn;
			bool		strict;
		}			func;
		NullTestType nulltesttype;
	}			u;
} VectorExprStep;

struct VectorExprState
{
	MemoryContext context;
	VectorExprStep *root;
	int			capacity;
};

static VectorExprStep *build_step(Expr *expr, VectorBatch *batch, Index scanrelid);
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

bool
VectorExprIsSupported(Expr *expr, Index scanrelid)
{
	ListCell   *lc;

	if (expr == NULL)
		return true;

	switch (nodeTag(expr))
	{
		case T_Var:
			{
				Var		   *var = (Var *) expr;

				return var->varno == scanrelid && var->varattno > 0 &&
					var->varlevelsup == 0;
			}
		case T_Const:
			return true;
		case T_RelabelType:
			return VectorExprIsSupported(((RelabelType *) expr)->arg, scanrelid);
		case T_NullTest:
			{
				NullTest   *test = (NullTest *) expr;

				return !test->argisrow &&
					VectorExprIsSupported(test->arg, scanrelid);
			}
		case T_BoolExpr:
			foreach(lc, ((BoolExpr *) expr)->args)
			{
				if (!VectorExprIsSupported((Expr *) lfirst(lc), scanrelid))
					return false;
			}
			return true;
		case T_List:
			foreach(lc, (List *) expr)
			{
				if (!VectorExprIsSupported((Expr *) lfirst(lc), scanrelid))
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
					if (!VectorExprIsSupported((Expr *) lfirst(lc), scanrelid))
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
					if (!VectorExprIsSupported((Expr *) lfirst(lc), scanrelid))
						return false;
				}
				return true;
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
	if (kind == VEC_AND || kind == VEC_OR)
	{
		step->scratch[0] = palloc(sizeof(int32) * capacity);
		step->scratch[1] = palloc(sizeof(int32) * capacity);
	}
	return step;
}

static VectorExprStep *
build_func_step(Oid functionid, Oid resulttype, int32 resulttypmod,
				Oid collation, List *args, VectorBatch *batch, Index scanrelid)
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
		step->args[argno] = build_step((Expr *) lfirst(lc), batch, scanrelid);
		if (step->args[argno] == NULL)
			return NULL;
		argno++;
	}
	return step;
}

static VectorExprStep *
build_step(Expr *expr, VectorBatch *batch, Index scanrelid)
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

				if (var->varno != scanrelid || var->varattno <= 0 ||
					var->varattno > batch->ncols || var->varlevelsup != 0)
					return NULL;
				step = alloc_step(VEC_VAR, 0, batch->capacity);
				column = &batch->columns[var->varattno - 1];
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
				step->args[0] = build_step(relabel->arg, batch, scanrelid);
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
									   op->inputcollid, op->args, batch, scanrelid);
			}
		case T_FuncExpr:
			{
				FuncExpr   *func = (FuncExpr *) expr;

				if (func->funcvariadic)
					return NULL;
				return build_func_step(func->funcid, func->funcresulttype, -1,
									   func->inputcollid, func->args, batch, scanrelid);
			}
		case T_NullTest:
			{
				NullTest   *test = (NullTest *) expr;

				if (test->argisrow)
					return NULL;
				step = alloc_step(VEC_NULLTEST, 1, batch->capacity);
				init_value(&step->value, BOOLOID, -1, batch->capacity, true);
				step->u.nulltesttype = test->nulltesttype;
				step->args[0] = build_step(test->arg, batch, scanrelid);
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
					step->args[argno] = build_step((Expr *) lfirst(lc), batch, scanrelid);
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
					step->args[argno] = build_step((Expr *) lfirst(lc), batch, scanrelid);
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
				  Index scanrelid)
{
	VectorExprState *state;
	MemoryContext oldcontext;

	if (!VectorExprIsSupported(expr, scanrelid))
		return NULL;

	state = MemoryContextAllocZero(parent, sizeof(VectorExprState));
	state->context = AllocSetContextCreate(parent, "Vector Expression",
										 ALLOCSET_DEFAULT_MINSIZE,
										 ALLOCSET_DEFAULT_INITSIZE,
										 ALLOCSET_DEFAULT_MAXSIZE);
	state->capacity = batch->capacity;
	oldcontext = MemoryContextSwitchTo(state->context);
	state->root = build_step(expr, batch, scanrelid);
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
	}
}

void
VectorExprFilter(VectorBatch *batch, VectorExprState *state)
{
	MemoryContext oldcontext;
	int			write = 0;
	int			pos;

	if (state == NULL)
		return;

	CHECK_FOR_INTERRUPTS();
	oldcontext = MemoryContextSwitchTo(batch->varlena_context);
	eval_step(state->root, batch, batch->selection, batch->nselected);
	for (pos = 0; pos < batch->nselected; pos++)
	{
		int			row = batch->selection[pos];

		if (!value_is_null(&state->root->value, row) &&
			DatumGetBool(state->root->value.values[row]))
			batch->selection[write++] = row;
	}
	batch->nselected = write;
	MemoryContextSwitchTo(oldcontext);
}
