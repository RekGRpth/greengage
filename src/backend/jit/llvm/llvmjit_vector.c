/*-------------------------------------------------------------------------
 *
 * llvmjit_vector.c
 *	  JIT compilation of fixed-width vector expression trees.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#ifdef USE_LLVM

#include <stddef.h>

#include "executor/vectorExpr.h"
#include "jit/llvmjit.h"
#include "jit/llvmjit_emit.h"
#include "utils/numeric.h"

typedef struct JitResult
{
	LLVMValueRef value;
	LLVMValueRef scale;
	LLVMValueRef dscale;
	LLVMValueRef isnull;
	LLVMValueRef overflow;
} JitResult;

static LLVMValueRef
rt_load(LLVMBuilderRef b, LLVMValueRef rt, LLVMTypeRef type,
		size_t offset, const char *name)
{
	LLVMValueRef idx = l_sizet_const(offset /
		(LLVMGetTypeKind(type) == LLVMPointerTypeKind ? sizeof(void *) : sizeof(int32)));
	LLVMValueRef typed_rt = LLVMBuildBitCast(b, rt, l_ptr(type), "");
	LLVMValueRef ptr = l_gep(b, type, typed_rt, &idx, 1, name);

	return l_load(b, type, ptr, name);
}

static LLVMValueRef
bool_const(LLVMContextRef lc, bool value)
{
	return LLVMConstInt(LLVMInt1TypeInContext(lc), value ? 1 : 0, false);
}

static LLVMValueRef
emit_null_bit(LLVMBuilderRef b, LLVMValueRef nulls, LLVMValueRef row)
{
	LLVMContextRef lc = LLVMGetTypeContext(LLVMTypeOf(row));
	LLVMTypeRef i64 = LLVMInt64TypeInContext(lc);
	LLVMValueRef wordidx = LLVMBuildLShr(b, row, l_int32_const(lc, 6), "");
	LLVMValueRef wordptr = l_gep(b, i64, nulls, &wordidx, 1, "");
	LLVMValueRef word = l_load(b, i64, wordptr, "");
	LLVMValueRef bit = LLVMBuildShl(b, l_int64_const(lc, 1),
								   LLVMBuildAnd(b, row, l_int32_const(lc, 63), ""), "");
	LLVMValueRef masked = LLVMBuildAnd(b, word, bit, "");

	return LLVMBuildICmp(b, LLVMIntNE, masked, l_int64_const(lc, 0), "");
}

static JitResult
emit_node(LLVMBuilderRef b, LLVMModuleRef mod, VectorJitNode *node,
		  LLVMValueRef row, LLVMValueRef gen)
{
	JitResult result = {0};
	LLVMContextRef lc = LLVMGetModuleContext(mod);
	LLVMTypeRef i32 = LLVMInt32TypeInContext(lc);
	LLVMTypeRef i64 = LLVMInt64TypeInContext(lc);

	result.overflow = bool_const(lc, false);
	switch (node->kind)
	{
		case VECTOR_JIT_VAR:
			{
				LLVMValueRef values = l_ptr_const(node->values, l_ptr(i64));
				LLVMValueRef nulls = l_ptr_const(node->nulls, l_ptr(i64));
				LLVMValueRef ptr = l_gep(b, i64, values, &row, 1, "");
				LLVMValueRef datum = l_load(b, i64, ptr, "");

				result.value = LLVMBuildTrunc(b, datum, i32, "");
				result.isnull = emit_null_bit(b, nulls, row);
				return result;
			}
		case VECTOR_JIT_CONST:
			result.value = l_int32_const(lc, node->const_value);
			result.isnull = bool_const(lc, false);
			return result;
		case VECTOR_JIT_INT4_ADD:
		case VECTOR_JIT_INT4_MUL:
			{
				JitResult left = emit_node(b, mod, node->left, row, gen);
				JitResult right = emit_node(b, mod, node->right, row, gen);
				const char *intrinsic = node->kind == VECTOR_JIT_INT4_ADD ?
					"llvm.sadd.with.overflow.i32" : "llvm.smul.with.overflow.i32";
				LLVMTypeRef overload[1] = {i32};
				LLVMValueRef fn = LLVMGetIntrinsicDeclaration(mod,
					LLVMLookupIntrinsicID(intrinsic, strlen(intrinsic)), overload, 1);
				LLVMValueRef args[2] = {left.value, right.value};
				LLVMValueRef packed = l_call(b, LLVMGetFunctionType(fn),
											 fn, args, 2, "");
				LLVMValueRef ov;

				result.value = LLVMBuildExtractValue(b, packed, 0, "");
				ov = LLVMBuildExtractValue(b, packed, 1, "");
				result.overflow = LLVMBuildOr(b, left.overflow, right.overflow, "");
				result.overflow = LLVMBuildOr(b, result.overflow, ov, "");
				result.isnull = LLVMBuildOr(b, left.isnull, right.isnull, "");
				return result;
			}
		case VECTOR_JIT_NUMERIC_VAR:
			{
				LLVMTypeRef params[6] = {l_ptr(LLVMInt8TypeInContext(lc)), i32, i32,
					l_ptr(i64), l_ptr(i32), l_ptr(i32)};
				LLVMTypeRef fn_type = LLVMFunctionType(LLVMInt1TypeInContext(lc), params, 6, false);
				LLVMValueRef fn = l_ptr_const((void *) VectorExprNumericArgDecode,
											 l_ptr(fn_type));
				LLVMValueRef mant_ptr = LLVMBuildAlloca(b, i64, "mant");
				LLVMValueRef scale_ptr = LLVMBuildAlloca(b, i32, "scale");
				LLVMValueRef dscale_ptr = LLVMBuildAlloca(b, i32, "dscale");
				LLVMValueRef args[6] = {l_ptr_const(node->step, l_ptr(LLVMInt8TypeInContext(lc))),
					row, gen, mant_ptr, scale_ptr, dscale_ptr};
				LLVMValueRef valid = l_call(b, fn_type, fn, args, 6, "");

				result.value = l_load(b, i64, mant_ptr, "mant");
				result.scale = l_load(b, i32, scale_ptr, "scale");
				result.dscale = l_load(b, i32, dscale_ptr, "dscale");
				result.isnull = emit_null_bit(b, l_ptr_const(node->nulls, l_ptr(i64)), row);
				result.overflow = LLVMBuildNot(b, valid, "decode_failed");
				return result;
			}
		case VECTOR_JIT_NUMERIC_CONST:
			result.value = l_int64_const(lc, node->mantissa);
			result.scale = l_int32_const(lc, node->scale);
			result.dscale = l_int32_const(lc, node->dscale);
			result.isnull = bool_const(lc, false);
			return result;
		case VECTOR_JIT_NUMERIC_ADD:
		case VECTOR_JIT_NUMERIC_SUB:
			{
				JitResult left = emit_node(b, mod, node->left, row, gen);
				JitResult right = emit_node(b, mod, node->right, row, gen);
				LLVMValueRef smax = LLVMBuildSelect(b,
					LLVMBuildICmp(b, LLVMIntUGT, left.scale, right.scale, ""),
					left.scale, right.scale, "smax");
				LLVMValueRef dres = LLVMBuildSelect(b,
					LLVMBuildICmp(b, LLVMIntUGT, left.dscale, right.dscale, ""),
					left.dscale, right.dscale, "dres");
				LLVMValueRef ma = LLVMBuildAlloca(b, i64, "ma");
				LLVMValueRef mb = LLVMBuildAlloca(b, i64, "mb");
				LLVMValueRef ok1,
					ok2,
					sum,
					ov;
				LLVMTypeRef scale_params[3] = {l_ptr(i64), i32, i32};
				LLVMTypeRef scale_type = LLVMFunctionType(LLVMInt1TypeInContext(lc),
					scale_params, 3, false);
				LLVMValueRef scale_fn = l_ptr_const((void *) numeric_scale_fast,
					l_ptr(scale_type));
				LLVMValueRef round_fn = l_ptr_const((void *) numeric_round_scale_fast,
					l_ptr(scale_type));
				const char *intrinsic = node->kind == VECTOR_JIT_NUMERIC_ADD ?
					"llvm.sadd.with.overflow.i64" : "llvm.ssub.with.overflow.i64";
				LLVMTypeRef overload[1] = {i64};
				LLVMValueRef fn = LLVMGetIntrinsicDeclaration(mod,
					LLVMLookupIntrinsicID(intrinsic, strlen(intrinsic)), overload, 1);
				LLVMValueRef args[2] = {ma, mb};
				LLVMValueRef packed;
				LLVMValueRef round_args[3];

				LLVMBuildStore(b, left.value, ma);
				LLVMBuildStore(b, right.value, mb);
				{
					LLVMValueRef a1[3] = {ma, left.scale, smax};
					LLVMValueRef a2[3] = {mb, right.scale, smax};

					ok1 = l_call(b, scale_type, scale_fn, a1, 3, "");
					ok2 = l_call(b, scale_type, scale_fn, a2, 3, "");
				}
				packed = l_call(b, LLVMGetFunctionType(fn), fn, args, 2, "");
				sum = LLVMBuildExtractValue(b, packed, 0, "");
				ov = LLVMBuildExtractValue(b, packed, 1, "");
				round_args[0] = ma;
				round_args[1] = smax;
				round_args[2] = dres;
				LLVMBuildStore(b, sum, ma);
				result.value = l_load(b, i64, ma, "sum");
				result.scale = smax;
				result.dscale = dres;
				result.isnull = LLVMBuildOr(b, left.isnull, right.isnull, "");
				result.overflow = LLVMBuildOr(b, left.overflow, right.overflow, "");
				result.overflow = LLVMBuildOr(b, result.overflow, ov, "");
				result.overflow = LLVMBuildOr(b, result.overflow,
					LLVMBuildNot(b, ok1, ""), "");
				result.overflow = LLVMBuildOr(b, result.overflow,
					LLVMBuildNot(b, ok2, ""), "");
				result.overflow = LLVMBuildOr(b, result.overflow,
					LLVMBuildNot(b, l_call(b, scale_type, round_fn, round_args, 3, ""), ""), "");
				return result;
			}
		case VECTOR_JIT_NUMERIC_MUL:
			{
				JitResult left = emit_node(b, mod, node->left, row, gen);
				JitResult right = emit_node(b, mod, node->right, row, gen);
				LLVMTypeRef i128 = LLVMIntTypeInContext(lc, 128);
				LLVMValueRef left128 = LLVMBuildSExt(b, left.value, i128, "");
				LLVMValueRef right128 = LLVMBuildSExt(b, right.value, i128, "");
				LLVMValueRef prod = LLVMBuildMul(b, left128, right128, "prod");
				LLVMValueRef max128 = l_int64_const(lc, PG_INT64_MAX);
				LLVMValueRef min128 = l_int64_const(lc, PG_INT64_MIN);
				LLVMValueRef dres = LLVMBuildAdd(b, left.dscale, right.dscale, "dres");
				LLVMValueRef scale = LLVMBuildAdd(b, left.scale, right.scale, "scale");
				LLVMValueRef ov = LLVMBuildOr(b,
					LLVMBuildICmp(b, LLVMIntSGT, prod, LLVMBuildSExt(b, max128, i128, ""), ""),
					LLVMBuildICmp(b, LLVMIntSLT, prod, LLVMBuildSExt(b, min128, i128, ""), ""), "");
				LLVMValueRef display_ov = LLVMBuildICmp(b, LLVMIntUGT, dres,
					l_int32_const(lc, NUMERIC_MAX_DISPLAY_SCALE), "");
				LLVMValueRef mant_ptr = LLVMBuildAlloca(b, i64, "mant");
				LLVMTypeRef scale_params[3] = {l_ptr(i64), i32, i32};
				LLVMTypeRef scale_type = LLVMFunctionType(LLVMInt1TypeInContext(lc),
					scale_params, 3, false);
				LLVMValueRef round_fn = l_ptr_const((void *) numeric_round_scale_fast,
					l_ptr(scale_type));
				LLVMValueRef round_args[3] = {mant_ptr, scale, dres};

				max128 = LLVMBuildSExt(b, max128, i128, "");
				min128 = LLVMBuildSExt(b, min128, i128, "");
				ov = LLVMBuildOr(b,
					LLVMBuildICmp(b, LLVMIntSGT, prod, max128, ""),
					LLVMBuildICmp(b, LLVMIntSLT, prod, min128, ""), "");
				LLVMBuildStore(b, LLVMBuildTrunc(b, prod, i64, ""), mant_ptr);
				result.value = l_load(b, i64, mant_ptr, "mant");
				result.scale = scale;
				result.dscale = dres;
				result.isnull = LLVMBuildOr(b, left.isnull, right.isnull, "");
				result.overflow = LLVMBuildOr(b, left.overflow, right.overflow, "");
				result.overflow = LLVMBuildOr(b, result.overflow, ov, "");
				result.overflow = LLVMBuildOr(b, result.overflow, display_ov, "");
				result.overflow = LLVMBuildOr(b, result.overflow,
					LLVMBuildNot(b, l_call(b, scale_type, round_fn, round_args, 3, ""), ""), "");
				return result;
			}
	}
	pg_unreachable();
}

static void
emit_result_store(LLVMBuilderRef b, LLVMValueRef out_values,
				  LLVMValueRef out_nulls, LLVMValueRef row,
				  JitResult result, bool deferred, LLVMValueRef expr)
{
	LLVMContextRef lc = LLVMGetTypeContext(LLVMTypeOf(row));
	LLVMTypeRef i64 = LLVMInt64TypeInContext(lc);
	LLVMValueRef valueptr = l_gep(b, i64, out_values, &row, 1, "");
	LLVMValueRef value64 = deferred ? l_int64_const(lc, 0) :
		LLVMBuildSExt(b, result.value, i64, "");
	LLVMValueRef wordidx = LLVMBuildLShr(b, row, l_int32_const(lc, 6), "");
	LLVMValueRef wordptr = l_gep(b, i64, out_nulls, &wordidx, 1, "");
	LLVMValueRef word = l_load(b, i64, wordptr, "");
	LLVMValueRef bit = LLVMBuildShl(b, l_int64_const(lc, 1),
								   LLVMBuildAnd(b, row, l_int32_const(lc, 63), ""), "");
	LLVMValueRef set = LLVMBuildOr(b, word, bit, "");
	LLVMValueRef clear = LLVMBuildAnd(b, word,
									 LLVMBuildNot(b, bit, ""), "");
	LLVMValueRef updated = LLVMBuildSelect(b, result.isnull, set, clear, "");

	LLVMBuildStore(b, value64, valueptr);
	LLVMBuildStore(b, updated, wordptr);
	if (deferred)
	{
		LLVMTypeRef params[5] = {l_ptr(LLVMInt8TypeInContext(lc)),
			LLVMInt32TypeInContext(lc), i64, LLVMInt32TypeInContext(lc),
			LLVMInt1TypeInContext(lc)};
		LLVMTypeRef fn_type = LLVMFunctionType(LLVMVoidTypeInContext(lc),
			params, 5, false);
		LLVMValueRef fn = l_ptr_const((void *) VectorExprStoreDeferred,
			l_ptr(fn_type));
		LLVMValueRef args[5] = {expr, row, result.value, result.dscale,
			result.isnull};

		l_call(b, fn_type, fn, args, 5, "");
	}
}

void *
llvm_compile_vector_expr(struct VectorExprState *state)
{
	EState	   *estate = VectorExprJitEState(state);
	VectorJitNode *tree = VectorExprJitGetTree(state);
	LLVMJitContext *context;
	LLVMModuleRef mod;
	LLVMContextRef lc;
	LLVMBuilderRef b;
	LLVMBasicBlockRef entry,
		loop,
		fallback,
		store,
		next,
		exit;
	LLVMValueRef eval_fn;
	LLVMValueRef rt;
	LLVMValueRef selection;
	LLVMValueRef nselected;
	LLVMValueRef gen;
	LLVMValueRef out_values;
	LLVMValueRef out_nulls;
	LLVMValueRef expr;
	LLVMValueRef batch;
	LLVMValueRef index;
	LLVMValueRef zero;
	LLVMValueRef row;
	LLVMValueRef nextindex;
	LLVMValueRef cond;
	LLVMValueRef fallback_fn;
	LLVMTypeRef param_types[1];
	LLVMTypeRef fallback_params[3];
	LLVMTypeRef fallback_type;
	JitResult result;
	bool		deferred = tree && tree->kind >= VECTOR_JIT_NUMERIC_VAR;
	char	   *funcname;

	if (tree == NULL || estate == NULL)
		return NULL;

	llvm_enter_fatal_on_oom();
	if (estate->es_jit)
		context = (LLVMJitContext *) estate->es_jit;
	else
	{
		context = llvm_create_context(estate->es_jit_flags);
		estate->es_jit = &context->base;
	}

	mod = llvm_mutable_module(context);
	lc = LLVMGetModuleContext(mod);
	b = LLVMCreateBuilderInContext(lc);
	funcname = llvm_expand_funcname(context, deferred ? "vector_numeric" : "vector_int4");
	param_types[0] = l_ptr(LLVMInt8TypeInContext(lc));
	eval_fn = LLVMAddFunction(mod, funcname,
							 LLVMFunctionType(LLVMVoidTypeInContext(lc), param_types, 1, false));
	LLVMSetLinkage(eval_fn, LLVMExternalLinkage);
	LLVMSetVisibility(eval_fn, LLVMDefaultVisibility);
	llvm_copy_attributes(AttributeTemplate, eval_fn);

	entry = LLVMAppendBasicBlockInContext(lc, eval_fn, "entry");
	loop = LLVMAppendBasicBlockInContext(lc, eval_fn, "loop");
	fallback = LLVMAppendBasicBlockInContext(lc, eval_fn, "fallback");
	store = LLVMAppendBasicBlockInContext(lc, eval_fn, "store");
	next = LLVMAppendBasicBlockInContext(lc, eval_fn, "next");
	exit = LLVMAppendBasicBlockInContext(lc, eval_fn, "exit");
	rt = LLVMGetParam(eval_fn, 0);

	LLVMPositionBuilderAtEnd(b, entry);
	selection = rt_load(b, rt, l_ptr(LLVMInt32TypeInContext(lc)),
						offsetof(VectorExprJitRuntime, selection), "selection");
	nselected = rt_load(b, rt, LLVMInt32TypeInContext(lc),
						offsetof(VectorExprJitRuntime, nselected), "nselected");
	gen = rt_load(b, rt, LLVMInt32TypeInContext(lc),
					offsetof(VectorExprJitRuntime, gen), "gen");
	out_values = rt_load(b, rt, l_ptr(LLVMInt64TypeInContext(lc)),
						 offsetof(VectorExprJitRuntime, out_values), "out_values");
	out_nulls = rt_load(b, rt, l_ptr(LLVMInt64TypeInContext(lc)),
						offsetof(VectorExprJitRuntime, out_nulls), "out_nulls");
	expr = rt_load(b, rt, l_ptr(LLVMInt8TypeInContext(lc)),
				   offsetof(VectorExprJitRuntime, expr), "expr");
	batch = rt_load(b, rt, l_ptr(LLVMInt8TypeInContext(lc)),
					offsetof(VectorExprJitRuntime, batch), "batch");
	LLVMBuildBr(b, loop);

	LLVMPositionBuilderAtEnd(b, loop);
	index = LLVMBuildPhi(b, LLVMInt32TypeInContext(lc), "index");
	zero = l_int32_const(lc, 0);
	LLVMAddIncoming(index, &zero, &entry, 1);
	row = l_load(b, LLVMInt32TypeInContext(lc),
				 l_gep(b, LLVMInt32TypeInContext(lc), selection, &index, 1, ""), "row");
	result = emit_node(b, mod, tree, row, gen);
	if (deferred)
	{
		LLVMValueRef too_large = LLVMBuildOr(b,
			LLVMBuildICmp(b, LLVMIntSGT, result.value,
				l_int64_const(lc, PG_INT64_MAX / 1000), ""),
			LLVMBuildICmp(b, LLVMIntSLT, result.value,
				l_int64_const(lc, PG_INT64_MIN / 1000), ""), "");

		result.overflow = LLVMBuildOr(b, result.overflow, too_large, "");
	}
	cond = LLVMBuildICmp(b, LLVMIntEQ, result.isnull, bool_const(lc, false), "");
	cond = LLVMBuildAnd(b, result.overflow, cond, "need_fallback");
	LLVMBuildCondBr(b, cond, fallback, store);

	LLVMPositionBuilderAtEnd(b, fallback);
	fallback_params[0] = l_ptr(LLVMInt8TypeInContext(lc));
	fallback_params[1] = l_ptr(LLVMInt8TypeInContext(lc));
	fallback_params[2] = LLVMInt32TypeInContext(lc);
	fallback_type = LLVMFunctionType(LLVMVoidTypeInContext(lc),
									 fallback_params, 3, false);
	fallback_fn = l_ptr_const((void *) VectorExprEvalRowFallback,
							 l_ptr(fallback_type));
	{
		LLVMValueRef args[3] = {expr, batch, row};

		l_call(b, fallback_type, fallback_fn, args, 3, "");
	}
	LLVMBuildBr(b, next);

	LLVMPositionBuilderAtEnd(b, store);
	emit_result_store(b, out_values, out_nulls, row, result, deferred, expr);
	LLVMBuildBr(b, next);

	LLVMPositionBuilderAtEnd(b, next);
	nextindex = LLVMBuildAdd(b, index, l_int32_const(lc, 1), "nextindex");
	LLVMAddIncoming(index, &nextindex, &next, 1);
	cond = LLVMBuildICmp(b, LLVMIntULT, nextindex, nselected, "continue");
	LLVMBuildCondBr(b, cond, loop, exit);

	LLVMPositionBuilderAtEnd(b, exit);
	LLVMBuildRetVoid(b);
	LLVMDisposeBuilder(b);

	{
		void	   *native = llvm_get_function(context, funcname);

		llvm_leave_fatal_on_oom();
		return native;
	}
}

#endif							/* USE_LLVM */
