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

typedef struct JitScalar
{
	LLVMValueRef value;
	LLVMValueRef isnull;
	LLVMValueRef overflow;
} JitScalar;

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

static JitScalar
emit_node(LLVMBuilderRef b, LLVMModuleRef mod, VectorJitNode *node,
		  LLVMValueRef row)
{
	JitScalar result;
	LLVMContextRef lc = LLVMGetModuleContext(mod);
	LLVMTypeRef i32 = LLVMInt32TypeInContext(lc);
	LLVMTypeRef i64 = LLVMInt64TypeInContext(lc);

	result.overflow = LLVMConstInt(LLVMInt1TypeInContext(LLVMGetModuleContext(mod)), 0, false);
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
			result.isnull = LLVMConstInt(LLVMInt1TypeInContext(LLVMGetModuleContext(mod)), 0, false);
			return result;
		case VECTOR_JIT_INT4_ADD:
		case VECTOR_JIT_INT4_MUL:
			{
				JitScalar left = emit_node(b, mod, node->left, row);
				JitScalar right = emit_node(b, mod, node->right, row);
				const char *intrinsic = node->kind == VECTOR_JIT_INT4_ADD ?
					"llvm.sadd.with.overflow.i32" : "llvm.smul.with.overflow.i32";
				LLVMTypeRef overload[1] = {LLVMInt32TypeInContext(lc)};
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
	}
	pg_unreachable();
}

static void
emit_result_store(LLVMBuilderRef b, LLVMValueRef out_values,
				  LLVMValueRef out_nulls, LLVMValueRef row,
				  JitScalar result)
{
	LLVMContextRef lc = LLVMGetTypeContext(LLVMTypeOf(row));
	LLVMTypeRef i64 = LLVMInt64TypeInContext(lc);
	LLVMValueRef valueptr = l_gep(b, i64, out_values, &row, 1, "");
	LLVMValueRef value64 = LLVMBuildSExt(b, result.value, i64, "");
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
	JitScalar result;
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
	funcname = llvm_expand_funcname(context, "vector_int4");
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
	result = emit_node(b, mod, tree, row);
	cond = LLVMBuildICmp(b, LLVMIntEQ, result.isnull,
						 LLVMConstInt(LLVMInt1TypeInContext(lc), 0, false), "");
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
	emit_result_store(b, out_values, out_nulls, row, result);
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
