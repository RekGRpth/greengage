/*-------------------------------------------------------------------------
 *
 * nodeLimit.h
 *
 *
 *
 * Portions Copyright (c) 1996-2019, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * src/include/executor/nodeLimit.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef NODELIMIT_H
#define NODELIMIT_H

#include "nodes/execnodes.h"

extern LimitState *ExecInitLimit(Limit *node, EState *estate, int eflags);
extern void ExecEndLimit(LimitState *node);
extern void ExecReScanLimit(LimitState *node);

typedef enum VectorLimitResult
{
	VECTOR_LIMIT_SKIP,
	VECTOR_LIMIT_EMIT,
	VECTOR_LIMIT_EOF
} VectorLimitResult;

extern bool VectorLimitPlanIsSupported(Limit *limit);
extern bool VectorLimitStart(LimitState *node);
extern VectorLimitResult VectorLimitNext(LimitState *node);

#endif							/* NODELIMIT_H */
