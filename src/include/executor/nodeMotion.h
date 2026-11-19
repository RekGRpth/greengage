/*-------------------------------------------------------------------------
 *
 * nodeMotion.h
 *
 * Portions Copyright (c) 2012-Present VMware, Inc. or its affiliates.
 * Portions Copyright (c) 1996-2004, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 *
 * IDENTIFICATION
 *	    src/include/executor/nodeMotion.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef NODEMOTION_H
#define NODEMOTION_H

#include "nodes/execnodes.h"

extern MotionState *ExecInitMotion(Motion *node, EState *estate, int eflags);
extern void ExecEndMotion(MotionState *node);

/* Batch interconnect: receive the next columnar batch from an unsorted
 * motion, or NULL (*tuples_only marks a tuple stream the caller must
 * read row-wise). */
struct VectorBatch;
extern struct VectorBatch *MotionStateRecvBatch(MotionState *node,
												bool *tuples_only);
extern void ExecReScanMotion(MotionState *node);

extern void ExecSquelchMotion(MotionState *node);

#endif   /* NODEMOTION_H */
