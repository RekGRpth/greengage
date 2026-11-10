/*-------------------------------------------------------------------------
 *
 * vectorBatch.h
 *	  Column batches used by the vectorized executor.
 *
 * A batch owns fixed-size column arrays and a selection vector.  The
 * selection vector contains row indexes into those arrays, so filtering does
 * not require moving values.  Pass-by-reference values live in a separate
 * memory context which is reset before the batch is filled again.
 *
 *-------------------------------------------------------------------------
 */
#ifndef VECTORBATCH_H
#define VECTORBATCH_H

#include "access/tupdesc.h"
#include "utils/memutils.h"

#define VECTOR_BATCH_DEFAULT_CAPACITY	2048
#define VECTOR_BATCH_NULLWORDS(capacity) (((capacity) + 63) / 64)

typedef struct VectorColumn
{
	Oid			typid;
	int32		typmod;
	int16		typlen;
	bool		typbyval;
	Datum	   *values;
	uint64	   *nulls;

	/*
	 * Decoded numeric cache: every step reading this column decodes each
	 * row once per batch.  cache_gen is the batch generation the entries
	 * were decoded in; per-row bits in cache_valid say which rows hold one.
	 */
	uint32		cache_gen;
	int			cache_nwords;
	uint64    *cache_valid;
	int64	   *cache_mant;
	int16	   *cache_scale;
	int16	   *cache_dscale;
} VectorColumn;

typedef struct VectorBatch
{
	int			ncols;
	int			capacity;
	int			nrows;
	int			nselected;
	uint32		gen;		/* batch generation, bumped on every reset */
	VectorColumn *columns;
	int32	   *selection;
	TupleDesc	tupdesc;
	MemoryContext varlena_context;
} VectorBatch;

extern VectorBatch *VectorBatchCreate(MemoryContext parent,
									  TupleDesc tupdesc, int capacity);
extern void VectorBatchDestroy(VectorBatch *batch);
extern void VectorBatchReset(VectorBatch *batch);
extern void VectorBatchSetRowCount(VectorBatch *batch, int nrows);
extern void VectorBatchSetValue(VectorBatch *batch, int col, int row,
								Datum value, bool isnull);
extern Datum VectorBatchCopyDatum(VectorBatch *batch, int col, Datum value,
								  bool isnull);
extern void VectorBatchCopyColumns(VectorBatch *dst, const VectorBatch *src,
								   const int *dstcol);

static inline bool
VectorBatchIsNull(const VectorBatch *batch, int col, int row)
{
	const VectorColumn *column = &batch->columns[col];

	return (column->nulls[row / 64] & ((uint64) 1 << (row % 64))) != 0;
}

static inline int
VectorBatchSelectedRow(const VectorBatch *batch, int position)
{
	Assert(position >= 0 && position < batch->nselected);
	return batch->selection[position];
}

#endif							/* VECTORBATCH_H */
