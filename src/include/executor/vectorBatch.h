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
#include "lib/stringinfo.h"
#include "catalog/pg_type_d.h"
#include "utils/datum.h"
#include "utils/memutils.h"
#include "utils/float.h"

#define VECTOR_BATCH_DEFAULT_CAPACITY	2048
#define VECTOR_BATCH_NULLWORDS(capacity) (((capacity) + 63) / 64)

typedef struct VectorColumn
{
	Oid			typid;
	int32		typmod;
	int16		typlen;
	bool		typbyval;
	Datum	   *values;
	int32	   *i32_values;	/* packed storage for INT4/DATE/FLOAT4 */
	int16	   *i16_values;	/* packed storage for INT2 */
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
	bool		packed_i32;
	TupleDesc	tupdesc;
	MemoryContext varlena_context;
	/*
	 * Known-identity selection: selection[i] == i for all i < nselected.
	 * Set by every batch fill, cleared by compaction; lets dense qual
	 * kernels address column arrays contiguously.
	 */
	bool		sel_identity;
	/*
	 * Zero-copy motion payload: a decoded batch's by-reference values
	 * point into this palloc'd wire buffer, so VectorBatchDestroy frees
	 * it.  NULL for every batch that is not a motion decode.
	 */
	char	   *wire_data;
	/*
	 * Motion-layer recycling: destroying a decoded batch hands the shell
	 * (columns, selection, varlena context) and its payload buffer back
	 * to the owning motion node through this hook instead of pfreeing
	 * them part by part.  Destroy is the release point of the batch's
	 * lifetime contract, so nothing can reference the storage once it is
	 * pooled.  NULL for non-motion batches.
	 */
	void	   *wire_owner;
	void		(*wire_recycle) (void *owner, struct VectorBatch *batch);
} VectorBatch;

extern VectorBatch *VectorBatchCreate(MemoryContext parent,
									  TupleDesc tupdesc, int capacity,
									  bool packed_i32);
/*
 * Batch interconnect codec: a columnar payload riding one
 * START..MID*..END chunk sequence of a motion stream.  Encode takes an
 * optional row-index list (for hash-redistribute partitioning); decode
 * builds a plain (non-packed) batch that owns the payload buffer: its
 * by-reference values point into the caller-supplied palloc'd payload
 * (offsets MAXALIGNed by the encoder), and VectorBatchDestroy frees it.
 */
extern void VectorBatchEncodeRows(StringInfo buf, const VectorBatch *batch,
								  int ncols, const int *rows, int nrows);
extern VectorBatch *VectorBatchDecodeRows(MemoryContext parent,
										  TupleDesc tupdesc,
										  char *payload, int len);
/*
 * Decode into an existing batch shell (a recycled motion batch):
 * grows the fixed arrays when the payload's nrows exceed the shell's
 * capacity; takes ownership of the payload just like
 * VectorBatchDecodeRows.
 */
extern void VectorBatchDecodeRowsInto(VectorBatch *batch,
									  char *payload, int len);
extern void VectorBatchDestroy(VectorBatch *batch);
extern void VectorBatchReset(VectorBatch *batch);
extern void VectorBatchSetRowCount(VectorBatch *batch, int nrows);
extern void VectorBatchSetValue(VectorBatch *batch, int col, int row,
								Datum value, bool isnull);
extern Datum VectorBatchCopyDatum(VectorBatch *batch, int col, Datum value,
								  bool isnull);
extern void VectorBatchCopyColumns(VectorBatch *dst, const VectorBatch *src,
								   const int *dstcol);

/*
 * Copy a value into a batch assuming the caller has already switched to
 * batch->varlena_context: the fill loops hoist the context switch out of
 * their per-value calls.  Same semantics as VectorBatchCopyDatum otherwise.
 */
static inline Datum
VectorBatchCopyDatumInContext(VectorBatch *batch, int col, Datum value,
							  bool isnull)
{
	VectorColumn *column;

	if (isnull)
		return (Datum) 0;

	column = &batch->columns[col];
	if (column->typbyval)
		return value;

	return datumCopy(value, column->typbyval, column->typlen);
}

static inline Datum
VectorColumnGetDatum(const VectorColumn *column, int row)
{
	if (column->i16_values != NULL)
		return Int16GetDatum(column->i16_values[row]);
	if (column->i32_values != NULL)
	{
		if (column->typid == FLOAT4OID)
		{
			uint32		bits = (uint32) column->i32_values[row];
			float4		value;

			memcpy(&value, &bits, sizeof(value));
			return Float4GetDatum(value);
		}
		return Int32GetDatum(column->i32_values[row]);
	}
	return column->values[row];
}

static inline void
VectorColumnSetDatum(VectorColumn *column, int row, Datum value, bool isnull)
{
	uint64		mask = (uint64) 1 << (row % 64);

	if (column->i16_values != NULL)
		column->i16_values[row] = DatumGetInt16(value);
	else if (column->i32_values != NULL)
	{
		if (column->typid == FLOAT4OID)
		{
			float4		float_value = DatumGetFloat4(value);
			uint32		bits;

			memcpy(&bits, &float_value, sizeof(bits));
			column->i32_values[row] = (int32) bits;
		}
		else
			column->i32_values[row] = DatumGetInt32(value);
	}
	else
		column->values[row] = value;
	if (isnull)
		column->nulls[row / 64] |= mask;
	else
		column->nulls[row / 64] &= ~mask;
}

static inline Datum
VectorBatchGetDatum(const VectorBatch *batch, int col, int row)
{
	return VectorColumnGetDatum(&batch->columns[col], row);
}

static inline bool
VectorBatchIsNull(const VectorBatch *batch, int col, int row)
{
	const VectorColumn *column = &batch->columns[col];

	return (column->nulls[row / 64] & ((uint64) 1 << (row % 64))) != 0;
}

static inline int
VectorBatchSelectedRow(const VectorBatch *batch, int position)
{
	/*
	 * A batch published with sel_identity (VectorBatchSetRowCount) has no
	 * selection array behind it: its rows are positional.
	 */
	if (batch->sel_identity)
		return position;

	Assert(position >= 0 && position < batch->nselected);
	return batch->selection[position];
}

#endif							/* VECTORBATCH_H */
