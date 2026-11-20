/*-------------------------------------------------------------------------
 *
 * vectorBatch.c
 *	  Memory management and primitive operations for column batches.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "executor/vectorBatch.h"
#include "catalog/pg_type_d.h"
#include "utils/datum.h"

VectorBatch *
VectorBatchCreate(MemoryContext parent, TupleDesc tupdesc, int capacity,
				  bool packed_i32)
{
	VectorBatch *batch;
	MemoryContext oldcontext;
	int			col;

	Assert(parent != NULL);
	Assert(tupdesc != NULL);
	Assert(capacity > 0);

	oldcontext = MemoryContextSwitchTo(parent);

	batch = palloc0(sizeof(VectorBatch));
	batch->ncols = tupdesc->natts;
	batch->capacity = capacity;
	batch->packed_i32 = packed_i32;
	batch->columns = palloc0(sizeof(VectorColumn) * Max(batch->ncols, 1));
	batch->selection = palloc(sizeof(int32) * capacity);
	batch->tupdesc = tupdesc;
	batch->varlena_context = AllocSetContextCreate(parent,
											 "Vector Batch Values",
											 ALLOCSET_DEFAULT_MINSIZE,
											 ALLOCSET_DEFAULT_INITSIZE,
											 ALLOCSET_DEFAULT_MAXSIZE);

	for (col = 0; col < batch->ncols; col++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, col);
		VectorColumn *column = &batch->columns[col];

		column->typid = attr->atttypid;
		column->typmod = attr->atttypmod;
		column->typlen = attr->attlen;
		column->typbyval = attr->attbyval;
		if (packed_i32 && column->typid == INT2OID)
			column->i16_values = palloc0(sizeof(int16) * capacity);
		else if (packed_i32 &&
			(column->typid == INT4OID || column->typid == DATEOID ||
			 column->typid == FLOAT4OID))
			column->i32_values = palloc0(sizeof(int32) * capacity);
		else
			column->values = palloc0(sizeof(Datum) * capacity);
		column->nulls = palloc0(sizeof(uint64) * VECTOR_BATCH_NULLWORDS(capacity));
		if (column->typid == NUMERICOID)
		{
			column->cache_gen = 0;
			column->cache_nwords = VECTOR_BATCH_NULLWORDS(capacity);
			column->cache_valid = palloc0(sizeof(uint64) *
										  VECTOR_BATCH_NULLWORDS(capacity));
			column->cache_mant = palloc(sizeof(int64) * capacity);
			column->cache_scale = palloc(sizeof(int16) * capacity);
			column->cache_dscale = palloc(sizeof(int16) * capacity);
		}
	}

	MemoryContextSwitchTo(oldcontext);
	return batch;
}

void
VectorBatchDestroy(VectorBatch *batch)
{
	int			col;

	if (batch == NULL)
		return;

	MemoryContextDelete(batch->varlena_context);
	for (col = 0; col < batch->ncols; col++)
	{
		if (batch->columns[col].i16_values != NULL)
			pfree(batch->columns[col].i16_values);
		else if (batch->columns[col].i32_values != NULL)
			pfree(batch->columns[col].i32_values);
		else
			pfree(batch->columns[col].values);
		pfree(batch->columns[col].nulls);
		if (batch->columns[col].cache_mant != NULL)
		{
			pfree(batch->columns[col].cache_valid);
			pfree(batch->columns[col].cache_mant);
			pfree(batch->columns[col].cache_scale);
			pfree(batch->columns[col].cache_dscale);
		}
	}
	pfree(batch->columns);
	pfree(batch->selection);
	pfree(batch);
}

void
VectorBatchReset(VectorBatch *batch)
{
	int			col;

	batch->nrows = 0;
	batch->nselected = 0;
	batch->sel_identity = true; /* empty selection is trivially identity */
	batch->gen++;
	MemoryContextReset(batch->varlena_context);
	for (col = 0; col < batch->ncols; col++)
		memset(batch->columns[col].nulls, 0,
			   sizeof(uint64) * VECTOR_BATCH_NULLWORDS(batch->capacity));
}

/* Publish nrows values in each column and select all of them. */
void
VectorBatchSetRowCount(VectorBatch *batch, int nrows)
{
	int			row;

	Assert(nrows >= 0 && nrows <= batch->capacity);

	batch->nrows = nrows;
	batch->nselected = nrows;
	batch->sel_identity = true;
	for (row = 0; row < nrows; row++)
		batch->selection[row] = row;
}

void
VectorBatchSetValue(VectorBatch *batch, int col, int row,
					Datum value, bool isnull)
{
	VectorColumn *column;

	Assert(col >= 0 && col < batch->ncols);
	Assert(row >= 0 && row < batch->capacity);

	column = &batch->columns[col];
	VectorColumnSetDatum(column, row, value, isnull);
}

/* Copy a value into storage that survives until the next batch reset. */
Datum
VectorBatchCopyDatum(VectorBatch *batch, int col, Datum value, bool isnull)
{
	VectorColumn *column;
	MemoryContext oldcontext;
	Datum		result;

	Assert(col >= 0 && col < batch->ncols);

	if (isnull)
		return (Datum) 0;

	column = &batch->columns[col];
	if (column->typbyval)
		return value;

	oldcontext = MemoryContextSwitchTo(batch->varlena_context);
	result = datumCopy(value, column->typbyval, column->typlen);
	MemoryContextSwitchTo(oldcontext);
	return result;
}

/*
 * Copy every column of src into dst: dstcol[j] names the destination
 * column of src column j, or -1 to skip it.  By-ref values are stolen
 * (not copied): they live until src is reset, which the caller may do
 * only after dst has been consumed.  The row count and selection vector
 * move with the values; dst must have been reset by the caller.
 */
void
VectorBatchCopyColumns(VectorBatch *dst, const VectorBatch *src,
					   const int *dstcol)
{
	int			j;

	for (j = 0; j < src->ncols; j++)
	{
		int			d = dstcol[j];
		const VectorColumn *sc = &src->columns[j];
		VectorColumn *dc;

		if (d < 0)
			continue;
		dc = &dst->columns[d];
		if (dc->i16_values != NULL)
			memcpy(dc->i16_values, sc->i16_values, sizeof(int16) * src->nrows);
		else if (dc->i32_values != NULL)
			memcpy(dc->i32_values, sc->i32_values, sizeof(int32) * src->nrows);
		else
			memcpy(dc->values, sc->values, sizeof(Datum) * src->nrows);
		memcpy(dc->nulls, sc->nulls,
			   sizeof(uint64) * VECTOR_BATCH_NULLWORDS(src->capacity));
	}
	memcpy(dst->selection, src->selection, sizeof(int32) * src->nrows);
	dst->nrows = src->nrows;
	dst->nselected = src->nselected;
}

/*
 * Batch interconnect codec.
 *
 * Payload layout (little-endian, self-describing per chunk sequence):
 *   u32  magic 'VBAT'
 *   u16  nrows
 *   u16  ncols
 *   per column:
 *     u8   kind: 0 = fixed-width by-value, 1 = by-reference blob
 *     u8   pad
 *     u16  typlen (of the column type, for kind 0)
 *     u8   null bitmap, ceil(nrows/8) bytes, padded up to a multiple of 8
 *     kind 0: nrows * typlen raw datum bits
 *     kind 1: u32 blob length, u32 offsets[nrows] (0xFFFFFFFF = NULL),
 *             blob bytes
 *
 * Encode takes an explicit column count (the motion output may be a
 * prefix of the batch columns) and an optional row list so a
 * hash-redistribute sender can partition one batch into per-route
 * payloads without materializing intermediate batches.  The varlena
 * layout is all offsets first, then the blob.
 */
#define VBATCH_MAGIC	0x54414256	/* 'VBAT' */

static void
vbatch_encode_u32(StringInfo buf, uint32 v)
{
	appendBinaryStringInfo(buf, (char *) &v, sizeof(v));
}

static void
vbatch_encode_u16(StringInfo buf, uint16 v)
{
	appendBinaryStringInfo(buf, (char *) &v, sizeof(v));
}

void
VectorBatchEncodeRows(StringInfo buf, const VectorBatch *batch,
					  int ncols, const int *rows, int nrows)
{
	int			col;
	int			row;

	Assert(ncols <= batch->ncols);
	vbatch_encode_u32(buf, VBATCH_MAGIC);
	vbatch_encode_u16(buf, (uint16) nrows);
	vbatch_encode_u16(buf, (uint16) ncols);

	for (col = 0; col < ncols; col++)
	{
		const VectorColumn *c = &batch->columns[col];
		bool		byval = c->typbyval && c->typlen > 0;
		uint8		kind = byval ? 0 : 1;
		int			bitbytes = (nrows + 7) / 8;
		uint8		zeros[8] = {0};

		appendBinaryStringInfo(buf, (char *) &kind, 1);
		appendStringInfoCharMacro(buf, 0);
		vbatch_encode_u16(buf, (uint16) (byval ? c->typlen : -1));

		/* null bitmap */
		for (row = 0; row < bitbytes; row++)
		{
			uint8		b = 0;
			int			i;

			for (i = 0; i < 8; i++)
			{
				int			r = row * 8 + i;

				if (r < nrows &&
					VectorBatchIsNull(batch, col,
									  rows ? rows[r] : VectorBatchSelectedRow(batch, r)))
					b |= (uint8) (1 << i);
			}
			appendBinaryStringInfo(buf, (char *) &b, 1);
		}
		if (bitbytes % 8 != 0)
			appendBinaryStringInfo(buf, (char *) zeros, 8 - bitbytes % 8);

		if (byval)
		{
			for (row = 0; row < nrows; row++)
			{
				int			r = rows ? rows[row] : VectorBatchSelectedRow(batch, row);

				if (VectorBatchIsNull(batch, col, r))
					appendBinaryStringInfo(buf, (char *) zeros, c->typlen);
				else
				{
					Datum		v = VectorBatchGetDatum(batch, col, r);

					appendBinaryStringInfo(buf, (char *) &v, c->typlen);
				}
			}
		}
		else
		{
			/*
			 * Two passes: all offsets first, then the blob, matching the
			 * decoder's layout [bloblen][offsets n][blob bytes].
			 */
			uint32		bloblen = 0;
			uint32	   *offs;

			offs = (uint32 *) palloc(sizeof(uint32) * Max(nrows, 1));
			for (row = 0; row < nrows; row++)
			{
				int			r = rows ? rows[row] : VectorBatchSelectedRow(batch, row);

				if (VectorBatchIsNull(batch, col, r))
					offs[row] = 0xFFFFFFFF;
				else
				{
					Datum		v = VectorBatchGetDatum(batch, col, r);
					Size		sz;

					if (c->typlen == -2)
						sz = strlen((char *) DatumGetPointer(v)) + 1;
					else
					{
						/*
						 * Flatten expanded objects; the flattened copy is
						 * palloc'd in the caller's short-lived context.
						 */
						if (VARATT_IS_EXTERNAL_EXPANDED(DatumGetPointer(v)))
							v = datumCopy(v, false, -1);
						sz = VARSIZE_ANY(DatumGetPointer(v));
					}
					offs[row] = bloblen;
					bloblen += (uint32) sz;
				}
			}

			vbatch_encode_u32(buf, bloblen);
			for (row = 0; row < nrows; row++)
				vbatch_encode_u32(buf, offs[row]);
			for (row = 0; row < nrows; row++)
			{
				if (offs[row] != 0xFFFFFFFF)
				{
					int			r = rows ? rows[row] : VectorBatchSelectedRow(batch, row);
					Datum		v = VectorBatchGetDatum(batch, col, r);

					if (c->typlen != -2 &&
						VARATT_IS_EXTERNAL_EXPANDED(DatumGetPointer(v)))
						v = datumCopy(v, false, -1);
					appendBinaryStringInfo(buf, (char *) DatumGetPointer(v),
										   c->typlen == -2 ?
										   strlen((char *) DatumGetPointer(v)) + 1 :
										   VARSIZE_ANY(DatumGetPointer(v)));
				}
			}
			pfree(offs);
		}
	}
}

static uint32
vbatch_decode_u32(const char **p, const char *end)
{
	uint32		v;

	if (*p + sizeof(v) > end)
		elog(ERROR, "malformed vector batch chunk");
	memcpy(&v, *p, sizeof(v));
	*p += sizeof(v);
	return v;
}

static uint16
vbatch_decode_u16(const char **p, const char *end)
{
	uint16		v;

	if (*p + sizeof(v) > end)
		elog(ERROR, "malformed vector batch chunk");
	memcpy(&v, *p, sizeof(v));
	*p += sizeof(v);
	return v;
}

VectorBatch *
VectorBatchDecodeRows(MemoryContext parent, TupleDesc tupdesc,
					  const char *payload, int len)
{
	const char *p = payload;
	const char *end = payload + len;
	VectorBatch *batch;
	uint32		magic;
	int			nrows;
	int			ncols;
	int			col;

	if (len < 8)
		elog(ERROR, "malformed vector batch chunk");
	magic = vbatch_decode_u32(&p, end);
	if (magic != VBATCH_MAGIC)
		elog(ERROR, "bad vector batch magic 0x%x", magic);
	nrows = vbatch_decode_u16(&p, end);
	ncols = vbatch_decode_u16(&p, end);
	if (ncols != tupdesc->natts)
		elog(ERROR, "vector batch column count %d does not match motion output %d",
			 ncols, tupdesc->natts);

	batch = VectorBatchCreate(parent, tupdesc, Max(nrows, 1), false);

	for (col = 0; col < ncols; col++)
	{
		VectorColumn *c = &batch->columns[col];
		uint8		kind;
		int16		typlen;
		int			bitbytes;
		int			row;
		uint8		b = 0;

		if (p + 4 > end)
			elog(ERROR, "malformed vector batch chunk");
		kind = (uint8) *p++;
		p++;					/* pad */
		typlen = (int16) vbatch_decode_u16(&p, end);

		bitbytes = (nrows + 7) / 8;
		for (row = 0; row < nrows; row++)
		{
			if (row % 8 == 0)
			{
				if (p >= end)
					elog(ERROR, "malformed vector batch chunk");
				b = (uint8) *p++;
			}
			if (b & (1 << (row % 8)))
				c->nulls[row / 64] |= ((uint64) 1 << (row % 64));
		}
		if (bitbytes % 8 != 0)
			p += 8 - bitbytes % 8;
		if (p > end)
			elog(ERROR, "malformed vector batch chunk");

		if (kind == 0)
		{
			if (p + (size_t) nrows * typlen > end)
				elog(ERROR, "malformed vector batch chunk");
			for (row = 0; row < nrows; row++)
			{
				Datum		v = 0;

				memcpy((char *) &v, p, typlen);
				p += typlen;
				c->values[row] = v;
			}
		}
		else
		{
			uint32		bloblen = vbatch_decode_u32(&p, end);
			const char *offsets = p;
			const char *blob;
			MemoryContext oldcxt;

			p += (size_t) nrows * sizeof(uint32);
			if (p + bloblen > end)
				elog(ERROR, "malformed vector batch chunk");
			blob = p;
			p += bloblen;

			oldcxt = MemoryContextSwitchTo(batch->varlena_context);
			for (row = 0; row < nrows; row++)
			{
				uint32		off;

				memcpy(&off, offsets + (size_t) row * sizeof(uint32), sizeof(off));
				if (off != 0xFFFFFFFF)
				{
					Size		sz;
					char	   *dst;

					if (off >= bloblen)
						elog(ERROR, "corrupt vector batch chunk: offset %u beyond blob %u (col %d row %d typlen %d)",
							 off, bloblen, col, row, typlen);
					if (typlen == -2)
					{
						Size		next;

						if (row + 1 < nrows)
						{
							int			r2;

							/*
							 * size = up to the next non-null offset or the
							 * blob end
							 */
							next = bloblen;
							for (r2 = row + 1; r2 < nrows; r2++)
							{
								uint32		o2;

								memcpy(&o2, offsets + (size_t) r2 * sizeof(uint32), sizeof(o2));
								if (o2 != 0xFFFFFFFF)
								{
									if (o2 > off)
										next = o2;
									break;
								}
							}
						}
						else
							next = bloblen;
						sz = next - off;	/* includes NUL */
					}
					else
						sz = VARSIZE_ANY(blob + off);
					if (sz == 0 || sz > bloblen - off || sz > 0x40000000)
						elog(ERROR, "corrupt vector batch chunk: bad size %lu at offset %u of %u (col %d row %d typlen %d)",
							 (unsigned long) sz, off, bloblen, col, row, typlen);
					dst = (char *) palloc(sz);
					memcpy(dst, blob + off, sz);
					c->values[row] = PointerGetDatum(dst);
				}
			}
			MemoryContextSwitchTo(oldcxt);
		}
	}

	{
		int			row;

		batch->nrows = nrows;
		batch->nselected = nrows;
		batch->sel_identity = true;
		for (row = 0; row < nrows; row++)
			batch->selection[row] = row;
	}
	return batch;
}
