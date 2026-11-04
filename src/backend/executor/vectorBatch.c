/*-------------------------------------------------------------------------
 *
 * vectorBatch.c
 *	  Memory management and primitive operations for column batches.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "executor/vectorBatch.h"
#include "utils/datum.h"

VectorBatch *
VectorBatchCreate(MemoryContext parent, TupleDesc tupdesc, int capacity)
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
	batch->columns = palloc0(sizeof(VectorColumn) * Max(batch->ncols, 1));
	batch->selection = palloc(sizeof(int32) * capacity);
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
		column->values = palloc0(sizeof(Datum) * capacity);
		column->nulls = palloc0(sizeof(uint64) * VECTOR_BATCH_NULLWORDS(capacity));
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
		pfree(batch->columns[col].values);
		pfree(batch->columns[col].nulls);
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
	for (row = 0; row < nrows; row++)
		batch->selection[row] = row;
}

void
VectorBatchSetValue(VectorBatch *batch, int col, int row,
					Datum value, bool isnull)
{
	VectorColumn *column;
	uint64		mask;

	Assert(col >= 0 && col < batch->ncols);
	Assert(row >= 0 && row < batch->capacity);

	column = &batch->columns[col];
	column->values[row] = value;
	mask = (uint64) 1 << (row % 64);
	if (isnull)
		column->nulls[row / 64] |= mask;
	else
		column->nulls[row / 64] &= ~mask;
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
