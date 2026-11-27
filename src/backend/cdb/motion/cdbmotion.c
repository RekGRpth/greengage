/*-------------------------------------------------------------------------
 *
 * cdbmotion.c
 *		Access into the motion-layer in order to send and receive tuples
 *		within a motion node ID within a particular process group id.
 *
 * Portions Copyright (c) 2005-2008, Greenplum inc
 * Portions Copyright (c) 2012-Present VMware, Inc. or its affiliates.
 *
 *
 * IDENTIFICATION
 *	    src/backend/cdb/motion/cdbmotion.c
 *
 * Reviewers: jzhang, tkordas
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "access/htup.h"
#include "libpq-fe.h"
#include "libpq-int.h"
#include "cdb/cdbconn.h"
#include "cdb/cdbmotion.h"
#include "executor/vectorBatch.h"
#include "access/htup_details.h"
#include "cdb/cdbvars.h"
#include "cdb/htupfifo.h"
#include "cdb/ml_ipc.h"
#include "cdb/tupleremap.h"
#include "cdb/tupser.h"
#include "utils/memutils.h"
#include "utils/typcache.h"


/*
 * MOTION NODE INFO DATA STRUCTURES
 */
int			Gp_max_tuple_chunk_size;

/*
 * STATIC STATE VARS
 *
 * Note, the alignment here isn't quite right.
 * GCC4.0 doesn't like our earlier initializer method of declaration.
 *
 * static TupleChunkListItemData s_eos_chunk_data = {NULL, TUPLE_CHUNK_HEADER_SIZE, NULL, "                "};
 */
static uint8 s_eos_buffer[sizeof(TupleChunkListItemData) + 8];
static TupleChunkListItem s_eos_chunk_data = (TupleChunkListItem) s_eos_buffer;

/*
 * HELPER FUNCTION DECLARATIONS
 */
static MotionNodeEntry *getMotionNodeEntry(MotionLayerState *mlStates, int16 motNodeID);
static ChunkSorterEntry *getChunkSorterEntry(MotionLayerState *mlStates,
					MotionNodeEntry *motNodeEntry,
					int16 srcRoute);
static void addChunkToSorter(MotionLayerState *mlStates,
							 ChunkTransportState *transportStates,
							 MotionNodeEntry *pMNEntry,
							 TupleChunkListItem tcItem,
							 int16 motNodeID,
							 ChunkSorterEntry *chunkSorterEntry,
							 ChunkTransportStateEntry *pEntry,
							 MotionConn *conn,
							 int16 srcRoute);

static void processIncomingChunks(MotionLayerState *mlStates,
					  ChunkTransportState *transportStates,
					  MotionNodeEntry *pMNEntry,
					  int16 motNodeID,
					  int16 srcRoute);

static inline void reconstructTuple(MotionNodeEntry *pMNEntry, ChunkSorterEntry *pCSEntry, TupleRemapper *remapper);

/* Stats-function declarations. */
static void statSendTuple(MotionLayerState *mlStates, MotionNodeEntry *pMNEntry, TupleChunkList tcList);
static void statSendEOS(MotionLayerState *mlStates, MotionNodeEntry *pMNEntry);
static void statChunksProcessed(MotionLayerState *mlStates, MotionNodeEntry *pMNEntry, int chunksProcessed, int chunkBytes, int tupleBytes);
static void statNewTupleArrived(MotionNodeEntry *pMNEntry, ChunkSorterEntry *pCSEntry);
static void statRecvTuple(MotionNodeEntry *pMNEntry, ChunkSorterEntry *pCSEntry);
static bool ShouldSendRecordCache(MotionConn *conn, SerTupInfo *pSerInfo);
static void UpdateSentRecordCache(MotionConn *conn);



/* Helper function to perform the operations necessary to reconstruct a
 * HeapTuple from a list of tuple-chunks, and then update the Motion Layer
 * state appropriately.  This includes storing the tuple, cleaning out the
 * tuple-chunk list, and recording statistics about the newly formed tuple.
 */
static inline void
reconstructTuple(MotionNodeEntry *pMNEntry, ChunkSorterEntry *pCSEntry, TupleRemapper *remapper)
{
	MinimalTuple tup;
	SerTupInfo *pSerInfo = &pMNEntry->ser_tup_info;

	/*
	 * Convert the list of chunks into a tuple, then stow it away.
	 */
	tup = CvtChunksToTup(&pCSEntry->chunk_list, pSerInfo, remapper);

	/* We're done with the chunks now. */
	clearTCList(NULL, &pCSEntry->chunk_list);

	if (!tup)
		return;

	tup = TRCheckAndRemap(remapper, pSerInfo->tupdesc, tup);

	htfifo_addtuple(pCSEntry->ready_tuples, tup);

	/* Stats */
	statNewTupleArrived(pMNEntry, pCSEntry);
}

/*
 * FUNCTION DEFINITIONS
 */

/*
 * This function deletes the Motion Layer state at the end of a query
 * execution.
 */
void
RemoveMotionLayer(MotionLayerState *mlStates)
{
	if (!mlStates)
		return;

	if (Gp_role == GP_ROLE_UTILITY)
		return;

#ifdef AMS_VERBOSE_LOGGING
	/* Emit statistics to log */
	if (gp_log_interconnect >= GPVARS_VERBOSITY_VERBOSE)
		elog(LOG, "RemoveMotionLayer(): dumping stats\n"
			 "      Sent: %9u chunks %9u total bytes %9u tuple bytes\n"
			 "  Received: %9u chunks %9u total bytes %9u tuple bytes; "
			 "%9u chunkproc calls\n",
			 mlStates->stat_total_chunks_sent,
			 mlStates->stat_total_bytes_sent,
			 mlStates->stat_tuple_bytes_sent,
			 mlStates->stat_total_chunks_recvd,
			 mlStates->stat_total_bytes_recvd,
			 mlStates->stat_tuple_bytes_recvd,
			 mlStates->stat_total_chunkproc_calls);
#endif

	/*
	 * Free all memory used by the Motion Layer in the processing of this
	 * query.
	 */
	if (mlStates->motion_layer_mctx != NULL)
		MemoryContextDelete(mlStates->motion_layer_mctx);
}


MotionLayerState *
createMotionLayerState(int maxMotNodeID)
{
	MemoryContext oldCtxt;
	MemoryContext ml_mctx;
	uint8	   *pData;
	MotionLayerState *mlState;

	if (Gp_role == GP_ROLE_UTILITY)
		return NULL;

	if (Gp_interconnect_type == INTERCONNECT_TYPE_UDPIFC)
		Gp_max_tuple_chunk_size = Gp_max_packet_size - sizeof(struct icpkthdr) - TUPLE_CHUNK_HEADER_SIZE;
	else if (Gp_interconnect_type == INTERCONNECT_TYPE_TCP ||
			 Gp_interconnect_type == INTERCONNECT_TYPE_PROXY)
		Gp_max_tuple_chunk_size = Gp_max_packet_size - PACKET_HEADER_SIZE - TUPLE_CHUNK_HEADER_SIZE;

	/*
	 * Use the statically allocated chunk that is intended for sending end-of-
	 * stream messages so that we don't incur allocation and deallocation
	 * overheads.
	 */
	s_eos_chunk_data->p_next = NULL;
	s_eos_chunk_data->inplace = NULL;
	s_eos_chunk_data->chunk_length = TUPLE_CHUNK_HEADER_SIZE;

	pData = s_eos_chunk_data->chunk_data;

	SetChunkDataSize(pData, 0);
	SetChunkType(pData, TC_END_OF_STREAM);

	/*
	 * Create the memory-contexts that we will use within the Motion Layer.
	 *
	 * We make the Motion Layer memory-context a child of the ExecutorState
	 * Context, as it lives inside of the estate of a specific query and needs
	 * to get freed when the query is finished.
	 *
	 * The tuple-serial memory-context is a child of the Motion Layer
	 * memory-context
	 *
	 * NOTE: we need to be sure the caller is in ExecutorState memory context
	 * (estate->es_query_cxt) before calling us .
	 */
	ml_mctx =
		AllocSetContextCreate(CurrentMemoryContext, "MotionLayerMemCtxt",
							  ALLOCSET_SMALL_MINSIZE,
							  ALLOCSET_SMALL_INITSIZE,
							  ALLOCSET_DEFAULT_MAXSIZE);	/* use a setting bigger
															 * than "small" */

	/*
	 * Switch to the Motion Layer memory context, so that we can clean things
	 * up easily.
	 */
	oldCtxt = MemoryContextSwitchTo(ml_mctx);

	mlState = palloc0(sizeof(MotionLayerState));

	mlState->mnEntries = palloc0(maxMotNodeID * sizeof(MotionNodeEntry));
	mlState->mneCount = maxMotNodeID;

	for (int motNodeID = 1; motNodeID <= maxMotNodeID; motNodeID++)
	{
		MotionNodeEntry *pEntry = &mlState->mnEntries[motNodeID - 1];

		pEntry->motion_node_id = motNodeID;
		pEntry->valid = true;

		/*
		 * we'll just set this to 0.  later, ml_ipc will call
		 * UpdateMotionExpectedReceivers() to set this if we are a "Receiving" motion node.
		 */
		pEntry->num_senders = 0;
	}

	/* Allocation is done.	Go back to caller memory-context. */
	MemoryContextSwitchTo(oldCtxt);

	/*
	 * Keep our motion layer memory context in our newly created motion layer.
	 */
	mlState->motion_layer_mctx = ml_mctx;

	return mlState;
}

/*
 * Initialize a single motion node.  This is called by the executor when a
 * motion node in the plan tree is being initialized.
 *
 * This function is called from:  ExecInitMotion()
 */
void
UpdateMotionLayerNode(MotionLayerState *mlStates, int16 motNodeID, bool preserveOrder, TupleDesc tupDesc)
{
	MemoryContext oldCtxt;
	MotionNodeEntry *pEntry;

	if (motNodeID < 1 || motNodeID > mlStates->mneCount)
		elog(ERROR, "invalid motion node ID %d", motNodeID);

	AssertArg(tupDesc != NULL);

	/*
	 * Switch to the Motion Layer's memory-context, so that the motion node
	 * can be reset later.
	 */
	oldCtxt = MemoryContextSwitchTo(mlStates->motion_layer_mctx);

	pEntry = &mlStates->mnEntries[motNodeID - 1];

	Assert(pEntry->valid);

	pEntry->motion_node_id = motNodeID;

	/* Finish up initialization of the motion node entry. */
	pEntry->preserve_order = preserveOrder;
	pEntry->tuple_desc = CreateTupleDescCopy(tupDesc);
	InitSerTupInfo(pEntry->tuple_desc, &pEntry->ser_tup_info);

	if (!preserveOrder)
	{
		/* Create a tuple-store for the motion node's incoming tuples. */
		pEntry->ready_tuples = htfifo_create();
		/* ready_batch_head/tail stay NULL: mnEntries is zero-allocated */
	}
	else
		pEntry->ready_tuples = NULL;


	pEntry->num_stream_ends_recvd = 0;

	/* Initialize statistics counters. */
	pEntry->stat_total_chunks_sent = 0;
	pEntry->stat_total_bytes_sent = 0;
	pEntry->stat_tuple_bytes_sent = 0;
	pEntry->stat_total_sends = 0;
	pEntry->stat_total_recvs = 0;
	pEntry->stat_tuples_available = 0;
	pEntry->stat_tuples_available_hwm = 0;
	pEntry->stat_total_chunks_recvd = 0;
	pEntry->stat_total_bytes_recvd = 0;
	pEntry->stat_tuple_bytes_recvd = 0;

	pEntry->cleanedUp = false;
	pEntry->stopped = false;
	pEntry->moreNetWork = true;


	/* All done!  Go back to caller memory-context. */
	MemoryContextSwitchTo(oldCtxt);
}

void
UpdateMotionExpectedReceivers(MotionLayerState *mlStates, SliceTable *sliceTable)
{
	ExecSlice  *mySlice;
	ExecSlice  *aSlice;
	ListCell   *cell;
	CdbProcess *cdbProc;
	MotionNodeEntry *pEntry;

	mySlice = &sliceTable->slices[sliceTable->localSlice];
	foreach(cell, mySlice->children)
	{
		int			totalNumProcs, activeNumProcs, i;
		int			childId = lfirst_int(cell);

		aSlice = &sliceTable->slices[childId];

		/*
		 * If we're using directed-dispatch we have dummy primary-process
		 * entries, so we count the entries.
		 */
		activeNumProcs = 0;
		totalNumProcs = list_length(aSlice->primaryProcesses);
		for (i = 0; i < totalNumProcs; i++)
		{
			cdbProc = list_nth(aSlice->primaryProcesses, i);
			if (cdbProc)
				activeNumProcs++;
		}

		pEntry = getMotionNodeEntry(mlStates, childId);
		pEntry->num_senders = activeNumProcs;
	}
}

void
SendStopMessage(MotionLayerState *mlStates,
				ChunkTransportState *transportStates,
				int16 motNodeID)
{
	MotionNodeEntry *pEntry = getMotionNodeEntry(mlStates, motNodeID);

	pEntry->stopped = true;
	if (transportStates != NULL && transportStates->doSendStopMessage != NULL)
		transportStates->doSendStopMessage(transportStates, motNodeID);
}

void
CheckAndSendRecordCache(MotionLayerState *mlStates,
						ChunkTransportState *transportStates,
						int16 motNodeID,
						int16 targetRoute)
{
	MotionNodeEntry *pMNEntry;
	TupleChunkListData tcList;
	MemoryContext oldCtxt;
	ChunkTransportStateEntry *pEntry = NULL;
	MotionConn *conn;

	getChunkTransportState(transportStates, motNodeID, &pEntry);

	/*
	 * for broadcast we only mark sent_record_typmod for connection 0 for
	 * efficiency and convenience
	 */
	if (targetRoute == BROADCAST_SEGIDX)
		conn = &pEntry->conns[0];
	else
		conn = &pEntry->conns[targetRoute];

	/*
	 * Analyze tools.  Do not send any thing if this slice is in the bit mask
	 */
	if (gp_motion_slice_noop != 0 && (gp_motion_slice_noop & (1 << currentSliceId)) != 0)
		return;

	/*
	 * Pull up the motion node entry with the node's details.  This includes
	 * details that affect sending, such as whether the motion node needs to
	 * include backup segment-dbs.
	 */
	pMNEntry = getMotionNodeEntry(mlStates, motNodeID);

	if (!ShouldSendRecordCache(conn, &pMNEntry->ser_tup_info))
		return;

#ifdef AMS_VERBOSE_LOGGING
	elog(DEBUG5, "Serializing RecordCache for sending.");
#endif

	/* Create and store the serialized form, and some stats about it. */
	oldCtxt = MemoryContextSwitchTo(mlStates->motion_layer_mctx);

	SerializeRecordCacheIntoChunks(&pMNEntry->ser_tup_info, &tcList, conn);

	MemoryContextSwitchTo(oldCtxt);

#ifdef AMS_VERBOSE_LOGGING
	elog(DEBUG5, "Serialized RecordCache for sending:\n"
		 "\ttarget-route %d \n"
		 "\t%d bytes in serial form\n"
		 "\tbroken into %d chunks",
		 targetRoute,
		 tcList.serialized_data_length,
		 tcList.num_chunks);
#endif

	/* do the send. */
	if (!SendTupleChunkToAMS(mlStates, transportStates, motNodeID, targetRoute, tcList.p_first))
	{
		pMNEntry->stopped = true;
	}
	else
	{
		/* update stats */
		statSendTuple(mlStates, pMNEntry, &tcList);
	}

	/* cleanup */
	clearTCList(&pMNEntry->ser_tup_info.chunkCache, &tcList);

	UpdateSentRecordCache(conn);
}

/*
 * Function:  SendTuple - Sends a portion or whole tuple to the AMS layer.
 */
SendReturnCode
SendTuple(MotionLayerState *mlStates,
		  ChunkTransportState *transportStates,
		  int16 motNodeID,
		  TupleTableSlot *slot,
		  int16 targetRoute)
{
	MotionNodeEntry *pMNEntry;
	TupleChunkListData tcList;
	MemoryContext oldCtxt;
	SendReturnCode rc;

	AssertArg(!TupIsNull(slot));

	/*
	 * Analyze tools.  Do not send any thing if this slice is in the bit mask
	 */
	if (gp_motion_slice_noop != 0 && (gp_motion_slice_noop & (1 << currentSliceId)) != 0)
		return SEND_COMPLETE;

	/*
	 * Pull up the motion node entry with the node's details.  This includes
	 * details that affect sending, such as whether the motion node needs to
	 * include backup segment-dbs.
	 */
	pMNEntry = getMotionNodeEntry(mlStates, motNodeID);

#ifdef AMS_VERBOSE_LOGGING
	elog(DEBUG5, "Serializing HeapTuple for sending.");
#endif

	struct directTransportBuffer b;
	if (targetRoute != BROADCAST_SEGIDX)
		getTransportDirectBuffer(transportStates, motNodeID, targetRoute, &b);

	int			sent = 0;

	/* Create and store the serialized form, and some stats about it. */
	oldCtxt = MemoryContextSwitchTo(mlStates->motion_layer_mctx);

	sent = SerializeTuple(slot, &pMNEntry->ser_tup_info, &b, &tcList, targetRoute);

	MemoryContextSwitchTo(oldCtxt);
	if (sent > 0)
	{
		putTransportDirectBuffer(transportStates, motNodeID, targetRoute, sent);

		/* fill-in tcList fields to update stats */
		tcList.num_chunks = 1;
		tcList.serialized_data_length = sent;

		/* update stats */
		statSendTuple(mlStates, pMNEntry, &tcList);

		return SEND_COMPLETE;
	}
	/* Otherwise fall-through */

#ifdef AMS_VERBOSE_LOGGING
	elog(DEBUG5, "Serialized HeapTuple for sending:\n"
		 "\ttarget-route %d \n"
		 "\t%d bytes in serial form\n"
		 "\tbroken into %d chunks",
		 targetRoute,
		 tcList.serialized_data_length,
		 tcList.num_chunks);
#endif

	/* do the send. */
	if (!SendTupleChunkToAMS(mlStates, transportStates, motNodeID, targetRoute, tcList.p_first))
	{
		pMNEntry->stopped = true;
		rc = STOP_SENDING;
	}
	else
	{
		/* update stats */
		statSendTuple(mlStates, pMNEntry, &tcList);

		rc = SEND_COMPLETE;
	}

	/* cleanup */
	clearTCList(&pMNEntry->ser_tup_info.chunkCache, &tcList);

	return rc;
}

/*
 * Send one encoded vector batch (VectorBatchEncodeRows payload) to a
 * single route.  The payload is split over TC_BATCH_START..MID*..END
 * chunks the same way large tuples are, so the interconnect treats it
 * as opaque bytes.
 */
SendReturnCode
SendBatchChunk(MotionLayerState *mlStates,
			   ChunkTransportState *transportStates,
			   int16 motNodeID,
			   const char *payload, int32 len,
			   int16 targetRoute)
{
	MotionNodeEntry *pMNEntry;
	const char *pos = payload;
	int32		remaining = len;
	MemoryContext oldCtxt;

	if (gp_motion_slice_noop != 0 && (gp_motion_slice_noop & (1 << currentSliceId)) != 0)
		return SEND_COMPLETE;

	pMNEntry = getMotionNodeEntry(mlStates, motNodeID);
	oldCtxt = MemoryContextSwitchTo(mlStates->motion_layer_mctx);

	{
		int32		space = Gp_max_tuple_chunk_size - TUPLE_CHUNK_HEADER_SIZE;

		/*
		 * A batch that fits one chunk travels as a single TC_BATCH_END
		 * chunk carrying the whole payload (a bare END marker would be
		 * payload-less and may not survive the interconnect).
		 */
		if (len <= space)
		{
			TupleChunkListItem item = getChunkFromCache(&pMNEntry->ser_tup_info.chunkCache);

			SetChunkDataSize(item->chunk_data, len);
			SetChunkType(item->chunk_data, TC_BATCH_END);
			memcpy(item->chunk_data + TUPLE_CHUNK_HEADER_SIZE, payload, len);
			item->chunk_length = len + TUPLE_CHUNK_HEADER_SIZE;
			item->inplace = NULL;
			item->p_next = NULL;
			if (!SendTupleChunkToAMS(mlStates, transportStates, motNodeID,
									 targetRoute, item))
			{
				pMNEntry->stopped = true;
				MemoryContextSwitchTo(oldCtxt);
				return STOP_SENDING;
			}
			MemoryContextSwitchTo(oldCtxt);
			return SEND_COMPLETE;
		}

		for (;;)
		{
			TupleChunkListItem item;
			int32		n = Min(space, remaining);
			bool		last = (remaining <= space);
			TupleChunkType tcType;

			if (pos == payload)
				tcType = TC_BATCH_START;
			else
				tcType = last ? TC_BATCH_END : TC_BATCH_MID;

			item = getChunkFromCache(&pMNEntry->ser_tup_info.chunkCache);
			SetChunkDataSize(item->chunk_data, n);
			SetChunkType(item->chunk_data, tcType);
			if (n > 0)
				memcpy(item->chunk_data + TUPLE_CHUNK_HEADER_SIZE, pos, n);
			item->chunk_length = n + TUPLE_CHUNK_HEADER_SIZE;
			item->inplace = NULL;
			item->p_next = NULL;

			if (!SendTupleChunkToAMS(mlStates, transportStates, motNodeID,
									 targetRoute, item))
			{
				pMNEntry->stopped = true;
				MemoryContextSwitchTo(oldCtxt);
				return STOP_SENDING;
			}

			pos += n;
			remaining -= n;

			if (last)
				break;
		}

	}

	MemoryContextSwitchTo(oldCtxt);
	return SEND_COMPLETE;
}

/*
 * The wire payload buffer carries its allocation size in a leading
 * Size word (the payload proper starts one MAXALIGN step in, so the
 * datum alignment is preserved): the recycling pool needs the size at
 * hand-back, when nothing else remembers it.  MAXALIGN keeps the
 * payload Datum-aligned; offsetof(StringInfoData, data) would be zero
 * and put the header inside the payload itself.
 */
#define WIRE_PAYLOAD_HDR	MAXALIGN(sizeof(Size))

/*
 * VectorBatchDestroy's recycle hook for fast-table aggregate batches:
 * the owner pools the shell (columns, selection, varlena context) and
 * the payload buffer.  Destroy is the release point of the batch
 * lifetime contract, so nothing can hold the storage once it is
 * pooled.
 */
void
VectorMotionRecycleBatch(void *owner, VectorBatch *batch)
{
	MotionNodeEntry *pMNEntry = (MotionNodeEntry *) owner;

	if (batch->wire_data != NULL)
	{
		char	   *raw = batch->wire_data - WIRE_PAYLOAD_HDR;
		Size		sz = *(Size *) raw;

		if (pMNEntry->wire_pool_count < MOTION_WIRE_POOL_MAX)
		{
			pMNEntry->wire_pool[pMNEntry->wire_pool_count] = raw;
			pMNEntry->wire_pool_cap[pMNEntry->wire_pool_count] = sz;
			pMNEntry->wire_pool_count++;
		}
		else
		{
			/* pool full: drop the smallest, keep the bigger buffers */
			int			i_min = 0;
			int			i;

			for (i = 1; i < MOTION_WIRE_POOL_MAX; i++)
				if (pMNEntry->wire_pool_cap[i] < pMNEntry->wire_pool_cap[i_min])
					i_min = i;
			if (sz >= pMNEntry->wire_pool_cap[i_min])
			{
				pfree(pMNEntry->wire_pool[i_min]);
				pMNEntry->wire_pool[i_min] = raw;
				pMNEntry->wire_pool_cap[i_min] = sz;
			}
			else
				pfree(raw);
		}
		batch->wire_data = NULL;
	}

	/* park the shell with its columns and context intact */
	batch->wire_owner = NULL;
	batch->wire_recycle = NULL;
	VectorBatchReset(batch);
	if (pMNEntry->shell_pool_count < MOTION_SHELL_POOL_MAX)
		pMNEntry->shell_pool[pMNEntry->shell_pool_count++] = batch;
	else
		VectorBatchDestroy(batch);
}

/*
 * Take a payload buffer from the recycling pool (the first one big
 * enough) or allocate a fresh one; the caller assembles into the data
 * after the size header.
 */
static char *
TakeWireBuffer(MotionNodeEntry *pMNEntry, int len)
{
	Size		need = (Size) len + 1 + WIRE_PAYLOAD_HDR;
	int			i;

	for (i = 0; i < pMNEntry->wire_pool_count; i++)
	{
		if (pMNEntry->wire_pool_cap[i] >= need)
		{
			char	   *raw = pMNEntry->wire_pool[i];

			pMNEntry->wire_pool[i] = pMNEntry->wire_pool[--pMNEntry->wire_pool_count];
			pMNEntry->wire_pool_cap[i] = pMNEntry->wire_pool_cap[pMNEntry->wire_pool_count];
			*(Size *) raw = need;
			return raw + WIRE_PAYLOAD_HDR;
		}
	}
	{
		char	   *raw = (char *) palloc(need);

		*(Size *) raw = need;
		return raw + WIRE_PAYLOAD_HDR;
	}
}

/*
 * Take a recycled batch shell (columns, selection and varlena context
 * intact) or create one, and wire it to this entry's recycle hook.
 */
static VectorBatch *
TakeMotionBatch(MotionLayerState *mlStates, MotionNodeEntry *pMNEntry)
{
	VectorBatch  *batch;

	if (pMNEntry->shell_pool_count > 0)
		batch = pMNEntry->shell_pool[--pMNEntry->shell_pool_count];
	else
		batch = VectorBatchCreate(mlStates->motion_layer_mctx,
								  pMNEntry->tuple_desc,
								  VECTOR_BATCH_DEFAULT_CAPACITY, false);
	batch->wire_owner = pMNEntry;
	batch->wire_recycle = VectorMotionRecycleBatch;
	return batch;
}

/*
 * Reassemble a TC_BATCH_* chunk sequence into one columnar batch and
 * queue it on the motion node entry for a vectorized consumer.
 */
static void
reconstructBatch(MotionLayerState *mlStates, MotionNodeEntry *pMNEntry,
				 ChunkSorterEntry *pCSEntry)
{
	TupleChunkListItem tcItem;
	StringInfoData buf;
	VectorBatch  *batch;
	int			len = 0;

	/*
	 * One payload buffer per batch, recycled through the entry's pool at
	 * destroy: the decode hands the buffer over to the batch and its
	 * by-reference values point into it (zero-copy), so the buffer must
	 * stay untouched until the batch is destroyed.  The exact presize
	 * never grows the allocation.
	 */
	for (tcItem = pCSEntry->chunk_list.p_first; tcItem != NULL;
		 tcItem = tcItem->p_next)
		len += tcItem->chunk_length - TUPLE_CHUNK_HEADER_SIZE;

	buf.data = TakeWireBuffer(pMNEntry, len);
	buf.maxlen = len + 1;
	buf.len = 0;
	buf.cursor = 0;

	for (tcItem = pCSEntry->chunk_list.p_first; tcItem != NULL;
		 tcItem = tcItem->p_next)
		appendBinaryStringInfo(&buf,
							   GetChunkDataPtr(tcItem) + TUPLE_CHUNK_HEADER_SIZE,
							   tcItem->chunk_length - TUPLE_CHUNK_HEADER_SIZE);

	batch = TakeMotionBatch(mlStates, pMNEntry);
	VectorBatchDecodeRowsInto(batch, buf.data, buf.len);

	{
		ReadyBatch *node = (ReadyBatch *) palloc(sizeof(ReadyBatch));

		node->batch = batch;
		node->next = NULL;
		if (pMNEntry->ready_batch_tail != NULL)
			pMNEntry->ready_batch_tail->next = node;
		else
			pMNEntry->ready_batch_head = node;
		pMNEntry->ready_batch_tail = node;
	}

	clearTCList(&pMNEntry->ser_tup_info.chunkCache, &pCSEntry->chunk_list);
}

/*
 * Receive the next decoded vector batch from an unsorted motion, or
 * NULL at end of stream.  Tuples already received stay in the tuple
 * FIFO; a stream is homogeneous per sender, so mixed streams do not
 * occur in practice.
 */
VectorBatch *
RecvBatchFrom(MotionLayerState *mlStates,
			  ChunkTransportState *transportStates,
			  int16 motNodeID,
			  bool *tuples_only)
{
	MotionNodeEntry *pMNEntry = getMotionNodeEntry(mlStates, motNodeID);

	Assert(!pMNEntry->preserve_order);
	if (tuples_only != NULL)
		*tuples_only = false;

	for (;;)
	{
		ReadyBatch *node = pMNEntry->ready_batch_head;
		VectorBatch *batch;

		if (node != NULL)
		{
			pMNEntry->ready_batch_head = node->next;
			if (pMNEntry->ready_batch_head == NULL)
				pMNEntry->ready_batch_tail = NULL;
			batch = node->batch;
			pfree(node);
			return batch;
		}

		/*
		 * Tuple chunks on the FIFO mean the sender did not batch; leave
		 * them for the per-row path.
		 */
		if (pMNEntry->ready_tuples->p_first != NULL)
		{
			if (tuples_only != NULL)
				*tuples_only = true;
			return NULL;
		}

		if (!pMNEntry->moreNetWork)
			return NULL;

		processIncomingChunks(mlStates, transportStates, pMNEntry,
							  motNodeID, ANY_ROUTE);
	}
}

/*
 * Sends a token to all peer Motion Nodes, indicating that this motion
 * node has no more tuples to send out.
 */
void
SendEndOfStream(MotionLayerState *mlStates,
				ChunkTransportState *transportStates,
				int motNodeID)
{
	MotionNodeEntry *pMNEntry;

	/*
	 * Pull up the motion node entry with the node's details.  This includes
	 * details that affect sending, such as whether the motion node needs to
	 * include backup segment-dbs.
	 */
	pMNEntry = getMotionNodeEntry(mlStates, motNodeID);

	transportStates->SendEos(transportStates, motNodeID, s_eos_chunk_data);

	/*
	 * We increment our own "stream-ends received" count when we send our own,
	 * as well as when we receive one.
	 */
	pMNEntry->num_stream_ends_recvd++;

	/* We record EOS as if a tuple were sent. */
	statSendEOS(mlStates, pMNEntry);
}

/*
 * A row-wise consumer drains a vectorized stream through the tuple FIFO:
 * decompose the next queued batch into minimal tuples.  Batches live in
 * the motion layer context, the deformed tuples follow the usual FIFO
 * ownership (the consumer frees them).
 */
/*
 * Take ownership of the queued batch at the head of the batch queue:
 * the ReadyBatch node is unlinked and freed, the decoded batch moves
 * to the entry's pending slot for row-by-row vending.
 */
static void
TakePendingBatch(MotionNodeEntry *pMNEntry)
{
	ReadyBatch *node = pMNEntry->ready_batch_head;

	if (node == NULL)
		return;
	pMNEntry->ready_batch_head = node->next;
	if (pMNEntry->ready_batch_head == NULL)
		pMNEntry->ready_batch_tail = NULL;
	pMNEntry->pending_batch = node->batch;
	pMNEntry->pending_row = 0;
	pfree(node);
}

/*
 * Form the pending batch's current row as the next tuple for a row-wise
 * consumer, advancing the vending cursor; destroy the batch when its
 * rows run out.  Values are read selection-aware and by-reference data
 * needs no copy: heap_form_minimal_tuple copies it into the tuple body,
 * and the batch outlives the call.
 */
static MinimalTuple
VendPendingBatchTuple(MotionNodeEntry *pMNEntry)
{
	VectorBatch *batch = pMNEntry->pending_batch;
	TupleDesc	tupdesc = pMNEntry->tuple_desc;
	int			natts = tupdesc->natts;
	Datum	   *values = (Datum *) palloc(sizeof(Datum) * Max(natts, 1));
	bool	   *isnull = (bool *) palloc(sizeof(bool) * Max(natts, 1));
	MinimalTuple tup;
	int			col;
	int			row = VectorBatchSelectedRow(batch, pMNEntry->pending_row);

	for (col = 0; col < natts; col++)
	{
		isnull[col] = VectorBatchIsNull(batch, col, row);
		values[col] = isnull[col] ? 0 :
			VectorBatchGetDatum(batch, col, row);
	}
	tup = heap_form_minimal_tuple(tupdesc, values, isnull);
	pfree(values);
	pfree(isnull);

	pMNEntry->pending_row++;
	if (pMNEntry->pending_row >= batch->nselected)
	{
		VectorBatchDestroy(batch);
		pMNEntry->pending_batch = NULL;
		pMNEntry->pending_row = 0;
	}
	return tup;
}

/*
 * Receive one tuple from a sender. An unordered receiver will call this with
 * srcRoute == ANY_ROUTE.
 *
 * The tuple is stored in *slot.
 */
MinimalTuple
RecvTupleFrom(MotionLayerState *mlStates,
			  ChunkTransportState *transportStates,
			  int16 motNodeID,
			  int16 srcRoute)
{
	MotionNodeEntry *pMNEntry;
	ChunkSorterEntry *pCSEntry;
	htup_fifo	ReadyList;
	MinimalTuple tuple = NULL;

#ifdef AMS_VERBOSE_LOGGING
	elog(DEBUG5, "RecvTupleFrom( motNodeID = %d, srcRoute = %d )", motNodeID, srcRoute);
#endif

	pMNEntry = getMotionNodeEntry(mlStates, motNodeID);

	if (srcRoute == ANY_ROUTE)
	{
		Assert(pMNEntry->preserve_order == 0);
		pCSEntry = NULL;

		ReadyList = pMNEntry->ready_tuples;
	}
	else
	{
		Assert(pMNEntry->preserve_order != 0);

		/*
		 * Pull up the chunk-sorter entry for the specified sender, and get
		 * the tuple-store we should use.
		 */
		pCSEntry = getChunkSorterEntry(mlStates, pMNEntry, srcRoute);
		ReadyList = pCSEntry->ready_tuples;
	}

	for (;;)
	{
		/*
		 * A pending batch vends first: batch chunks and tuple chunks
		 * never mix on one stream, so the two sources cannot both be
		 * non-empty.
		 */
		if (pMNEntry->pending_batch != NULL)
		{
			tuple = VendPendingBatchTuple(pMNEntry);
			if (tuple)
				break;
			continue;
		}
		/* Get the next tuple from the FIFO, if one is available. */
		tuple = htfifo_gettuple(ReadyList);
		if (tuple)
			break;

		/* row-wise consumers drain vectorized streams via tuples */
		if (pMNEntry->ready_batch_head != NULL)
		{
			TakePendingBatch(pMNEntry);
			continue;
		}

		/*
		 * We need to get more chunks before we have a full tuple to return. Loop
		 * until we get one, or we reach end-of-stream.
		 */
		if ((srcRoute == ANY_ROUTE && !pMNEntry->moreNetWork) ||
			(srcRoute != ANY_ROUTE && pCSEntry->end_of_stream))
		{
			/*
			 * No tuple was available (tuple-store was at EOF), and
			 * end-of-stream has been marked.  No more tuples are going to
			 * show up.
			 */
			break;
		}

		processIncomingChunks(mlStates, transportStates, pMNEntry, motNodeID, srcRoute);
	}

	/* Stats */
	if (tuple)
		statRecvTuple(pMNEntry, pCSEntry);

	return tuple;
}


/*
 * This helper function is the receive-tuple workhorse.  It pulls
 * tuple chunks from the AMS, and pushes them to the chunk-sorter
 * where they can be sorted based on sender, and reconstituted into
 * whole HeapTuples.  Functions like RecvTuple() and RecvTupleFrom()
 * should definitely call this before trying to get a HeapTuple to
 * return.  It can also be called during other operations if that
 * seems like a good idea.  For example, it can be called sometime
 * during send-tuple operations as well.
 */
static void
processIncomingChunks(MotionLayerState *mlStates,
					  ChunkTransportState *transportStates,
					  MotionNodeEntry *pMNEntry,
					  int16 motNodeID,
					  int16 srcRoute)
{
	TupleChunkListItem tcItem,
				tcNext;
	MemoryContext oldCtxt;
	ChunkSorterEntry *chunkSorterEntry;
	ChunkTransportStateEntry *pEntry = NULL;
	MotionConn *conn;

	/* Keep track of processed chunk stats. */
	int			numChunks,
				chunkBytes,
				tupleBytes;

	oldCtxt = MemoryContextSwitchTo(mlStates->motion_layer_mctx);

	/*
	 * Get all of the currently available tuple-chunks, and push each one into
	 * the chunk-sorter.
	 */
	if (srcRoute == ANY_ROUTE)
		tcItem = transportStates->RecvTupleChunkFromAny(transportStates, motNodeID, &srcRoute);
	else
		tcItem = transportStates->RecvTupleChunkFrom(transportStates, motNodeID, srcRoute);

	/* Look up various things related to the sender that we received chunks from. */
	chunkSorterEntry = getChunkSorterEntry(mlStates, pMNEntry, srcRoute);
	getChunkTransportState(transportStates, motNodeID, &pEntry);
	conn = pEntry->conns + srcRoute;

	numChunks = 0;
	chunkBytes = 0;
	tupleBytes = 0;

	while (tcItem != NULL)
	{
		numChunks++;

		/* Detach the current chunk off of the front of the list. */
		tcNext = tcItem->p_next;
		tcItem->p_next = NULL;

		/* Track stats. */
		chunkBytes += tcItem->chunk_length;
		if (tcItem->chunk_length >= TUPLE_CHUNK_HEADER_SIZE)
		{
			tupleBytes += tcItem->chunk_length - TUPLE_CHUNK_HEADER_SIZE;
		}
		else
		{
			elog(ERROR, "Received tuple-chunk of size %u; smaller than"
				 " chunk header size %d!", tcItem->chunk_length,
				 TUPLE_CHUNK_HEADER_SIZE);
		}

		/* Stick the chunk into the sorter. */
		addChunkToSorter(mlStates,
						 transportStates,
						 pMNEntry,
						 tcItem,
						 motNodeID,
						 chunkSorterEntry,
						 pEntry,
						 conn,
						 srcRoute);

		tcItem = tcNext;
	}

	/* The chunk list we just processed freed-up our rx-buffer space. */
	if (numChunks > 0 && Gp_interconnect_type == INTERCONNECT_TYPE_UDPIFC)
		MlPutRxBufferIFC(transportStates, motNodeID, srcRoute);

	/* Stats */
	statChunksProcessed(mlStates, pMNEntry, numChunks, chunkBytes, tupleBytes);

	MemoryContextSwitchTo(oldCtxt);
}

void
EndMotionLayerNode(MotionLayerState *mlStates, int16 motNodeID, bool flushCommLayer)
{
	MotionNodeEntry *pMNEntry;
	ChunkSorterEntry *pCSEntry;
	int			i;

	pMNEntry = getMotionNodeEntry(mlStates, motNodeID);

#ifdef AMS_VERBOSE_LOGGING
	elog(DEBUG5, "Cleaning up Motion Layer details for motion node %d.",
		 motNodeID);
#endif

	/*
	 * Iterate through all entries in the motion layer's chunk-sort map, to
	 * see if we have gotten end-of-stream from all senders.
	 */
	if (pMNEntry->preserve_order && pMNEntry->ready_tuple_lists != NULL)
	{
		for (i = 0; i < pMNEntry->num_senders; i++)
		{
			pCSEntry = &pMNEntry->ready_tuple_lists[i];

			/*
			 * QD should not expect end-of-stream comes from QEs who is not
			 * members of direct dispatch
			 */
			if (!pCSEntry->init)
				continue;

			if (pMNEntry->preserve_order &&
				gp_log_interconnect >= GPVARS_VERBOSITY_DEBUG)
			{
				/* Print chunk-sorter entry statistics. */
				elog(DEBUG4, "Chunk-sorter entry [route=%d,node=%d] statistics:\n"
					 "\tAvailable Tuples High-Watermark: " UINT64_FORMAT,
					 i, pMNEntry->motion_node_id,
					 pMNEntry->stat_tuples_available_hwm);
			}

			if (!pMNEntry->stopped && !pCSEntry->end_of_stream)
			{
				if (flushCommLayer)
				{
					elog(FATAL, "Motion layer node %d cleanup - did not receive"
						 " end-of-stream from sender %d.", motNodeID, i);

					/*** TODO - get chunks until end-of-stream comes in. ***/
				}
				else
				{
					elog(LOG, "Motion layer node %d cleanup - did not receive"
						 " end-of-stream from sender %d.", motNodeID, i);
				}
			}
			else
			{
				/* End-of-stream is marked for this entry. */

				/*** TODO - do more than just complain! ***/

				if (pCSEntry->chunk_list.num_chunks > 0)
				{
					elog(LOG, "Motion layer node %d cleanup - there are still"
						 " %d chunks enqueued from sender %d.", motNodeID,
						 pCSEntry->chunk_list.num_chunks, i);
				}

				/***
					TODO - Make sure there are no outstanding tuples in the
					tuple-store.
				***/
			}

			/*
			 * Clean up the chunk-sorter entry, then remove it from the hash
			 * table.
			 */
			clearTCList(&pMNEntry->ser_tup_info.chunkCache, &pCSEntry->chunk_list);
			if (pMNEntry->preserve_order)	/* Clean up the tuple-store. */
				htfifo_destroy(pCSEntry->ready_tuples);
		}
	}
	pMNEntry->cleanedUp = true;

	/* Clean up the motion-node entry, then remove it from the hash table. */
	if (gp_log_interconnect >= GPVARS_VERBOSITY_VERBOSE)
	{
		if (pMNEntry->stat_total_bytes_sent > 0)
		{
			elog(LOG, "Interconnect seg%d slice%d sent " UINT64_FORMAT " tuples, "
				 UINT64_FORMAT " total bytes, " UINT64_FORMAT " tuple bytes, "
				 UINT64_FORMAT " chunks.",
				 GpIdentity.segindex,
				 currentSliceId,
				 pMNEntry->stat_total_sends,
				 pMNEntry->stat_total_bytes_sent,
				 pMNEntry->stat_tuple_bytes_sent,
				 pMNEntry->stat_total_chunks_sent
				);
		}
		if (pMNEntry->stat_total_bytes_recvd > 0)
		{
			elog(LOG, "Interconnect seg%d slice%d received from slice%d: " UINT64_FORMAT " tuples, "
				 UINT64_FORMAT " total bytes, " UINT64_FORMAT " tuple bytes, "
				 UINT64_FORMAT " chunks.",
				 GpIdentity.segindex,
				 currentSliceId,
				 motNodeID,
				 pMNEntry->stat_total_recvs,
				 pMNEntry->stat_total_bytes_recvd,
				 pMNEntry->stat_tuple_bytes_recvd,
				 pMNEntry->stat_total_chunks_recvd
				);
		}
	}

	CleanupSerTupInfo(&pMNEntry->ser_tup_info);
	FreeTupleDesc(pMNEntry->tuple_desc);
	if (!pMNEntry->preserve_order)
		htfifo_destroy(pMNEntry->ready_tuples);
	if (pMNEntry->pending_batch != NULL)
	{
		VectorBatchDestroy(pMNEntry->pending_batch);
		pMNEntry->pending_batch = NULL;
	}

	pMNEntry->valid = false;
}

/*
 * Helper function to get the motion node entry for a given ID.  NULL
 * is returned if the ID is unrecognized.
 */
static MotionNodeEntry *
getMotionNodeEntry(MotionLayerState *mlStates, int16 motNodeID)
{
	MotionNodeEntry *pMNEntry = NULL;

	if (motNodeID > mlStates->mneCount ||
		!mlStates->mnEntries[motNodeID - 1].valid)
	{
		ereport(ERROR,
				(errcode(ERRCODE_GP_INTERCONNECTION_ERROR),
				 errmsg("interconnect error: Unexpected Motion Node Id: %d",
						motNodeID),
				 errdetail("This means a motion node that wasn't setup is requesting interconnect resources.")));
	}
	else
		pMNEntry = &mlStates->mnEntries[motNodeID - 1];

	if (pMNEntry != NULL)
		Assert(pMNEntry->motion_node_id == motNodeID);

	return pMNEntry;
}

/*
 * Retrieve the chunk-sorter entry for the specified motion-node/source pair.
 * If one doesn't exist, it is created and initialized.
 *
 * It might not seem obvious why the MotionNodeEntry is required as an
 * argument.  It's in there to ensure that only valid motion nodes are
 * represented in the chunk-sorter.  The motion-node's entry is typically
 * retrieved using getMotionNodeEntry() before this function is called.
 */
ChunkSorterEntry *
getChunkSorterEntry(MotionLayerState *mlStates,
					MotionNodeEntry *motNodeEntry,
					int16 srcRoute)
{
	MemoryContext oldCtxt;
	ChunkSorterEntry *chunkSorterEntry = NULL;

	AssertArg(motNodeEntry != NULL);

	Assert(srcRoute >= 0);
	Assert(srcRoute < motNodeEntry->num_senders);

	/* Do we have a sorter initialized ? */
	if (motNodeEntry->ready_tuple_lists != NULL)
	{
		if (motNodeEntry->ready_tuple_lists[srcRoute].init)
			return &motNodeEntry->ready_tuple_lists[srcRoute];
	}

	/* We have to create an entry */
	oldCtxt = MemoryContextSwitchTo(mlStates->motion_layer_mctx);

	if (motNodeEntry->ready_tuple_lists == NULL)
		motNodeEntry->ready_tuple_lists = (ChunkSorterEntry *) palloc0(motNodeEntry->num_senders * sizeof(ChunkSorterEntry));

	chunkSorterEntry = &motNodeEntry->ready_tuple_lists[srcRoute];

	if (chunkSorterEntry == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("could not allocate entry for tuple chunk sorter")));

	chunkSorterEntry->chunk_list.serialized_data_length = 0;
	chunkSorterEntry->chunk_list.max_chunk_length = Gp_max_tuple_chunk_size;
	chunkSorterEntry->chunk_list.num_chunks = 0;
	chunkSorterEntry->chunk_list.p_first = NULL;
	chunkSorterEntry->chunk_list.p_last = NULL;
	chunkSorterEntry->end_of_stream = false;
	chunkSorterEntry->init = true;

	/*
	 * If motion node is not order-preserving, then all chunk-sorter entries
	 * share one global tuple-store.  If motion node is order- preserving then
	 * there is a tuple-store per sender per motion node.
	 */
	if (motNodeEntry->preserve_order)
	{
		chunkSorterEntry->ready_tuples = htfifo_create();

#ifdef AMS_VERBOSE_LOGGING
		elog(DEBUG5, "Motion node %d is order-preserving.  Creating tuple-store for entry [src=%d,mn=%d].",
			 motNodeEntry->motion_node_id,
			 srcRoute, motNodeEntry->motion_node_id);
#endif
	}
	else
	{
		chunkSorterEntry->ready_tuples = motNodeEntry->ready_tuples;

#ifdef AMS_VERBOSE_LOGGING
		elog(DEBUG5, "Motion node %d is not order-preserving.  Using shared tuple-store for entry [src=%d,mn=%d].",
			 motNodeEntry->motion_node_id,
			 srcRoute, motNodeEntry->motion_node_id);
#endif

		/* Sanity-check: */
		Assert(motNodeEntry->ready_tuples != NULL);
	}

	MemoryContextSwitchTo(oldCtxt);

	Assert(chunkSorterEntry != NULL);
	return chunkSorterEntry;
}

/*
 * Helper function for converting chunks from the receive state (where
 * they point into a share buffer) into the transient state (where they
 * have their own storage). We need to do this if we didn't receive enough
 * information in one chunk to reconstruct a tuple.
 */
static void
materializeChunk(TupleChunkListItem *tcItem)
{
	TupleChunkListItem newItem;

	/*
	 * This chunk needs to be converted from pointing to global receive buffer
	 * store to having its own storage
	 */
	Assert(tcItem != NULL);
	Assert(*tcItem != NULL);

	newItem = repalloc(*tcItem, sizeof(TupleChunkListItemData) + (*tcItem)->chunk_length);

	memcpy(newItem->chunk_data, newItem->inplace, newItem->chunk_length);
	newItem->inplace = NULL;	/* no need to free, someone else owns it */

	*tcItem = newItem;

	return;
}

/*
 * Add another tuple-chunk to the chunk sorter.  If the new chunk
 * completes another HeapTuple, that tuple will be deserialized and
 * stored into a htfifo.  If not, the chunk is added to the
 * appropriate list of chunks.
 */
static void
addChunkToSorter(MotionLayerState *mlStates,
				 ChunkTransportState *transportStates,
				 MotionNodeEntry *pMNEntry,
				 TupleChunkListItem tcItem,
				 int16 motNodeID,
				 ChunkSorterEntry *chunkSorterEntry,
				 ChunkTransportStateEntry *pEntry,
				 MotionConn *conn,
				 int16 srcRoute)
{
	TupleChunkType tcType;

	AssertArg(tcItem != NULL);

	/* Look at the chunk's type, to figure out what to do with it. */
	GetChunkType(tcItem, &tcType);

	switch (tcType)
	{
		case TC_WHOLE:
		case TC_EMPTY:
			/* There shouldn't be any partial tuple data in the list! */
			if (chunkSorterEntry->chunk_list.num_chunks != 0)
			{
				ereport(ERROR,
						(errcode(ERRCODE_GP_INTERCONNECTION_ERROR),
						 errmsg("received TC_WHOLE chunk from [src=%d,mn=%d] after partial tuple data",
								srcRoute, motNodeID)));
			}

			/* Put this chunk into the list, then turn it into a HeapTuple! */
			appendChunkToTCList(&chunkSorterEntry->chunk_list, tcItem);
			reconstructTuple(pMNEntry, chunkSorterEntry, conn->remapper);

			break;

		case TC_PARTIAL_START:

			/* There shouldn't be any partial tuple data in the list! */
			if (chunkSorterEntry->chunk_list.num_chunks != 0)
			{
				ereport(ERROR,
						(errcode(ERRCODE_GP_INTERCONNECTION_ERROR),
						 errmsg("received TC_PARTIAL_START chunk from [src=%d,mn=%d] after partial tuple data",
								srcRoute, motNodeID)));
			}

			/*
			 * we don't have enough to reconstruct the tuple, we need to copy
			 * the chunk data out of our shared buffer
			 */
			materializeChunk(&tcItem);

			/* Put this chunk into the list. */
			appendChunkToTCList(&chunkSorterEntry->chunk_list, tcItem);

			break;

		case TC_PARTIAL_MID:

			/* There should be partial tuple data in the list. */
			if (chunkSorterEntry->chunk_list.num_chunks <= 0)
			{
				ereport(ERROR,
						(errcode(ERRCODE_GP_INTERCONNECTION_ERROR),
						 errmsg("received TC_PARTIAL_MID chunk from [src=%d,mn=%d] without any leading tuple data",
								srcRoute, motNodeID)));
			}

			/*
			 * we don't have enough to reconstruct the tuple, we need to copy
			 * the chunk data out of our shared buffer
			 */
			materializeChunk(&tcItem);

			/* Append this chunk to the list. */
			appendChunkToTCList(&chunkSorterEntry->chunk_list, tcItem);

			break;

		case TC_PARTIAL_END:

			/* There should be partial tuple data in the list. */
			if (chunkSorterEntry->chunk_list.num_chunks <= 0)
			{
				ereport(ERROR,
						(errcode(ERRCODE_GP_INTERCONNECTION_ERROR),
						 errmsg("received TC_PARTIAL_END chunk from [src=%d,mn=%d] without any leading tuple data",
								srcRoute, motNodeID)));
			}

			/* Put this chunk into the list, then turn it into a HeapTuple! */
			appendChunkToTCList(&chunkSorterEntry->chunk_list, tcItem);
			reconstructTuple(pMNEntry, chunkSorterEntry, conn->remapper);

			break;

		case TC_BATCH_START:

			if (chunkSorterEntry->chunk_list.num_chunks != 0)
			{
				ereport(ERROR,
						(errcode(ERRCODE_GP_INTERCONNECTION_ERROR),
						 errmsg("received TC_BATCH_START chunk from [src=%d,mn=%d] after partial data",
								srcRoute, motNodeID)));
			}

			materializeChunk(&tcItem);
			appendChunkToTCList(&chunkSorterEntry->chunk_list, tcItem);

			break;

		case TC_BATCH_MID:

			if (chunkSorterEntry->chunk_list.num_chunks <= 0)
			{
				ereport(ERROR,
						(errcode(ERRCODE_GP_INTERCONNECTION_ERROR),
						 errmsg("received TC_BATCH_MID chunk from [src=%d,mn=%d] without leading batch data",
								srcRoute, motNodeID)));
			}

			materializeChunk(&tcItem);
			appendChunkToTCList(&chunkSorterEntry->chunk_list, tcItem);

			break;

		case TC_BATCH_END:

			/*
			 * A single-chunk batch arrives as a lone END carrying the
			 * whole payload.
			 */
			if (chunkSorterEntry->chunk_list.num_chunks <= 0 &&
				tcItem->chunk_length > TUPLE_CHUNK_HEADER_SIZE)
			{
				materializeChunk(&tcItem);
				appendChunkToTCList(&chunkSorterEntry->chunk_list, tcItem);
				reconstructBatch(mlStates, pMNEntry, chunkSorterEntry);
				break;
			}

			if (chunkSorterEntry->chunk_list.num_chunks <= 0)
			{
				ereport(ERROR,
						(errcode(ERRCODE_GP_INTERCONNECTION_ERROR),
						 errmsg("received TC_BATCH_END chunk from [src=%d,mn=%d] without leading batch data",
								srcRoute, motNodeID)));
			}

			appendChunkToTCList(&chunkSorterEntry->chunk_list, tcItem);
			reconstructBatch(mlStates, pMNEntry, chunkSorterEntry);

			break;

		case TC_END_OF_STREAM:
#ifdef AMS_VERBOSE_LOGGING
			elog(LOG, "Got end-of-stream. motnode %d route %d", motNodeID, srcRoute);
#endif
			/* There shouldn't be any partial tuple data in the list! */
			if (chunkSorterEntry->chunk_list.num_chunks != 0)
			{
				ereport(ERROR,
						(errcode(ERRCODE_GP_INTERCONNECTION_ERROR),
						 errmsg("received TC_END_OF_STREAM chunk from [src=%d,mn=%d] after partial tuple data",
								srcRoute, motNodeID)));
			}

			/* Make sure that we haven't already received end-of-stream! */

			if (chunkSorterEntry->end_of_stream)
			{
				ereport(ERROR,
						(errcode(ERRCODE_GP_INTERCONNECTION_ERROR),
						 errmsg("received end-of-stream chunk from [src=%d,mn=%d] when already marked as at end-of-stream",
								srcRoute, motNodeID)));
			}

			/* Mark the state as "end of stream." */
			chunkSorterEntry->end_of_stream = true;
			pMNEntry->num_stream_ends_recvd++;

			if (pMNEntry->num_stream_ends_recvd == pMNEntry->num_senders)
				pMNEntry->moreNetWork = false;

			/*
			 * Since we received an end-of-stream.	Then we no longer need
			 * read interest in the interconnect.
			 */
			DeregisterReadInterest(transportStates, motNodeID, srcRoute,
								   "end of stream");
			break;

		default:
			ereport(ERROR,
					(errcode(ERRCODE_GP_INTERCONNECTION_ERROR),
					 errmsg("received tuple chunk of unrecognized type %d (len %d) from [src=%d,mn=%d]",
							tcType, tcItem->chunk_length, srcRoute, motNodeID)));
	}
}



/*
 * STATISTICS HELPER-FUNCTIONS
 *
 * NOTE: the only fields that are required to be valid are
 * tcList->num_chunks and tcList->serialized_data_length, and
 * SerializeTupleDirect() only fills those fields out.
 */
static void
statSendTuple(MotionLayerState *mlStates, MotionNodeEntry *pMNEntry, TupleChunkList tcList)
{
	int			headerOverhead;

	AssertArg(pMNEntry != NULL);

	headerOverhead = TUPLE_CHUNK_HEADER_SIZE * tcList->num_chunks;

	/* per motion-node stats. */
	pMNEntry->stat_total_sends++;
	pMNEntry->stat_total_chunks_sent += tcList->num_chunks;
	pMNEntry->stat_total_bytes_sent += tcList->serialized_data_length + headerOverhead;
	pMNEntry->stat_tuple_bytes_sent += tcList->serialized_data_length;

	/* Update global motion-layer statistics. */
	mlStates->stat_total_chunks_sent += tcList->num_chunks;

	mlStates->stat_total_bytes_sent +=
		tcList->serialized_data_length + headerOverhead;

	mlStates->stat_tuple_bytes_sent += tcList->serialized_data_length;

}

static void
statSendEOS(MotionLayerState *mlStates, MotionNodeEntry *pMNEntry)
{
	AssertArg(pMNEntry != NULL);

	/* Update motion node statistics. */
	pMNEntry->stat_total_chunks_sent++;
	pMNEntry->stat_total_bytes_sent += TUPLE_CHUNK_HEADER_SIZE;

	/* Update global motion-layer statistics. */
	mlStates->stat_total_chunks_sent++;
	mlStates->stat_total_bytes_sent += TUPLE_CHUNK_HEADER_SIZE;
}

static void
statChunksProcessed(MotionLayerState *mlStates, MotionNodeEntry *pMNEntry, int chunksProcessed, int chunkBytes, int tupleBytes)
{
	AssertArg(chunksProcessed >= 0);
	AssertArg(chunkBytes >= 0);
	AssertArg(tupleBytes >= 0);

	/* Update Global Motion Layer Stats. */
	mlStates->stat_total_chunks_recvd += chunksProcessed;
	mlStates->stat_total_chunkproc_calls++;
	mlStates->stat_total_bytes_recvd += chunkBytes;
	mlStates->stat_tuple_bytes_recvd += tupleBytes;

	/* Update Motion-node stats. */
	pMNEntry->stat_total_chunks_recvd += chunksProcessed;
	pMNEntry->stat_total_bytes_recvd += chunkBytes;
	pMNEntry->stat_tuple_bytes_recvd += tupleBytes;
}

static void
statNewTupleArrived(MotionNodeEntry *pMNEntry, ChunkSorterEntry *pCSEntry)
{
	uint32		tupsAvail;

	AssertArg(pMNEntry != NULL);
	AssertArg(pCSEntry != NULL);

	/*
	 * High-watermarks:  We track the number of tuples available to receive,
	 * but that haven't yet been received.  The high-watermark is recorded.
	 *
	 * Also, if the motion node is order-preserving, we track a per-sender
	 * high-watermark as well.
	 */
	tupsAvail = ++(pMNEntry->stat_tuples_available);
	if (pMNEntry->stat_tuples_available_hwm < tupsAvail)
	{
		/* New high-watermark! */
		pMNEntry->stat_tuples_available_hwm = tupsAvail;
	}
}

static void
statRecvTuple(MotionNodeEntry *pMNEntry, ChunkSorterEntry *pCSEntry)
{
	AssertArg(pMNEntry != NULL);
	AssertArg(pCSEntry != NULL || !pMNEntry->preserve_order);

	/* Count tuples received. */
	pMNEntry->stat_total_recvs++;

	/* Update "tuples available" counts for high watermark stats. */
	pMNEntry->stat_tuples_available--;
}

/*
 * Return true if the record cache should be sent to coordinator
 */
static bool
ShouldSendRecordCache(MotionConn *conn, SerTupInfo *pSerInfo)
{
	return pSerInfo->has_record_types &&
		NextRecordTypmod > 0 &&
		NextRecordTypmod > conn->sent_record_typmod;
}

/*
 * Update the number of sent record types.
 */
static void
UpdateSentRecordCache(MotionConn *conn)
{
	conn->sent_record_typmod = NextRecordTypmod;
}
