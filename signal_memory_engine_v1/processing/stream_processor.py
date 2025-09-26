# processing/stream_processor.py
# Work in progress: this module will continuously read raw events from a source, and send it to normalizer.py to be cleaned and stored.
import logging
import time
from collections.abc import Iterator
from typing import Any

from normalizer import normalize_event

from vector_store.pinecone_index import upsert_batch  # your helper to upsert into Pinecone

logger = logging.getLogger(__name__)


def ingest_event_stream(
    source: Iterator[dict[str, Any]], batch_size: int = 50, interval: float = 1.0
):
    """
    Continuously read raw events from `source` (could be Kafka consumer, webhook queue, etc.),
    normalize them, and batch‐upsert into Pinecone.
    """
    batch = []
    for raw in source:
        try:
            evt = normalize_event(raw)
            batch.append(evt)
        except Exception as e:
            logger.warning("Failed to normalize %s: %s", raw, e)
            continue

        if len(batch) >= batch_size:
            _flush_batch(batch)
            batch = []

        time.sleep(interval)  # simple rate‐limit

    # final flush
    if batch:
        _flush_batch(batch)


def _flush_batch(batch: list[dict[str, Any]]):
    """
    Upsert a batch of normalized events into your vector store.
    Assumes each event has at least `text` and `timestamp`.
    """
    # Build upsert payloads: [{ "id": ..., "vector": [...], "metadata": {...} }, ...]
    try:
        upsert_batch(batch)
        logger.info("Upserted %d events", len(batch))
    except Exception as e:
        logger.error("Error upserting batch: %s", e)
