#!/usr/bin/env python3
"""
Vector DB Utilities - Qdrant-only helpers for knowledge base storage and updates.
- Client creation and collection management
- Payload indexing
- Stable ID generation and content hashing
- Batched upserts and optional source replacement
- Public ingest function to store embedded chunks
"""

import datetime
import hashlib
from typing import List, Dict, Any

import numpy as np

# Qdrant imports and setup
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        HnswConfigDiff,
        OptimizersConfigDiff,
        PointStruct,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    raise ImportError("qdrant-client is required. Install with: pip install qdrant-client")


# -----------------------------
# Client / Collection
# -----------------------------

def get_qdrant_client(config: Dict[str, Any]) -> QdrantClient:
    """Create Qdrant client with required configuration.
    
    Args:
        config: Dictionary containing Qdrant connection configuration
        Required keys: url, timeout
        Optional keys: api_key
    """
    if config is None:
        raise ValueError("Qdrant configuration is required")
    
    client = QdrantClient(
        url=config["url"],
        api_key=config.get("api_key"),
        timeout=config.get("timeout", 60.0),
        verify=False,
        check_compatibility=False
    )
    return client


def ensure_qdrant_collection(client: QdrantClient, 
                           collection_name: str, 
                           vector_size: int,
                           config: Dict[str, Any]):
    """Create the collection if it doesn't exist with required configuration.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create
        vector_size: Dimension of the vectors
        config: Dictionary containing collection configuration
        Required keys: distance, on_disk
        Optional keys: hnsw_config, optimizers_config
    """
    from qdrant_client.http.exceptions import UnexpectedResponse

    if config is None:
        raise ValueError("Collection configuration is required")

    try:
        exists = client.collection_exists(collection_name=collection_name)
    except UnexpectedResponse:
        exists = False

    if exists:
        return

    # Get distance metric
    distance_str = config.get("distance", "COSINE").upper()
    distance = getattr(Distance, distance_str, Distance.COSINE)

    # Create collection with provided parameters
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance,
            on_disk=config.get("on_disk", True),
        ),
        hnsw_config=HnswConfigDiff(
            m=config.get("hnsw_config", {}).get("m", 32),
            ef_construct=config.get("hnsw_config", {}).get("ef_construct", 256),
            full_scan_threshold=config.get("hnsw_config", {}).get("full_scan_threshold", 10000),
        ),
        optimizers_config=OptimizersConfigDiff(
            memmap_threshold=config.get("optimizers_config", {}).get("memmap_threshold", 20000),
            indexing_threshold=config.get("optimizers_config", {}).get("indexing_threshold", 10000),
            flush_interval_sec=config.get("optimizers_config", {}).get("flush_interval_sec", 5),
        ),
    )


def ensure_payload_indexes(client: QdrantClient, collection_name: str):
    """Create payload indexes commonly used for KB filtering and updates."""
    from qdrant_client.models import PayloadSchemaType
    try:
        client.create_payload_index(collection_name, field_name="source_id", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, field_name="content_hash", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, field_name="tags", field_schema=PayloadSchemaType.KEYWORD)
    except Exception:
        # Index may already exist; ignore
        pass

# -----------------------------
# Helpers
# -----------------------------

def _l2_normalize(vector: List[float]) -> List[float]:
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_point_id(source_id: str, chunk_id: str, content_hash: str) -> int:
    combined = f"{source_id}|{chunk_id}|{content_hash}"
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _build_points_from_chunks(embedded_chunks: List[Dict[str, Any]]) -> List["PointStruct"]:
    points: List[PointStruct] = []
    for chunk in embedded_chunks:
        vector = chunk.get('embedding') or []
        norm_vector = _l2_normalize(vector)

        metadata = chunk.get("metadata", {})
        source_id = metadata.get("source_id") or metadata.get("source_url") or "unknown_source"
        chunk_id = chunk.get("id") or _sha256(chunk.get("content", ""))[:12]
        content = chunk.get("content", "")
        content_hash = _sha256(content)
        version = metadata.get("version", 1)
        tags = metadata.get("tags") or []

        payload = {
            "original_id": chunk.get("id"),
            "source_id": source_id,
            "chunk_id": chunk_id,
            "content_hash": content_hash,
            "version": version,
            "title": metadata.get("title"),
            "source_url": metadata.get("source_url"),
            "tokens": metadata.get("tokens"),
            "embedding_model": metadata.get("embedding_model"),
            "embedding_dimension": metadata.get("embedding_dimension"),
            "tags": tags,
            "content": content,
            "ingested_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

        pid = _stable_point_id(source_id, chunk_id, content_hash)
        points.append(PointStruct(id=pid, vector=norm_vector, payload=payload))
    return points


def _upsert_points_in_batches(client: QdrantClient, collection_name: str, points: List["PointStruct"], batch_size: int):
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=collection_name, points=batch, wait=True)


def _replace_source(client: QdrantClient, collection_name: str, source_id: str, keep_hashes: set):
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    existing_ids: List[int] = []
    next_page = None
    while True:
        res = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]),
            with_payload=True,
            with_vectors=False,
            limit=1000,
            offset=next_page,
        )
        points_page, next_page = res[0], res[1]
        for p in points_page:
            phash = (p.payload or {}).get("content_hash")
            if phash and phash not in keep_hashes:
                existing_ids.append(p.id)
        if next_page is None:
            break

    if existing_ids:
        client.delete(collection_name=collection_name, points_selector=existing_ids, wait=True)

# -----------------------------
# Public API
# -----------------------------

def ingest_to_qdrant(embedded_chunks: List[Dict[str, Any]], 
                    *,
                    collection_name: str, 
                    replace_source: bool, 
                    batch_size: int,
                    qdrant_config: Dict[str, Any]) -> None:
    """Idempotently upsert embedded chunks into Qdrant with required configuration.

    Args:
        embedded_chunks: List of chunks with 'embedding' and 'metadata'.
        collection_name: Name of the collection to ingest into.
        replace_source: If True, remove outdated chunks per source.
        batch_size: Upsert batch size.
        qdrant_config: Complete Qdrant configuration (required).
    """
    if not embedded_chunks:
        return

    if qdrant_config is None:
        raise ValueError("Qdrant configuration is required")

    client = get_qdrant_client(qdrant_config)
    vector_size = int(embedded_chunks[0]["metadata"]["embedding_dimension"]) if embedded_chunks else 0
    ensure_qdrant_collection(client, collection_name, vector_size, qdrant_config)
    ensure_payload_indexes(client, collection_name)

    points = _build_points_from_chunks(embedded_chunks)

    if replace_source:
        from collections import defaultdict
        source_to_hashes: Dict[str, set] = defaultdict(set)
        for p in points:
            src = (p.payload or {}).get("source_id", "unknown_source")
            phash = (p.payload or {}).get("content_hash")
            if phash:
                source_to_hashes[src].add(phash)
        for src, hashes in source_to_hashes.items():
            _replace_source(client, collection_name, src, hashes)

    _upsert_points_in_batches(client, collection_name, points, batch_size) 