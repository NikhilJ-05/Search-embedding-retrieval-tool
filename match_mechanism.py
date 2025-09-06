#!/usr/bin/env python3
"""
Semantic search over the knowledge base stored in Qdrant.
- Embeds the query with the same model used for ingestion
- Normalizes vectors (COSINE) and searches the configured collection
- Returns top matches with title, url, snippet, score, and published_date
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time

from embedding_tool import get_embedding_model
from vector_DB import get_qdrant_client

DEFAULT_QUERY = "MCP"


def _l2_normalize(vector: List[float]) -> List[float]:
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def embed_query(text: str, model_name: str = "BAAI/bge-large-en-v1.5") -> List[float]:
    """Embed a single query string using the configured SentenceTransformer model."""
    model = get_embedding_model(model_name)
    vec = model.encode([text], convert_to_tensor=False, show_progress_bar=False)[0]
    if isinstance(vec, np.ndarray):
        vec = vec.tolist()
    return _l2_normalize(vec)


def search_kb(query: str,
             *,
             top_k: int = 5,
             model_name: str = "BAAI/bge-large-en-v1.5",
             collection_name: str,
             qdrant_config: Dict[str, Any],
             filter_payload: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Semantic search in Qdrant.

    Args:
        query: Natural language query
        top_k: Number of results to return
        model_name: Embedding model name used for query
        collection_name: Name of the collection to search
        qdrant_config: Qdrant configuration dictionary (required)
        filter_payload: Optional exact-match payload filters, e.g., {"source_id": "foo"}

    Returns:
        List of search results with fields: score, title, source_url, snippet, chunk_id, source_id, published_date
    """
    if qdrant_config is None:
        raise ValueError("Qdrant configuration is required")
    
    client = get_qdrant_client(qdrant_config)
    collection = collection_name

    # Embed and normalize query vector
    query_vector = embed_query(query, model_name=model_name)

    # Build optional filter
    q_filter = None
    if filter_payload:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        must = []
        for key, value in filter_payload.items():
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        q_filter = Filter(must=must)

    # Execute semantic query (replacement for deprecated `search`)
    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=q_filter,
    )

    # Handle different response formats from Qdrant client
    if hasattr(response, 'points'):
        hits = response.points
    elif isinstance(response, tuple):
        hits = response[0] if len(response) > 0 else []
    else:
        hits = response

    results: List[Dict[str, Any]] = []
    for point in hits:
        # Handle both tuple and object formats
        if isinstance(point, tuple):
            # If point is a tuple, extract id, score, and payload
            point_id, point_score, point_payload = point[0], point[1], point[2] if len(point) > 2 else {}
        else:
            # If point is an object, access attributes normally
            point_id = point.id
            point_score = point.score
            point_payload = point.payload or {}
        
        payload = point_payload or {}
        content = payload.get("content", "")
        # Updated to match new chunk structure - no title field, use source_url for display
        source_url = payload.get("source_url")
        title = source_url or payload.get("source_id") or "Untitled"
        published_date = payload.get("published_date")
        snippet = content[:200] + ("..." if len(content) > 200 else "")
        results.append({
            "score": float(point_score),
            "id": point_id,
            "title": title,
            "source_url": source_url,
            "published_date": published_date,
            "snippet": snippet,
            "chunk_id": payload.get("chunk_id"),
            "source_id": payload.get("source_id"),
        })

    return results


def search_kb_with_timing(query: str,
                          *,
                          top_k: int = 5,
                          model_name: str = "BAAI/bge-large-en-v1.5",
                          collection_name: str,
                          qdrant_config: Dict[str, Any],
                          filter_payload: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], float]:
    """Run search_kb and return (results, elapsed_seconds)."""
    t0 = time.perf_counter()
    results = search_kb(query, top_k=top_k, model_name=model_name, collection_name=collection_name, qdrant_config=qdrant_config, filter_payload=filter_payload)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def format_results(results: List[Dict[str, Any]]) -> str:
    """Pretty-print search results similar to a simple web search listing."""
    lines: List[str] = []
    for i, r in enumerate(results, 1):
        pub = f" | published: {r.get('published_date')}" if r.get('published_date') else ""
        lines.append(f"{i}. {r.get('title')}  (score={r.get('score'):.4f}){pub}")
        if r.get('source_url'):
            lines.append(f"   URL: {r.get('source_url')}")
        lines.append(f"   {r.get('snippet')}")
    return "\n".join(lines)


def demo_semantic_search(query: str, qdrant_config: Dict[str, Any], collection_name: str = "gradus-kb", top_k: int = 5):
    """Demo function to perform semantic search with the updated structure."""
    print(f"🔍 Semantic Search Demo: '{query}'")
    print("="*60)
    
    try:
        results, took = search_kb_with_timing(
            query=query,
            top_k=top_k,
            collection_name=collection_name,
            qdrant_config=qdrant_config,
            model_name="BAAI/bge-large-en-v1.5"
        )
        
        if results:
            print(f"✅ Found {len(results)} results in {took:.3f}s")
            print("\n📄 Search Results:")
            print(format_results(results))
        else:
            print("❌ No results found")
            
        return results, took
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return [], 0.0


if __name__ == "__main__":
    print("🔧 Match Mechanism - Library Usage")
    print("This is a library module. Use match_demo.py for demonstrations.")
    print("Example usage:")
    print("  from match_mechanism import search_kb, demo_semantic_search")
    print("  results = search_kb(query, collection_name, qdrant_config)")
    print("  demo_semantic_search(query, qdrant_config, collection_name)")
