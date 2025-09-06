#!/usr/bin/env python3
"""
Simple demo script for match_mechanism.py library
Performs a single semantic search query and returns results with timing
"""

from match_mechanism import search_kb_with_timing

# Hardcoded query to search
SEARCH_QUERY = "Machine Learning"

# Qdrant Configuration (same as main.py)
QDRANT_CONFIG = {
    "url": "http://localhost:9500",
    "api_key": "NGrstbRdgFWbDoIZVrOT8xzxOZtY01gq",
    "timeout": 60.0,
    "collection": "gradus-kb",
    "replace_source_on_ingest": False,
    "upsert_batch_size": 512,
    "vector_size": 1024,  # BGE-large-en-v1.5 dimension
    "distance": "COSINE",
    "on_disk": True,
    "hnsw_config": {
        "m": 32,
        "ef_construct": 256,
        "full_scan_threshold": 10000
    },
    "optimizers_config": {
        "memmap_threshold": 20000,
        "indexing_threshold": 10000,
        "flush_interval_sec": 5
    }
}


def main():
    """Main function: Perform single semantic search and return results with timing"""
    print("🔍 SEMANTIC SEARCH DEMO")
    print("="*50)
    print(f"Query: '{SEARCH_QUERY}'")
    print("="*50)
    
    try:
        # Perform semantic search with timing
        results, elapsed = search_kb_with_timing(
            query=SEARCH_QUERY,
            collection_name="gradus-kb",
            qdrant_config=QDRANT_CONFIG,
            top_k=5
        )
        
        # Display results
        if results:
            print(f"✅ Found {len(results)} results in {elapsed:.3f}s")
            print("\n📄 Search Results:")
            print("-" * 30)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   URL: {result['source_url']}")
                print(f"   Snippet: {result['snippet']}")
        else:
            print(f"❌ No results found in {elapsed:.3f}s")
        
        print(f"\n⏱️  Search completed in {elapsed:.3f} seconds")
        return results, elapsed
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return [], 0.0


if __name__ == "__main__":
    main()
