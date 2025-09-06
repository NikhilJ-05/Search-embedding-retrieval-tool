#!/usr/bin/env python3
"""
Orchestrator for updating the knowledge base in-memory from a query.
- Single endpoint: returns search results immediately + processes in background
- Optional JSON writes for debugging (no reads)
"""

from orion_crawler import search_and_scrape
from chunking_tool import create_chunks_from_results, print_chunk_verification, save_chunks_to_json
from embedding_tool import process_chunks_with_embeddings
from vector_DB import ingest_to_qdrant
import time
import json
import datetime
import os
import threading
import asyncio


def _background_processing(scraped_data: dict,
                          min_words: int,
                          model_name: str,
                          collection_name: str,
                          replace_source: bool,
                          batch_size: int,
                          save_chunks: bool,
                          save_embedded: bool,
                          output_dir: str,
                          qdrant_config: dict,
                          callback=None):
    """Background processing function for chunking, embedding, and database ingestion."""
    try:
        t_start = time.perf_counter()
        
        # Step 1: Filter and chunk the data
        t0 = time.perf_counter()
        substantial_results = [r for r in scraped_data['results'] if r.get('word_count', 0) >= min_words]
        
        if not substantial_results:
            if callback:
                callback({"success": False, "error": f"No results meet minimum word count of {min_words}"})
            return
        
        chunks = create_chunks_from_results(substantial_results, scraped_data.get('query', ''))
        print_chunk_verification(chunks)
        t_chunk = time.perf_counter() - t0
        
        # Optional: Save chunks
        chunks_file = ""
        if save_chunks:
            try:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                chunks_file = os.path.join(output_dir, f"chunks_{timestamp}.json")
                with open(chunks_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)
            except Exception as e:
                chunks_file = f"Error saving chunks: {e}"
        
        # Step 2: Generate embeddings
        t0 = time.perf_counter()
        embedded_chunks, embedded_file = process_chunks_with_embeddings(
            chunks,
            model_name=model_name,
            save_json=save_embedded,
            save_numpy=False,
            output_dir=output_dir if save_embedded else None
        )
        t_embed = time.perf_counter() - t0
        
        # Step 3: Ingest to vector database
        t0 = time.perf_counter()
        ingest_to_qdrant(embedded_chunks, 
                        collection_name=collection_name or qdrant_config.get("collection", "gradus-kb"), 
                        replace_source=replace_source if replace_source is not None else qdrant_config.get("replace_source_on_ingest", False), 
                        batch_size=batch_size or qdrant_config.get("upsert_batch_size", 512),
                        qdrant_config=qdrant_config)
        t_ingest = time.perf_counter() - t0
        t_total = time.perf_counter() - t_start
        
        result = {
            "success": True,
            "query": scraped_data.get('query', ''),
            "num_results": len(substantial_results),
            "num_chunks": len(chunks),
            "chunks_file": chunks_file,
            "embedded_file": embedded_file,
            "timings_sec": {
                "chunk": round(t_chunk, 3),
                "embed": round(t_embed, 3),
                "ingest": round(t_ingest, 3),
                "total": round(t_total, 3),
            }
        }
        
        if callback:
            callback(result)
            
    except Exception as e:
        if callback:
            callback({"success": False, "error": f"Background processing failed: {str(e)}"})


def update_kb_from_query(query: str,
                         *,
                         min_words: int = 150,
                         model_name: str = "BAAI/bge-large-en-v1.5",
                         collection_name: str = None,
                         replace_source: bool = None,
                         batch_size: int = None,
                         max_concurrent: int = 15,
                         timeout: float = 15.0,
                         save_results: bool = False,
                         save_chunks: bool = False,
                         save_embedded: bool = False,
                         output_dir: str = "outputs",
                         background_callback=None,
                         qdrant_config: dict = None) -> dict:
    """Single endpoint: Returns search results immediately + processes in background.
    
    This function provides the best user experience by:
    1. Immediately returning search results to the user
    2. Processing chunking, embedding, and database ingestion in parallel
    
    Args:
        query (str): Search query to execute
        min_words (int): Minimum word count for filtering
        model_name (str): Embedding model name
        collection_name (str): Qdrant collection name
        replace_source (bool): Whether to replace existing data
        batch_size (int): Batch size for database operations
        max_concurrent (int): Maximum concurrent scraping operations
        timeout (float): Timeout per URL in seconds
        save_results (bool): Whether to save results to JSON file
        save_chunks (bool): Whether to save chunks to JSON
        save_embedded (bool): Whether to save embeddings to JSON
        output_dir (str): Directory for output files
        background_callback (callable): Optional callback for background processing results
        
    Returns:
        dict: Immediate search results with background processing status
    """
    t_start = time.perf_counter()
    
    # Step 1: Execute scraping using orion_crawler
    print(f"🔍 Searching for: '{query}'...")
    results = search_and_scrape(query, max_concurrent=max_concurrent, timeout=timeout)
    
    t_search = time.perf_counter() - t_start
    
    # Add timing information
    results['scraping_time'] = round(t_search, 3)
    results['background_processing'] = False
    
    # Optional: Save results to file
    results_file = ""
    if save_results and results.get('success'):
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(output_dir, f"search_results_{timestamp}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            results['results_file'] = results_file
        except Exception as e:
            results['save_error'] = str(e)
    
    # If scraping was successful, start background processing
    if results.get('success') and results.get('results'):
        print(f"✅ Search completed! Starting background processing...")
        results['background_processing'] = True
        
        # Start background thread for processing
        background_thread = threading.Thread(
            target=_background_processing,
            args=(
                results,
                min_words,
                model_name,
                collection_name,
                replace_source,
                batch_size,
                save_chunks,
                save_embedded,
                output_dir,
                qdrant_config,
                background_callback
            ),
            daemon=True
        )
        background_thread.start()
        
        # Add background processing info to immediate response
        results['background_info'] = {
            "status": "processing",
            "message": "Chunking, embedding, and database ingestion in progress",
            "thread_id": background_thread.ident
        }
    else:
        print(f"❌ Search failed: {results.get('error', 'Unknown error')}")
    
    return results




def extract_timings(summary: dict) -> dict:
    """Return the timings dict from update_kb_from_query summary (empty if missing)."""
    return (summary or {}).get("timings_sec", {})


def format_timings(timings: dict) -> str:
    """Return a human-readable one-line timing summary string."""
    if not timings:
        return ""
    parts = [
        f"search={timings.get('search', 0)}s",
        f"chunk={timings.get('chunk', 0)}s",
        f"embed={timings.get('embed', 0)}s",
        f"ingest={timings.get('ingest', 0)}s",
        f"total={timings.get('total', 0)}s",
    ]
    return " | ".join(parts)


if __name__ == "__main__":
    # Simple test to verify the library works
    print("Testing search_tool.py library...")
    
    # Test the combined approach
    result = update_kb_from_query(
        "Test query",
        min_words=150,
        save_results=False,
        save_chunks=False,
        save_embedded=False
    )
    
    if result.get('success'):
        print("✅ Library test passed!")
        print(f"Results: {result.get('num_results', 0)}")
        print(f"Chunks: {result.get('num_chunks', 0)}")
    else:
        print(f"❌ Library test failed: {result.get('error', 'Unknown error')}")
    
    print("For full demos, run main.py") 