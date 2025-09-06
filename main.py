#!/usr/bin/env python3
"""
Simple demo script that scrapes a hardcoded query and processes the data
through the complete pipeline: scrape → file → chunk → embed
"""

from search_tool import update_kb_from_query
import time

# Hardcoded query to scrape
SEARCH_QUERY = "Machine Learning Algorithms"

# Qdrant Configuration
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
    """Main function: Scrape hardcoded query and process through complete pipeline"""
    print("🔬 KNOWLEDGE BASE PIPELINE DEMO")
    print("="*60)
    print(f"Query: '{SEARCH_QUERY}'")
    print("Pipeline: Scrape → File → Chunk → Embed → Database")
    print("="*60)
    
    start_time = time.time()
    
    # Track background processing completion
    background_completed = False
    background_result = None
    
    def background_callback(result):
        nonlocal background_completed, background_result
        background_completed = True
        background_result = result
        
        if result.get('success'):
            print(f"\n✅ Background processing completed!")
            print(f"   📊 Results processed: {result.get('num_results', 0)}")
            print(f"   📦 Chunks created: {result.get('num_chunks', 0)}")
            print(f"   💾 Chunks file: {result.get('chunks_file', 'N/A')}")
            print(f"   💾 Embeddings file: {result.get('embedded_file', 'N/A')}")
            
            # Show timing breakdown
            timings = result.get('timings_sec', {})
            if timings:
                print(f"   ⏱️  Chunking: {timings.get('chunk', 0):.2f}s")
                print(f"   ⏱️  Embedding: {timings.get('embed', 0):.2f}s")
                print(f"   ⏱️  Database: {timings.get('ingest', 0):.2f}s")
                print(f"   ⏱️  Total: {timings.get('total', 0):.2f}s")
        else:
            print(f"❌ Background processing failed: {result.get('error', 'Unknown error')}")
    
    try:
        print(f"\n🚀 Starting search and processing for '{SEARCH_QUERY}'...")
        
        # Execute the complete pipeline
        immediate_result = update_kb_from_query(
            query=SEARCH_QUERY,
            min_words=150,
            model_name="BAAI/bge-large-en-v1.5",
            collection_name="gradus-kb",
            save_results=True,      # Save scraped data to file
            save_chunks=True,       # Save chunks to file
            save_embedded=True,     # Save embeddings to file
            output_dir="outputs",
            background_callback=background_callback,
            qdrant_config=QDRANT_CONFIG
        )
        
        immediate_time = time.time() - start_time
        
        # Show immediate results
        if immediate_result.get('success'):
            print(f"\n✅ Search completed in {immediate_time:.2f}s!")
            print(f"   📊 URLs processed: {immediate_result.get('total_urls', 0)}")
            print(f"   ✅ Successful scrapes: {immediate_result.get('successful_scrapes', 0)}")
            print(f"   ⏱️  Search time: {immediate_result.get('scraping_time', 0):.2f}s")
            print(f"   💾 Results file: {immediate_result.get('results_file', 'N/A')}")
            print(f"   🔄 Background processing: {immediate_result.get('background_processing', False)}")
            
            # Show sample scraped content
            if immediate_result.get('results'):
                print(f"\n📄 Scraped Content Sample:")
                for i, result in enumerate(immediate_result['results'][:3], 1):
                    print(f"   {i}. {result['url']}")
                    print(f"      Words: {result.get('word_count', 0)} | Length: {result.get('length', 0)} chars")
                    print(f"      Preview: {result['content'][:150]}...")
        else:
            print(f"❌ Search failed: {immediate_result.get('error', 'Unknown error')}")
            return
        
        # Wait for background processing to complete
        if immediate_result.get('background_processing'):
            print(f"\n⏳ Waiting for background processing (chunking → embedding → database)...")
            timeout = 120  # 2 minutes timeout
            wait_start = time.time()
            
            while not background_completed and (time.time() - wait_start) < timeout:
                time.sleep(1)
                print(".", end="", flush=True)
            
            total_time = time.time() - start_time
            
            if background_completed and background_result:
                print(f"\n✅ Complete pipeline finished in {total_time:.2f}s!")
            else:
                print(f"\n⏰ Background processing timeout after {timeout}s")
        
        print("\n" + "="*60)
        print(" PIPELINE COMPLETE")
        print("="*60)
        print("✅ Scraping: Data collected and saved to file")
        print("✅ Chunking: Text split into chunks and saved to file")
        print("✅ Embedding: Chunks converted to vectors and saved to file")
        print("✅ Database: Vectors stored in Qdrant")
        print("📁 Check the 'outputs' directory for all saved files")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()