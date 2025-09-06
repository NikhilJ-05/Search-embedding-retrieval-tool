#!/usr/bin/env python3
"""
Orion Crawler Demo - Simple demo to test scraping capabilities.
Shows: hardcoded query → filtered results → timing
"""

import sys
import os
import time
from datetime import datetime

# Add the Knowledge Base directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Knowledge Base'))

from orion_crawler import search_and_scrape

def main():
    """Simple demo function with file output."""
    # Create output content
    output_lines = []
    
    def add_output(text):
        """Add text to both console and output buffer."""
        print(text)
        output_lines.append(text)
    
    add_output("="*60)
    add_output(" ORION CRAWLER DEMO")
    add_output("="*60)
    
    # Hardcoded search query
    SEARCH_QUERY = "Model Context protocol"
    add_output(f"🔍 Search Query: '{SEARCH_QUERY}'")
    add_output("⏳ Starting search and scrape process...")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Execute the search and scrape
        results = search_and_scrape(
            query=SEARCH_QUERY,
            max_concurrent=15,
            timeout=15.0
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        add_output("\n" + "="*60)
        add_output(" RESULTS")
        add_output("="*60)
        
        if results['success']:
            add_output(f"✅ Search completed successfully!")
            add_output(f"⏱️  Total time: {elapsed_time:.2f} seconds")
            add_output(f"🔗 URLs processed: {results['total_urls']}")
            add_output(f"✅ Successful scrapes: {results['successful_scrapes']}")
            
            if 'processing_time' in results:
                add_output(f"⚡ Processing time: {results['processing_time']:.2f} seconds")
            
            if results['results']:
                add_output(f"\n📄 Filtered Results ({len(results['results'])}):")
                add_output("-" * 60)
                
                for i, result in enumerate(results['results'], 1):
                    add_output(f"\n{i}. {result['url']}")
                    add_output(f"   Length: {result['length']} characters")
                    add_output(f"   Words: {result['word_count']} words")
                    add_output(f"   Content preview: {result['content'][:150]}...")
                    
                    # Add full content for detailed review
                    add_output(f"\n   Full Content:")
                    add_output("-" * 40)
                    add_output(result['content'])
                    add_output("-" * 40)
            else:
                add_output("⚠️  No content was successfully scraped")
        else:
            add_output(f"❌ Search failed: {results['error']}")
            
    except Exception as e:
        add_output(f"💥 Demo failed with error: {e}")
    
    add_output("\n" + "="*60)
    add_output(" DEMO COMPLETE")
    add_output("="*60)
    
    # Write output to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"orion_crawler_demo_output_{timestamp}.txt"
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\n💾 Output saved to: {output_filename}")
    except Exception as e:
        print(f"\n❌ Error saving output file: {e}")

if __name__ == "__main__":
    main()