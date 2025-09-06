#!/usr/bin/env python3
"""
Chunking Tool - A utility for converting text content into chunks for vector database embedding.
Uses LangChain RecursiveCharacterTextSplitter for optimal chunking.
"""

import json
import re
import datetime
from typing import List, Dict, Any

# Import LangChain text splitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not available. Install with: pip install langchain")

# Import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️  tiktoken not available. Install with: pip install tiktoken")

def get_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Get accurate token count using tiktoken.
    
    Args:
        text (str): Text to count tokens for
        model (str): Model to use for tokenization
        
    Returns:
        int: Token count
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback to character-based estimation
        return len(text) // 4
    
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"⚠️  Error with tiktoken: {e}. Using fallback.")
        return len(text) // 4

def create_text_splitter():
    """
    Create LangChain RecursiveCharacterTextSplitter with specified parameters.
    
    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for text splitting. Install with: pip install langchain")
    
    # Use tiktoken for accurate token counting if available
    if TIKTOKEN_AVAILABLE:
        try:
            length_function = lambda text: get_token_count(text)
            print("✅ Using tiktoken for accurate token counting")
        except Exception as e:
            print(f"⚠️  tiktoken failed: {e}. Using character length.")
            length_function = len
    else:
        length_function = len
        print("⚠️  tiktoken not available. Using character length for token counting.")
    
    return RecursiveCharacterTextSplitter(
        chunk_size=512,      # Ideal for bge-large-en-v1.5 (≈ 400 words)
        chunk_overlap=50,    # Helps preserve context between chunks
        length_function=length_function, # Use tiktoken if available, otherwise len
        separators=["\n\n", "\n", " ", ""]
    )


def create_chunks_from_results(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Convert search results into chunks for vector database embedding using LangChain.
    
    Args:
        results (List[Dict]): Search results from orion_crawler
        query (str): Original search query
        
    Returns:
        List[Dict]: List of chunks in the specified format
    """
    if not LANGCHAIN_AVAILABLE:
        print("❌ LangChain not available. Cannot create chunks.")
        return []
    
    all_chunks = []
    text_splitter = create_text_splitter()
    
    for i, result in enumerate(results):
        doc_id = f"doc{i+1:03d}"
        url = result.get('url', '')
        content = result.get('content', '')
        
        # Content is already filtered to 150+ words by orion_crawler
        if not content:
            continue
        
        # Use LangChain to chunk the content
        try:
            text_chunks = text_splitter.split_text(content)
        except Exception as e:
            print(f"⚠️  Error chunking content from {url}: {e}")
            continue
        
        for j, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
                
            chunk_id = f"{doc_id}_chunk_{j+1:03d}"
            tokens = get_token_count(chunk_text)
            
            chunk_data = {
                "id": chunk_id,
                "content": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "source_url": url,
                    "published_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "chunk_index": j + 1,
                    "total_chunks": len(text_chunks),
                    "tokens": tokens
                }
            }
            
            all_chunks.append(chunk_data)
    
    return all_chunks

def save_chunks_to_json(chunks: List[Dict[str, Any]], filename: str = None):
    """Save chunks to a JSON file for vector database ingestion."""
    if not filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"orion_chunks_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Chunks saved to: {filename}")
    return filename

def verify_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verify and analyze the chunks being produced.
    
    Args:
        chunks (List[Dict]): List of chunks to verify
        
    Returns:
        Dict: Verification statistics and analysis
    """
    if not chunks:
        return {"error": "No chunks to verify"}
    
    verification = {
        "total_chunks": len(chunks),
        "token_stats": {
            "min": min(chunk['metadata']['tokens'] for chunk in chunks),
            "max": max(chunk['metadata']['tokens'] for chunk in chunks),
            "avg": sum(chunk['metadata']['tokens'] for chunk in chunks) / len(chunks),
            "total": sum(chunk['metadata']['tokens'] for chunk in chunks)
        },
        "content_stats": {
            "min_length": min(len(chunk['content']) for chunk in chunks),
            "max_length": max(len(chunk['content']) for chunk in chunks),
            "avg_length": sum(len(chunk['content']) for chunk in chunks) / len(chunks)
        },
        "structure_validation": {
            "valid_ids": all('id' in chunk for chunk in chunks),
            "valid_content": all('content' in chunk and chunk['content'].strip() for chunk in chunks),
            "valid_metadata": all('metadata' in chunk for chunk in chunks),
            "valid_metadata_fields": all(
                all(field in chunk['metadata'] for field in ['doc_id', 'source_url', 'chunk_index', 'total_chunks', 'tokens'])
                for chunk in chunks
            )
        },
        "sample_chunks": chunks[:3] if len(chunks) >= 3 else chunks
    }
    
    return verification

def print_chunk_verification(chunks: List[Dict[str, Any]]):
    """
    Print detailed verification information about chunks.
    
    Args:
        chunks (List[Dict]): List of chunks to verify
    """
    verification = verify_chunks(chunks)
    
    print("\n" + "="*60)
    print("🔍 CHUNK VERIFICATION REPORT")
    print("="*60)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"Total chunks: {verification['total_chunks']}")
    
    print(f"\n🎯 TOKEN STATISTICS:")
    token_stats = verification['token_stats']
    print(f"Min tokens: {token_stats['min']}")
    print(f"Max tokens: {token_stats['max']}")
    print(f"Average tokens: {token_stats['avg']:.1f}")
    print(f"Total tokens: {token_stats['total']}")
    
    print(f"\n📏 CONTENT STATISTICS:")
    content_stats = verification['content_stats']
    print(f"Min length: {content_stats['min_length']} chars")
    print(f"Max length: {content_stats['max_length']} chars")
    print(f"Average length: {content_stats['avg_length']:.1f} chars")
    
    print(f"\n✅ STRUCTURE VALIDATION:")
    structure = verification['structure_validation']
    print(f"Valid IDs: {'✅' if structure['valid_ids'] else '❌'}")
    print(f"Valid Content: {'✅' if structure['valid_content'] else '❌'}")
    print(f"Valid Metadata: {'✅' if structure['valid_metadata'] else '❌'}")
    print(f"Valid Metadata Fields: {'✅' if structure['valid_metadata_fields'] else '❌'}")
    
    print(f"\n📄 SAMPLE CHUNKS:")
    for i, chunk in enumerate(verification['sample_chunks'], 1):
        print(f"\n--- SAMPLE CHUNK {i} ---")
        print(f"ID: {chunk['id']}")
        print(f"Source URL: {chunk['metadata']['source_url']}")
        print(f"Tokens: {chunk['metadata']['tokens']}")
        print(f"Chunk Index: {chunk['metadata']['chunk_index']}/{chunk['metadata']['total_chunks']}")
        print(f"Content Preview: {chunk['content'][:150]}...")
        print(f"Content Length: {len(chunk['content'])} chars")
    
    print("\n" + "="*60)

# Main function for standalone usage
def main():
    """Example standalone usage of the chunking tool."""
    print("🔧 Chunking Tool - Standalone Usage")
    print("This tool is designed to be imported and used by other scripts.")
    print("For usage examples, see search_tool.py")

if __name__ == "__main__":
    main() 