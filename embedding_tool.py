#!/usr/bin/env python3
"""
Embedding Tool - A utility for generating embeddings using SentenceTransformers.
Strictly embedding-related; storage/ingestion is handled by vector_DB.
"""

import json
import datetime
import numpy as np
from typing import List, Dict, Any, Tuple
import os
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Import SentenceTransformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  SentenceTransformers not available. Install with: pip install sentence-transformers")

# Global embedding model instance with thread safety
_embedding_model = None
_model_lock = threading.Lock()

def get_embedding_model(model_name: str = "BAAI/bge-large-en-v1.5"):
    """
    Get or create the embedding model instance with thread safety.
    """
    global _embedding_model

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("SentenceTransformers is required for embeddings. Install with: pip install sentence-transformers")

    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:  # Double-check locking
                print(f"🔄 Loading embedding model: {model_name}")
                start_time = time.time()
                _embedding_model = SentenceTransformer(model_name)
                load_time = time.time() - start_time
                print(f"✅ Embedding model loaded successfully in {load_time:.2f}s!")

    return _embedding_model

def get_embeddings(texts: List[str], 
                   model_name: str = "BAAI/bge-large-en-v1.5",
                   batch_size: int = 32,
                   show_progress: bool = True) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using SentenceTransformers with optimizations.
    
    Args:
        texts: List of texts to embed
        model_name: Model to use for embedding
        batch_size: Batch size for processing (default: 32)
        show_progress: Whether to show progress bar
    """
    if not texts:
        return []

    try:
        model = get_embedding_model(model_name)
        
        # Optimize batch size based on text length and count
        if len(texts) > 100:
            batch_size = min(batch_size, 64)  # Larger batches for many texts
        elif len(texts) < 10:
            batch_size = min(batch_size, 8)   # Smaller batches for few texts
        
        start_time = time.time()
        
        # Use optimized encoding with batching
        embeddings = model.encode(
            texts, 
            convert_to_tensor=False, 
            show_progress_bar=show_progress,
            batch_size=batch_size,
            normalize_embeddings=True  # Normalize for better similarity search
        )

        # Convert to list of lists if needed
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        processing_time = time.time() - start_time
        chunks_per_second = len(texts) / processing_time if processing_time > 0 else 0
        
        print(f"✅ Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
        print(f"⚡ Processing speed: {chunks_per_second:.1f} chunks/second ({processing_time:.2f}s total)")
        
        return embeddings

    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return []

def embed_chunks(chunks: List[Dict[str, Any]], 
                 model_name: str = "BAAI/bge-large-en-v1.5",
                 batch_size: int = 32,
                 show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Add embeddings to chunks using SentenceTransformers with optimizations.
    Returns a new list with embeddings added under 'embedding' and model info in metadata.
    """
    if not chunks:
        print("⚠️  No chunks to embed")
        return []

    # Extract texts efficiently
    texts = [chunk['content'] for chunk in chunks]
    
    # Generate embeddings with optimizations
    embeddings = get_embeddings(texts, model_name, batch_size, show_progress)

    if not embeddings:
        print("❌ Failed to generate embeddings")
        return chunks

    # Create embedded chunks efficiently using list comprehension
    embedded_chunks = []
    embedding_dim = len(embeddings[0]) if embeddings else 0
    
    for i, chunk in enumerate(chunks):
        embedded_chunk = chunk.copy()
        embedded_chunk['embedding'] = embeddings[i]
        embedded_chunk['metadata']['embedding_model'] = model_name
        embedded_chunk['metadata']['embedding_dimension'] = embedding_dim
        embedded_chunks.append(embedded_chunk)

    print(f"✅ Added embeddings to {len(embedded_chunks)} chunks")
    return embedded_chunks

def embed_chunks_batch_optimized(chunks: List[Dict[str, Any]], 
                                model_name: str = "BAAI/bge-large-en-v1.5",
                                max_workers: int = 4,
                                chunk_batch_size: int = 50) -> List[Dict[str, Any]]:
    """
    Process chunks in parallel batches for maximum speed.
    Splits chunks into batches and processes them concurrently.
    """
    if not chunks:
        print("⚠️  No chunks to embed")
        return []

    if len(chunks) <= chunk_batch_size:
        # Small batch, process normally
        return embed_chunks(chunks, model_name)

    print(f"🚀 Processing {len(chunks)} chunks in parallel batches...")
    start_time = time.time()

    # Split chunks into batches
    batches = [chunks[i:i + chunk_batch_size] for i in range(0, len(chunks), chunk_batch_size)]
    print(f"📦 Split into {len(batches)} batches of ~{chunk_batch_size} chunks each")

    all_embedded_chunks = []

    def process_batch(batch_chunks):
        """Process a single batch of chunks."""
        return embed_chunks(batch_chunks, model_name, show_progress=False)

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        
        # Collect results in order
        batch_results = [None] * len(batches)
        for future in future_to_batch:
            batch_idx = future_to_batch[future]
            try:
                batch_results[batch_idx] = future.result()
            except Exception as e:
                print(f"❌ Error processing batch {batch_idx}: {e}")
                batch_results[batch_idx] = []

        # Flatten results
        for batch_result in batch_results:
            if batch_result:
                all_embedded_chunks.extend(batch_result)

    total_time = time.time() - start_time
    chunks_per_second = len(chunks) / total_time if total_time > 0 else 0
    
    print(f"✅ Parallel processing completed: {len(all_embedded_chunks)} chunks in {total_time:.2f}s")
    print(f"⚡ Speed: {chunks_per_second:.1f} chunks/second")
    
    return all_embedded_chunks

def save_embedded_chunks_to_json(embedded_chunks: List[Dict[str, Any]], filename: str = None):
    """Save embedded chunks to a JSON file for later ingestion."""
    if not filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"orion_embedded_chunks_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)

    print(f"📁 Embedded chunks saved to: {filename}")
    return filename

def verify_embeddings(embedded_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify and analyze the embeddings (no storage side effects)."""
    if not embedded_chunks:
        return {"error": "No embedded chunks to verify"}

    has_embeddings = all('embedding' in chunk for chunk in embedded_chunks)
    if not has_embeddings:
        return {"error": "No embeddings found in chunks"}

    embedding_dims = [len(chunk['embedding']) for chunk in embedded_chunks]
    embeddings_array = np.array([chunk['embedding'] for chunk in embedded_chunks])

    verification = {
        "total_chunks": len(embedded_chunks),
        "has_embeddings": has_embeddings,
        "embedding_stats": {
            "dimension": embedding_dims[0] if embedding_dims else 0,
            "min_dimension": min(embedding_dims) if embedding_dims else 0,
            "max_dimension": max(embedding_dims) if embedding_dims else 0,
            "consistent_dimensions": len(set(embedding_dims)) == 1,
            "mean_norm": float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings_array, axis=1)))
        },
        "sample_embeddings": embedded_chunks[:2] if len(embedded_chunks) >= 2 else embedded_chunks
    }

    return verification

def print_embedding_verification(embedded_chunks: List[Dict[str, Any]]):
    """Print detailed verification information about embeddings."""
    verification = verify_embeddings(embedded_chunks)

    if "error" in verification:
        print(f"❌ {verification['error']}")
        return

    print("\n" + "="*60)
    print("🔍 EMBEDDING VERIFICATION REPORT")
    print("="*60)

    print(f"\n📊 BASIC STATISTICS:")
    print(f"Total chunks: {verification['total_chunks']}")
    print(f"Has embeddings: {'✅' if verification['has_embeddings'] else '❌'}")

    print(f"\n🎯 EMBEDDING STATISTICS:")
    embedding_stats = verification['embedding_stats']
    print(f"Dimension: {embedding_stats['dimension']}")
    print(f"Consistent dimensions: {'✅' if embedding_stats['consistent_dimensions'] else '❌'}")
    print(f"Mean norm: {embedding_stats['mean_norm']:.4f}")
    print(f"Std norm: {embedding_stats['std_norm']:.4f}")

    print(f"\n📄 SAMPLE EMBEDDINGS:")
    for i, chunk in enumerate(verification['sample_embeddings'], 1):
        print(f"\n--- EMBEDDED CHUNK {i} ---")
        print(f"ID: {chunk['id']}")
        print(f"Source URL: {chunk['metadata']['source_url']}")
        print(f"Tokens: {chunk['metadata']['tokens']}")
        print(f"Embedding dimension: {chunk['metadata']['embedding_dimension']}")
        print(f"Embedding model: {chunk['metadata']['embedding_model']}")
        print(f"Content Preview: {chunk['content'][:100]}...")
        print(f"Embedding preview: {chunk['embedding'][:5]}...")

    print("\n" + "="*60)

def process_chunks_with_embeddings(chunks: List[Dict[str, Any]], 
                                  model_name: str = "BAAI/bge-large-en-v1.5",
                                  save_json: bool = True,
                                  save_numpy: bool = False,
                                  output_dir: str = None,
                                  use_parallel: bool = True,
                                  max_workers: int = 4) -> Tuple[List[Dict[str, Any]], str]:
    """
    Complete workflow: embed chunks and optionally save results with optimizations.
    Returns embedded chunks for ingestion by vector_DB.
    
    Args:
        chunks: List of chunks to process
        model_name: Model to use for embedding
        save_json: Whether to save results to JSON
        save_numpy: Whether to save embeddings as numpy array
        output_dir: Directory to save files
        use_parallel: Whether to use parallel processing for large batches
        max_workers: Number of parallel workers for processing
    """
    if not chunks:
        print("⚠️  No chunks to process")
        return [], ""

    total_start_time = time.time()
    print(f"🔄 Processing {len(chunks)} chunks with embeddings...")

    # Choose processing method based on chunk count and parallel flag
    if use_parallel and len(chunks) > 20:
        print("🚀 Using parallel processing for large batch")
        embedded_chunks = embed_chunks_batch_optimized(chunks, model_name, max_workers)
    else:
        print("🔄 Using standard processing")
        embedded_chunks = embed_chunks(chunks, model_name)

    if not embedded_chunks:
        print("❌ Failed to create embedded chunks")
        return chunks, ""

    # Quick verification (skip detailed verification for speed)
    if len(embedded_chunks) <= 10:
        print_embedding_verification(embedded_chunks)
    else:
        print(f"✅ Generated {len(embedded_chunks)} embeddings (skipping detailed verification for speed)")

    filename = ""
    if save_json:
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception:
                pass
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"orion_embedded_chunks_{timestamp}.json")
            filename = save_embedded_chunks_to_json(embedded_chunks, filename)
        else:
            filename = save_embedded_chunks_to_json(embedded_chunks)

    if save_numpy:
        embeddings = [chunk['embedding'] for chunk in embedded_chunks]
        embeddings_array = np.array(embeddings)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"orion_embeddings_{timestamp}.npy") if output_dir else f"orion_embeddings_{timestamp}.npy"
        try:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        except Exception:
            pass
        np.save(out_path, embeddings_array)

    total_time = time.time() - total_start_time
    chunks_per_second = len(chunks) / total_time if total_time > 0 else 0
    
    print(f"✅ Successfully processed {len(embedded_chunks)} chunks with embeddings")
    print(f"⚡ Total processing time: {total_time:.2f}s ({chunks_per_second:.1f} chunks/second)")
    
    return embedded_chunks, filename

if __name__ == "__main__":
    print("🔧 Embedding Tool - Standalone Usage")
    print("This module provides embedding utilities only. Use vector_DB for ingestion.") 