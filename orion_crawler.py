#!/usr/bin/env python3
"""
Orion Crawler - Optimized library for search and web scraping.
Provides fast, efficient web scraping with targeted content filtering.
"""

import json, urllib.parse, urllib.request, gzip, zlib
import asyncio
import aiohttp
from crawl4ai import AsyncWebCrawler
import re
import trafilatura
from typing import List, Tuple, Dict, Any
import time
from functools import lru_cache
import hashlib
from collections import Counter
import logging
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

BASE_URL = "http://localhost:9000"
ENGINES = "google,duckduckgo"

# ============================================================================
# OPTIMIZED FILTERING SYSTEM
# ============================================================================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Navigation & UI Elements Blocklist
NAVIGATION_BLOCKLIST = {
    # Menu items
    'home', 'about', 'contact', 'privacy policy', 'terms of service', 'terms', 'conditions',
    # Button text
    'submit', 'read more', 'click here', 'learn more', 'continue reading', 'view all', 'show more',
    # Pagination
    'next', 'previous', 'page', 'of',
    # Login/Register
    'login', 'sign up', 'signup', 'register', 'sign in', 'signin', 'logout',
    # Navigation
    'menu', 'navigation', 'nav', 'footer', 'header', 'sidebar', 'breadcrumb'
}

# Advertisements & Marketing Blocklist
AD_MARKETING_BLOCKLIST = {
    'ad', 'ads', 'advertisement', 'advertisements', 'sponsored', 'promotion', 'promotional',
    'subscribe', 'newsletter', 'sign up', 'download our app', 'download app',
    'we use cookies', 'accept all cookies', 'cookie policy', 'cookies'
}

# Boilerplate & Repeated Content Blocklist
BOILERPLATE_BLOCKLIST = {
    'copyright', 'all rights reserved', 'reserved', 'company name',
    'follow us on', 'follow us', 'social media', 'twitter', 'facebook', 'instagram',
    'tags:', 'category:', 'categories:', 'tagged:', 'author:', 'last updated:',
    'enable javascript', 'javascript required', 'var _gaq', 'google analytics'
}

# Combined blocklist for efficient filtering
COMBINED_BLOCKLIST = NAVIGATION_BLOCKLIST | AD_MARKETING_BLOCKLIST | BOILERPLATE_BLOCKLIST

# Optimized regex patterns for content filtering
FILTERING_PATTERNS = {
    # Contact information
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
    'url': re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
    
    # Timestamps and dates
    'timestamp': re.compile(r'\b\d{1,2}:\d{2}(?:\s?(?:am|pm))?\b'),
    'date': re.compile(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b'),
    'last_updated': re.compile(r'last\s+updated:\s*[^,\n]+', re.IGNORECASE),
    
    # HTML and formatting
    'html_tags': re.compile(r'<[^>]+>'),
    'script_style': re.compile(r'<(script|style)[^>]*>.*?</\1>', re.DOTALL | re.IGNORECASE),
    'heading_pattern': re.compile(r'^#{1,6}\s+', re.MULTILINE),
    'html_heading': re.compile(r'<h[1-6][^>]*>.*?</h[1-6]>', re.DOTALL | re.IGNORECASE),
    
    # Formatting noise
    'special_chars': re.compile(r'[=*~\-_]{4,}'),
    'empty_lines': re.compile(r'^\s*$', re.MULTILINE),
    'excessive_whitespace': re.compile(r'\s{3,}'),
    
    # Tracking and analytics
    'analytics': re.compile(r'var\s+_gaq|google\s+analytics|gtag|fbq', re.IGNORECASE),
    'javascript_required': re.compile(r'enable\s+javascript|javascript\s+required', re.IGNORECASE),
    
    # Social media and tags
    'social_media': re.compile(r'follow\s+us\s+on|@\w+|#\w+', re.IGNORECASE),
    'tags_categories': re.compile(r'(tags?|categories?|tagged?):\s*[^,\n]+', re.IGNORECASE),
    
    # Copyright and legal
    'copyright': re.compile(r'©\s*\d{4}|copyright\s+\d{4}|all\s+rights\s+reserved', re.IGNORECASE)
}

# Global storage for boilerplate detection
boilerplate_hashes = set()
chunk_frequency = Counter()

def is_heading(text: str) -> bool:
    """Check if text is a heading (markdown or HTML)."""
    text = text.strip()
    if not text:
        return False
    
    # Check for markdown headings
    if FILTERING_PATTERNS['heading_pattern'].match(text):
        return True
    
    # Check for HTML headings
    if FILTERING_PATTERNS['html_heading'].search(text):
        return True
    
    # Check for short lines that might be headings (heuristic)
    if len(text) < 100 and not text.endswith(('.', '!', '?', ':', ';')):
        return True
    
    return False

def remove_low_information_tokens(text: str) -> str:
    """
    Remove low-information tokens:
    - Drop text lines shorter than 3 words (unless it's a heading)
    - Remove lines with <3-4 characters
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:  # Skip empty or very short lines
            continue
            
        # Count words (simple whitespace split)
        words = line.split()
        word_count = len(words)
        
        # Keep headings regardless of length
        if is_heading(line):
            filtered_lines.append(line)
        # Keep lines with 3+ words
        elif word_count >= 3:
            filtered_lines.append(line)
        # Keep meaningful single words (longer than 2 chars, alphabetic)
        elif word_count == 1 and len(line) > 2 and line.isalpha():
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def detect_boilerplate_with_hashing(text: str, min_length: int = 50) -> str:
    """
    Detect and remove repeated boilerplate across multiple pages using hashing.
    """
    if not text:
        return ""
    
    # Split into chunks for hashing
    chunks = []
    current_chunk = ""
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            if current_chunk and len(current_chunk) >= min_length:
                chunks.append(current_chunk)
                current_chunk = ""
            continue
        
        current_chunk += line + "\n"
        
        # If chunk is long enough, process it
        if len(current_chunk) >= min_length:
            chunks.append(current_chunk)
            current_chunk = ""
    
    # Add remaining chunk
    if current_chunk and len(current_chunk) >= min_length:
        chunks.append(current_chunk)
    
    # Filter out boilerplate chunks
    filtered_chunks = []
    for chunk in chunks:
        # Create hash of normalized chunk
        normalized_chunk = re.sub(r'\s+', ' ', chunk.lower().strip())
        chunk_hash = hashlib.md5(normalized_chunk.encode()).hexdigest()
        
        # Check if this is boilerplate (seen multiple times)
        chunk_frequency[chunk_hash] += 1
        
        # Only keep chunks that haven't been seen too often
        if chunk_frequency[chunk_hash] <= 2:  # Allow up to 2 occurrences
            filtered_chunks.append(chunk)
        else:
            # This is likely boilerplate
            boilerplate_hashes.add(chunk_hash)
    
    return '\n'.join(filtered_chunks)

def apply_optimized_regex_filters(text: str) -> str:
    """
    Apply optimized regex filters for content cleaning.
    """
    if not text:
        return ""
    
    # Remove HTML tags and script/style contents first
    text = FILTERING_PATTERNS['script_style'].sub('', text)
    text = FILTERING_PATTERNS['html_tags'].sub('', text)
    
    # Remove contact information
    text = FILTERING_PATTERNS['email'].sub('[EMAIL]', text)
    text = FILTERING_PATTERNS['phone'].sub('[PHONE]', text)
    text = FILTERING_PATTERNS['url'].sub('[URL]', text)
    
    # Remove timestamps and dates
    text = FILTERING_PATTERNS['timestamp'].sub('[TIME]', text)
    text = FILTERING_PATTERNS['date'].sub('[DATE]', text)
    text = FILTERING_PATTERNS['last_updated'].sub('[UPDATED]', text)
    
    # Remove formatting noise
    text = FILTERING_PATTERNS['special_chars'].sub('', text)
    text = FILTERING_PATTERNS['excessive_whitespace'].sub(' ', text)
    
    # Remove tracking and analytics
    text = FILTERING_PATTERNS['analytics'].sub('', text)
    text = FILTERING_PATTERNS['javascript_required'].sub('', text)
    
    # Remove social media and tags
    text = FILTERING_PATTERNS['social_media'].sub('', text)
    text = FILTERING_PATTERNS['tags_categories'].sub('', text)
    
    # Remove copyright and legal text
    text = FILTERING_PATTERNS['copyright'].sub('', text)
    
    return text

def filter_blocklist_content(text: str) -> str:
    """
    Filter out sentences containing mostly blocklist words using optimized blocklist.
    """
    if not text:
        return ""
    
    sentences = re.split(r'[.!?]+', text)
    filtered_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Count blocklist words in sentence
        words = sentence.lower().split()
        if not words:
            continue
        
        # Check against combined blocklist for efficiency
        blocklist_count = sum(1 for word in words if word in COMBINED_BLOCKLIST)
        blocklist_ratio = blocklist_count / len(words)
        
        # Keep sentence if less than 50% blocklist words
        if blocklist_ratio < 0.5:
            filtered_sentences.append(sentence)
    
    return '. '.join(filtered_sentences) + '.' if filtered_sentences else ""

def deduplicate_within_page(text: str) -> str:
    """
    Deduplicate repeated content within a single page.
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    seen_lines = set()
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Normalize line for comparison
        normalized_line = re.sub(r'\s+', ' ', line.lower())
        
        # Skip if we've seen this exact line before
        if normalized_line in seen_lines:
            continue
        
        # Skip very common repeated phrases
        if normalized_line in {'read more', 'share', 'like', 'comment', 'follow', 'subscribe'}:
            continue
        
        seen_lines.add(normalized_line)
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def normalize_and_clean(text: str, lowercase: bool = False) -> str:
    """
    Normalize whitespace and punctuation, optionally lowercase.
    """
    if not text:
        return ""
    
    # Optionally lowercase
    if lowercase:
        text = text.lower()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize punctuation spacing
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()

def semantic_content_filtering(text: str, min_tokens: int = 30) -> str:
    """
    Keep only content-rich passages (paragraphs > min_tokens).
    """
    if not text:
        return ""
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    filtered_paragraphs = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # Count tokens (simple word count)
        tokens = paragraph.split()
        if len(tokens) >= min_tokens:
            filtered_paragraphs.append(paragraph)
    
    return '\n\n'.join(filtered_paragraphs)


def fast_content_quality_check(text: str) -> bool:
    """
    Quick quality check to determine if content needs heavy cleaning.
    Returns True if content appears to be high quality and can skip heavy processing.
    """
    if not text or len(text.strip()) < 100:
        return False
    
    # Check for obvious signs of good content
    words = text.split()
    if len(words) < 150:
        return False
    
    # Check for reasonable sentence structure
    sentences = text.split('.')
    if len(sentences) < 3:
        return False
    
    # Check for minimal HTML/technical artifacts
    html_count = text.count('<') + text.count('>')
    if html_count > len(text) * 0.05:  # More than 5% HTML
        return False
    
    # Check for reasonable word diversity
    unique_words = len(set(word.lower() for word in words if len(word) > 2))
    if unique_words < len(words) * 0.3:  # Less than 30% unique words
        return False
    
    return True

def apply_optimized_cleaning(text: str, 
                           lowercase: bool = False,
                           min_tokens: int = 30) -> str:
    """
    Apply optimized cleaning rules in sequence for faster processing.
    Uses fast quality check to skip heavy processing for good content.
    
    Args:
        text: Input text to clean
        lowercase: Whether to convert to lowercase
        min_tokens: Minimum tokens for content-rich passages
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Fast quality check - if content looks good, do minimal cleaning
    if fast_content_quality_check(text):
        # Just do basic normalization for high-quality content
        text = normalize_and_clean(text, lowercase)
        return text
    
    # Full cleaning pipeline for lower quality content
    # Step 1: Remove low-information tokens
    text = remove_low_information_tokens(text)
    
    # Step 2: Apply optimized regex filters (includes HTML removal)
    text = apply_optimized_regex_filters(text)
    
    # Step 3: Filter blocklist content
    text = filter_blocklist_content(text)
    
    # Step 4: Deduplicate within page
    text = deduplicate_within_page(text)
    
    # Step 5: Normalize and clean
    text = normalize_and_clean(text, lowercase)
    
    # Step 6: Semantic content filtering
    text = semantic_content_filtering(text, min_tokens)
    
    # Step 7: Detect and remove boilerplate (moved to end for efficiency)
    text = detect_boilerplate_with_hashing(text)
    
    return text



def get(query: str):
    params = {
        "q": query,
        "format": "json",
        "pageno": "1",
        "engines": ENGINES,
    }
    url = f"{BASE_URL}/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    })
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        data = resp.read()
        enc = (resp.headers.get("Content-Encoding") or "").lower()
        if enc == "gzip": data = gzip.decompress(data)
        elif enc == "deflate":
            try: data = zlib.decompress(data, -zlib.MAX_WBITS)
            except zlib.error: data = zlib.decompress(data)
    return json.loads(data.decode("utf-8"))


@lru_cache(maxsize=100)
def fetch_urls(query: str) -> List[str]:
    """
    Fetch URLs from SearXNG based on the given query.
    Returns exactly 3 URLs per engine (Google and DuckDuckGo only).
    Optimized for early termination when target count is reached.
    Cached to avoid repeated SearXNG calls for same queries.
    
    Args:
        query (str): The search query to execute
        
    Returns:
        List[str]: A list of URLs from the search results (exactly 3 per engine)
        
    Raises:
        Exception: If there's an error during the search request
    """
    try:
        payload = get(query)
        urls = []
        target_engines = set(ENGINES.split(','))
        engine_counts = {engine: 0 for engine in target_engines}
        
        for item in payload.get("results", []):
            url = item.get("url")
            engine = item.get("engine", "unknown")
            
            if url and engine in target_engines and engine_counts[engine] < 3:
                    urls.append(url)
                    engine_counts[engine] += 1
                    
                # Early termination: stop if we have exactly 3 for all engines
            if all(count == 3 for count in engine_counts.values()):
                    break
                    
        return urls
    except Exception as e:
        print(f"Error fetching URLs for query '{query}': {e}")
        return []


async def scrape_webpage(url: str, session: aiohttp.ClientSession, timeout: float = 20.0) -> str:
    """
    Optimized webpage scraping with connection pooling and faster processing.
    
    Args:
        url (str): The URL to scrape
        session (aiohttp.ClientSession): Shared session for connection pooling
        timeout (float): Timeout in seconds for the scraping operation
        
    Returns:
        str: The scraped content in clean text format
    """
    try:
        # Use aiohttp for faster initial fetch
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status != 200:
                logger.warning(f"HTTP {response.status} for {url}")
                return ""
            
            html_content = await response.text()
            
            # Use Trafilatura for fast text extraction
            extracted_text = trafilatura.extract(
                html_content,
                include_formatting=True,
                include_links=False,
                include_images=False,
                include_tables=True,
                no_fallback=False
            )
            
            if extracted_text:
                # Quick cleanup
                cleaned_text = re.sub(r'\n\s*\n', '\n\n', extracted_text)
                cleaned_text = re.sub(r' +', ' ', cleaned_text)
                return cleaned_text.strip()
            else:
                # Fallback to basic extraction
                return trafilatura.extract_text(html_content) or ""
                
    except asyncio.TimeoutError:
        logger.warning(f"Timeout scraping {url} after {timeout}s")
        return ""
    except Exception as e:
        logger.warning(f"Error scraping {url}: {e}")
        return ""

async def scrape_webpage_fallback(url: str, timeout: float = 25.0) -> str:
    """
    Fallback scraping method using crawl4ai for complex pages.
    """
    try:
        async with AsyncWebCrawler() as crawler:
            result = await asyncio.wait_for(
                crawler.arun(
                    url=url,
                    output_format="html",
                    include_links=False,
                    include_images=False,
                    include_tables=False
                ), 
                timeout=timeout
            )
            
            html_content = getattr(result, 'html', None) or getattr(result, 'content', None)
            if html_content:
                extracted_text = trafilatura.extract(html_content, include_formatting=True)
                if extracted_text:
                    return re.sub(r'\n\s*\n', '\n\n', extracted_text).strip()
            
            return str(result)
            
    except Exception as e:
        logger.warning(f"Fallback scraping failed for {url}: {e}")
        return ""



@lru_cache(maxsize=128)
def filter_content_quality(text: str, min_words: int = 150) -> bool:
    """
    Filter content based on quality metrics for embeddings.
    Optimized with early returns and caching.
    Now requires minimum 150 words for substantial content.
    
    Args:
        text (str): Text to evaluate
        min_words (int): Minimum word count to consider valid (default: 150)
        
    Returns:
        bool: True if content meets quality standards
    """
    # Early returns for obvious cases
    if not text or len(text.strip()) < 50:
        return False
    
    # Split into words once
    words = text.split()
    word_count = len(words)
    
    # Early return for word count
    if word_count < min_words:
        return False
    
    # Early return for very short texts
    if word_count < 20:
        return True  # Short but valid content
    
    # Process meaningful words only
    meaningful_words = [word.lower() for word in words if len(word) > 2]
    
    if len(meaningful_words) < min_words:
        return False
    
    # Check for excessive repetition (likely boilerplate) - LESS AGGRESSIVE
    word_freq = {}
    for word in meaningful_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Early return for repetition check - INCREASED THRESHOLD
    max_freq = max(word_freq.values())
    if max_freq > len(meaningful_words) * 0.5:  # Changed from 0.3 to 0.5
        return False
    
    # Check for reasonable content diversity - LESS AGGRESSIVE
    unique_words = len(word_freq)
    if unique_words < len(meaningful_words) * 0.2:  # Changed from 0.3 to 0.2
        return False
    
    return True


# ============================================================================
# BATCH PROCESSING
# ============================================================================

async def scrape_urls_batch(urls: List[str], max_concurrent: int = 15, timeout: float = 15.0) -> List[Tuple[str, str]]:
    """
    Optimized batch scraping with connection pooling and intelligent fallback.
    
    Args:
        urls (List[str]): List of URLs to scrape
        max_concurrent (int): Maximum number of concurrent scraping operations
        timeout (float): Timeout per URL in seconds
        
    Returns:
        List[Tuple[str, str]]: List of (url, content) tuples
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    
    # Create shared session for connection pooling with keep-alive
    connector = aiohttp.TCPConnector(
        limit=max_concurrent, 
        limit_per_host=5,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    timeout_config = aiohttp.ClientTimeout(total=timeout, connect=5)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout_config,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    ) as session:
        
        async def scrape_with_fallback(url: str) -> Tuple[str, str]:
            async with semaphore:
                try:
                    # Try optimized scraping first
                    content = await scrape_webpage(url, session, timeout)
                    
                    # Smart fallback: only try if content is very short or empty
                    if len(content.strip()) < 50:
                        logger.info(f"Trying fallback for {url} (content too short: {len(content.strip())} chars)")
                        fallback_content = await scrape_webpage_fallback(url, timeout)
                        if len(fallback_content.strip()) > len(content.strip()):
                            content = fallback_content
                    
                    return (url, content)
                    
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
                    return (url, "")
        
        # Create all scraping tasks
        tasks = [scrape_with_fallback(url) for url in urls]
        
        # Execute with optimized error handling
        try:
            scraped_contents = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results efficiently
            for i, result in enumerate(scraped_contents):
                if isinstance(result, Exception):
                    logger.error(f"Task failed for {urls[i]}: {result}")
                    results.append((urls[i], ""))
                else:
                    results.append(result)
                    
        except Exception as e:
            logger.error(f"Error in batch scraping: {e}")
            # Return empty results for failed URLs
            for url in urls:
                results.append((url, ""))
    
    return results

async def fetch_and_scrape_parallel(query: str, max_concurrent: int = 15, timeout: float = 15.0) -> List[Tuple[str, str]]:
    """
    Optimized fetch and scrape with better performance metrics and error handling.
    
    Args:
        query (str): The search query to execute
        max_concurrent (int): Maximum number of concurrent scraping operations
        timeout (float): Timeout per URL in seconds
        
    Returns:
        List[Tuple[str, str]]: List of tuples containing (url, scraped_content)
    """
    try:
        start_time = time.time()
        
        # First, get the URLs
        urls = fetch_urls(query)
        logger.info(f"Found {len(urls)} URLs to scrape for query: {query}")
        
        if not urls:
            logger.warning("No URLs found to scrape")
            return []
        
        # Use optimized batch processing
        results = await scrape_urls_batch(urls, max_concurrent, timeout)
        
        # Calculate performance metrics
        elapsed_time = time.time() - start_time
        successful_scrapes = sum(1 for _, content in results if content and len(content.strip()) > 50)
        
        logger.info(f"Scraping completed in {elapsed_time:.2f}s")
        logger.info(f"Successfully scraped {successful_scrapes}/{len(urls)} URLs")
        logger.info(f"Average time per URL: {elapsed_time/len(urls):.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in fetch_and_scrape_parallel_optimized: {e}")
        return []

def fetch_and_scrape_parallel_sync(query: str, max_concurrent: int = 15, timeout: float = 15.0) -> List[Tuple[str, str]]:
    """
    Optimized synchronous wrapper with better error handling and performance.
    
    Args:
        query (str): The search query to execute
        max_concurrent (int): Maximum number of concurrent scraping operations
        timeout (float): Timeout per URL in seconds
        
    Returns:
        List[Tuple[str, str]]: List of tuples containing (url, scraped_content)
    """
    # Suppress asyncio warnings
    import warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)
    
    try:
        # Use asyncio.run which handles cleanup automatically
        return asyncio.run(fetch_and_scrape_parallel(query, max_concurrent, timeout))
    except Exception as e:
        logger.error(f"Error in synchronous wrapper: {e}")
        return []

# ============================================================================
# MAIN LIBRARY INTERFACE
# ============================================================================

def search_and_scrape(query: str, 
                       max_concurrent: int = 15, 
                       timeout: float = 15.0) -> Dict[str, Any]:
    """
    Optimized main library function: Search query and return filtered results.
    Now filters out content with less than 150 words for substantial results.
    
    Args:
        query (str): Search query to execute
        max_concurrent (int): Maximum concurrent scraping operations (default: 15)
        timeout (float): Timeout per URL in seconds (default: 15.0)
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'success': bool - Whether operation was successful
            - 'query': str - Original search query
            - 'results': List[Dict] - List of results with url, content, length (150+ words)
            - 'total_urls': int - Total URLs processed
            - 'successful_scrapes': int - Number of successfully scraped URLs
            - 'error': str - Error message if success=False
    """
    try:
        start_time = time.time()
        
        # Execute optimized search and scraping
        results = fetch_and_scrape_parallel_sync(query, max_concurrent, timeout)
        
        if not results:
            return {
                'success': False,
                'query': query,
                'error': 'No results obtained from search',
                'results': [],
                'total_urls': 0,
                'successful_scrapes': 0
            }
        
        # Process results into clean format with parallel cleaning
        processed_results = []
        successful_scrapes = 0
        
        # Filter substantial content first (quick check) - require at least 150 words
        substantial_results = [(url, content) for url, content in results 
                              if content.strip() and len(content.strip()) > 50 and len(content.split()) >= 150]
        
        if substantial_results:
            # Process content cleaning in parallel using ThreadPoolExecutor for better performance
            
            def clean_content_sync(url_content_tuple):
                url, content = url_content_tuple
                try:
                    cleaned_content = apply_optimized_cleaning(content, lowercase=False, min_tokens=150)
                    if cleaned_content.strip() and len(cleaned_content.split()) >= 150:
                        return {
                            'url': url,
                            'content': cleaned_content,
                            'length': len(cleaned_content),
                            'word_count': len(cleaned_content.split()),
                            'embedding_ready': True
                        }
                except Exception as e:
                    logger.warning(f"Error cleaning content for {url}: {e}")
                return None
            
            # Process cleaning in parallel (limit to avoid overwhelming system)
            with ThreadPoolExecutor(max_workers=min(8, len(substantial_results))) as executor:
                future_to_result = {executor.submit(clean_content_sync, result): result 
                                  for result in substantial_results}
                
                for future in concurrent.futures.as_completed(future_to_result):
                    result = future.result()
                    if result is not None:
                        processed_results.append(result)
                        successful_scrapes += 1
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        # Return clean response
        return {
            'success': True,
            'query': query,
            'results': processed_results,
            'total_urls': len(results),
            'successful_scrapes': successful_scrapes,
            'processing_time': total_time,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error in search_and_scrape: {e}")
        return {
            'success': False,
            'query': query,
            'error': str(e),
            'results': [],
            'total_urls': 0,
            'successful_scrapes': 0
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_result_summary(results: List[Dict]) -> str:
    """
    Get a summary of search results.
    
    Args:
        results (List[Dict]): Results from search_and_scrape function
        
    Returns:
        str: Formatted summary string
    """
    if not results:
        return "No results found."
    
    summary = f"Found {len(results)} results:\n"
    for i, result in enumerate(results, 1):
        summary += f"{i}. {result['url'][:60]}... ({result['length']} chars)\n"
    
    return summary


def get_performance_stats(results: Dict[str, Any]) -> str:
    """
    Get performance statistics from search results.
    
    Args:
        results (Dict[str, Any]): Results from search_and_scrape
        
    Returns:
        str: Formatted performance statistics
    """
    if not results.get('success'):
        return f"Search failed: {results.get('error', 'Unknown error')}"
    
    stats = f"Query: {results['query']}\n"
    stats += f"Total URLs: {results['total_urls']}\n"
    stats += f"Successful scrapes: {results['successful_scrapes']}\n"
    stats += f"Success rate: {results['successful_scrapes']/results['total_urls']*100:.1f}%\n"
    
    if 'processing_time' in results:
        stats += f"Processing time: {results['processing_time']:.2f}s\n"
        if results['total_urls'] > 0:
            stats += f"Average time per URL: {results['processing_time']/results['total_urls']:.2f}s"
    
    return stats 