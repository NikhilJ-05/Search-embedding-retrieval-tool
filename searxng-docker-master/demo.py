#!/usr/bin/env python3

import argparse
import json
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


def build_search_url(base_url: str, query: str, engines: Optional[str] = None,
                     language: Optional[str] = None, page: int = 1) -> str:
	params: Dict[str, Any] = {
		"q": query,
		"format": "json",
		"pageno": str(page),
	}
	if engines:
		params["engines"] = engines
	if language:
		params["language"] = language

	query_string = urllib.parse.urlencode(params)
	base = base_url.rstrip("/")
	return f"{base}/search?{query_string}"


def http_get_json(url: str, timeout: float = 15.0) -> Dict[str, Any]:
	request = urllib.request.Request(
		url,
		headers={
			# Use a common browser UA to avoid bot detection
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
			# Include typical browser Accept headers
			"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
			# Language helps pass Accept-Language probe
			"Accept-Language": "en-US,en;q=0.9",
			# Allow compressed responses
			"Accept-Encoding": "gzip, deflate",
		}
	)
	with urllib.request.urlopen(request, timeout=timeout) as response:
		if response.status != 200:
			raise RuntimeError(f"HTTP {response.status} from {url}")
		data = response.read()
		encoding = (response.headers.get("Content-Encoding") or "").lower()
		if encoding == "gzip":
			import gzip
			data = gzip.decompress(data)
		elif encoding == "deflate":
			import zlib
			try:
				data = zlib.decompress(data, -zlib.MAX_WBITS)
			except zlib.error:
				data = zlib.decompress(data)
		return json.loads(data.decode("utf-8"))


def print_results(payload: Dict[str, Any], limit: int = 5) -> None:
	results: List[Dict[str, Any]] = payload.get("results", [])
	print(f"number_of_results: {payload.get('number_of_results')}")
	print(f"returned_results: {len(results)}")
	print()
	for idx, item in enumerate(results[:limit], start=1):
		title = item.get("title") or "<no title>"
		url = item.get("url") or "<no url>"
		engines = ",".join(item.get("engines", []) or [])
		print(f"{idx}. {title}\n   {url}\n   engines: {engines}")


def main(argv: List[str]) -> int:
	parser = argparse.ArgumentParser(description="Query SearXNG API and print top results")
	parser.add_argument("--base-url", default="http://localhost:9000", help="Base URL of SearXNG (default: http://localhost:9000)")
	parser.add_argument("--q", dest="query", default="test", help="Query string (default: test)")
	parser.add_argument("--engines", default=None, help="Comma-separated engines (optional)")
	parser.add_argument("--language", default=None, help="Language code, e.g., en (optional)")
	parser.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
	parser.add_argument("--limit", type=int, default=5, help="Number of results to print (default: 5)")
	args = parser.parse_args(argv)

	url = build_search_url(
		base_url=args.base_url,
		query=args.query,
		engines=args.engines,
		language=args.language,
		page=args.page,
	)

	try:
		payload = http_get_json(url)
	except Exception as exc:
		print(f"Request failed: {exc}", file=sys.stderr)
		return 2

	print_results(payload, limit=args.limit)
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))
