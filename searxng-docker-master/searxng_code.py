#!/usr/bin/env python3

import json, urllib.parse, urllib.request, gzip, zlib

BASE_URL = "http://localhost:9000"
ENGINES = "wikipedia,wikidata,arxiv,doaj,duckduckgo,google"

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

if __name__ == "__main__":
    query = "quantum computing"
    payload = get(query)
    for item in payload.get("results", []):
        url = item.get("url")
        if url: print(url) 