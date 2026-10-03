import os
import concurrent.futures
import requests
from langchain_core.tools import tool

# ── OpenSERP configuration ────────────────────────────────────────────────────
# Set OPENSERP_BASE_URL in .env to point at your self-hosted OpenSERP instance.
# Falls back to DuckDuckGo automatically when OpenSERP is unreachable.
OPENSERP_BASE_URL = os.getenv("OPENSERP_BASE_URL", "http://localhost:7070")

_DDGS_TIMEOUT = 20   # seconds before a single DDGS call is abandoned


def _ddgs_raw(query: str, max_results: int = 5) -> list[dict]:
    """
    DuckDuckGo search via the ddgs package — no API key required.
    Runs in a thread with a hard timeout so a hung/rate-limited call never blocks forever.
    """
    def _do() -> list[dict]:
        from ddgs import DDGS
        out = []
        for r in DDGS().text(query, max_results=max_results):
            out.append({
                "title":       r.get("title", ""),
                "url":         r.get("href", ""),
                "description": r.get("body", ""),
            })
        return out

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_do)
            return future.result(timeout=_DDGS_TIMEOUT)
    except concurrent.futures.TimeoutError:
        print(f"[DDGS] timeout after {_DDGS_TIMEOUT}s for '{query[:60]}'")
        return []
    except Exception as e:
        print(f"[DDGS] error for '{query[:60]}': {e}")
        return []


def _openserp_raw(query: str, max_results: int = 5) -> list[dict]:
    """
    Query the OpenSERP HTTP API.
    Falls back to DuckDuckGo automatically when OpenSERP is unreachable.
    """
    try:
        resp = requests.get(
            f"{OPENSERP_BASE_URL}/google",
            params={"text": query, "nresults": max_results},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data[:max_results]
        if isinstance(data, dict):
            for key in ("results", "organic", "items", "data"):
                if key in data:
                    return data[key][:max_results]
        return []
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError):
        # ConnectionError = OpenSERP not running; HTTPError includes 403 from AirPlay on port 7000
        return _ddgs_raw(query, max_results)
    except Exception as e:
        print(f"[OpenSERP] unexpected error — using DuckDuckGo: {e}")
        return _ddgs_raw(query, max_results)


def _format_results(results: list[dict], max_snippet: int = 300) -> str:
    """Format search results as markdown for LLM consumption."""
    parts = []
    for r in results:
        title   = r.get("title", "No title")
        url     = r.get("url", r.get("link", r.get("href", "")))
        snippet = (
            r.get("description")
            or r.get("snippet")
            or r.get("content")
            or r.get("body")
            or ""
        )
        line = f"**{title}**\n{url}"
        if snippet:
            line += f"\n{snippet[:max_snippet].strip()}"
        parts.append(line)
    return "\n\n".join(parts)


def get_openserp_search_tool(max_results: int = 5):
    """
    Return a LangChain @tool for web search.
    Uses OpenSERP when available; falls back to DuckDuckGo transparently.
    """
    _n = max_results

    @tool
    def web_search(query: str) -> str:
        """Search the web for information. Returns markdown-formatted titles, URLs and content extracted from search results."""
        results = _openserp_raw(query, _n)
        if not results:
            return f"No results found for: {query}"
        return _format_results(results)

    return web_search


@tool
def search_youtube_shorts(query: str) -> str:
    """Search YouTube for shorts on a given topic and return video count and top results."""
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    if youtube_api_key:
        result = _youtube_api_search(query, youtube_api_key)
        if not result.startswith("YouTube API error"):
            return result
    return _ddgs_youtube_search(query)


def _youtube_api_search(query: str, api_key: str) -> str:
    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.search().list(
            part="snippet",
            q=query + " shorts",
            type="video",
            videoDuration="short",
            maxResults=5,
            order="viewCount",
        )
        response = request.execute()
        items = response.get("items", [])
        if not items:
            return f"No YouTube Shorts found for: {query}"
        results = [
            f'- "{item["snippet"]["title"]}" by {item["snippet"]["channelTitle"]}'
            for item in items[:5]
        ]
        return f"Found {len(items)} shorts for '{query}':\n" + "\n".join(results)
    except Exception as e:
        return f"YouTube API error: {e}"


def _ddgs_youtube_search(query: str) -> str:
    """Search YouTube via DuckDuckGo — fallback when YouTube API is unavailable."""
    results = _ddgs_raw(f"site:youtube.com/shorts {query}", max_results=5)
    if not results:
        return f"No YouTube results found for: {query}"
    formatted = "\n".join(
        f"- {r.get('title', 'No title')}: {r.get('url', '')}"
        for r in results
    )
    return f"YouTube search results for '{query}':\n{formatted}"
