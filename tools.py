import os
from langchain_tavily import TavilySearch
from langchain_core.tools import tool


def get_tavily_search_tool(max_results: int = 5) -> TavilySearch:
    return TavilySearch(max_results=max_results)


@tool
def search_youtube_shorts(query: str) -> str:
    """Search YouTube for shorts on a given topic and return video count and top results."""
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")

    if youtube_api_key:
        return _youtube_api_search(query, youtube_api_key)
    else:
        return _tavily_youtube_search(query)


def _youtube_api_search(query: str, api_key: str) -> str:
    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.search().list(
            part="snippet",
            q=query + " shorts",
            type="video",
            videoDuration="short",
            maxResults=10,
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
        return f"YouTube API error: {e}. Falling back to web search."


def _tavily_youtube_search(query: str) -> str:
    tavily = TavilySearch(max_results=5)
    raw = tavily.invoke(f"site:youtube.com shorts {query}")
    # New langchain_tavily returns full API dict; old version returned a list
    if isinstance(raw, dict):
        results = raw.get("results", [])
    elif isinstance(raw, list):
        results = raw
    else:
        return f"No results found on YouTube for: {query}"
    if not results:
        return f"No results found on YouTube for: {query}"
    formatted = "\n".join(
        f"- {r.get('title', 'No title')}: {r.get('url', '')}"
        for r in results
        if isinstance(r, dict)
    )
    return f"YouTube search results for '{query}':\n{formatted}"
