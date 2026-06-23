import json
import os
import re
import requests
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

CHANNEL_HANDLE = "Worldaffairs0138"
CHANNEL_URL = "https://www.youtube.com/@Worldaffairs0138"
ANALYSIS_FILE = Path(__file__).parent.parent / "channel_analysis.json"

AREAS = [
    "history", "science", "latest science news", "geography", "nature", "space",
    "psychology", "psychology studies", "technology", "interesting events", "inspiring people",
    "archaeology news", "aviation reports", "research papers",
    "world cultures", "hidden mechanisms",
]

_YT = "https://www.googleapis.com/youtube/v3"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _duration_secs(iso: str) -> int:
    if not iso:
        return 0
    m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso)
    if not m:
        return 0
    return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)


def _is_short(secs: int, title: str, desc: str) -> bool:
    text = (title + " " + desc).lower()
    if "#shorts" in text or "#short" in text:
        return True
    return 0 < secs <= 180   # YouTube Shorts ≤ 3 min (extended limit since 2024)


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


# ── YouTube Data API fetches ──────────────────────────────────────────────────

def fetch_channel_data(api_key: str, max_videos: int = 500) -> tuple[dict, list[dict]]:
    """Return (channel_stats_dict, list_of_short_dicts)."""

    # 1. Channel ID + totals
    r = requests.get(f"{_YT}/channels", params={
        "part": "contentDetails,statistics",
        "forHandle": CHANNEL_HANDLE,
        "key": api_key,
    }, timeout=20)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        raise ValueError(f"Channel @{CHANNEL_HANDLE} not found — check YOUTUBE_API_KEY")

    ch = items[0]
    uploads_playlist = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    channel_stats = {
        "subscriber_count":  int(ch["statistics"].get("subscriberCount", 0)),
        "total_view_count":  int(ch["statistics"].get("viewCount", 0)),
        "total_video_count": int(ch["statistics"].get("videoCount", 0)),
    }

    # 2. All video stubs from uploads playlist
    stubs: list[dict] = []
    next_page = None
    while len(stubs) < max_videos:
        params = {"part": "snippet", "playlistId": uploads_playlist,
                  "maxResults": 50, "key": api_key}
        if next_page:
            params["pageToken"] = next_page
        r = requests.get(f"{_YT}/playlistItems", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        for item in data.get("items", []):
            sn = item.get("snippet", {})
            vid_id = sn.get("resourceId", {}).get("videoId")
            if vid_id:
                stubs.append({
                    "id": vid_id,
                    "title": sn.get("title", ""),
                    "published_at": sn.get("publishedAt", ""),
                    "description": sn.get("description", "")[:200],
                })
        next_page = data.get("nextPageToken")
        if not next_page:
            break

    # 3. Statistics + duration in batches of 50
    details: dict[str, dict] = {}
    for i in range(0, len(stubs), 50):
        ids = ",".join(v["id"] for v in stubs[i:i+50])
        r = requests.get(f"{_YT}/videos", params={
            "part": "statistics,contentDetails",
            "id": ids, "key": api_key,
        }, timeout=20)
        r.raise_for_status()
        for item in r.json().get("items", []):
            st = item.get("statistics", {})
            cd = item.get("contentDetails", {})
            details[item["id"]] = {
                "duration_seconds": _duration_secs(cd.get("duration", "")),
                "view_count":    int(st.get("viewCount",    0)),
                "like_count":    int(st.get("likeCount",    0)),
                "comment_count": int(st.get("commentCount", 0)),
            }

    # 4. Filter to Shorts
    shorts = []
    for v in stubs:
        d = details.get(v["id"], {})
        secs = d.get("duration_seconds", 0)
        if _is_short(secs, v["title"], v["description"]):
            shorts.append({
                "id": v["id"],
                "title": v["title"],
                "published_at": v["published_at"],
                "duration_seconds": secs,
                "view_count":    d.get("view_count", 0),
                "like_count":    d.get("like_count", 0),
                "comment_count": d.get("comment_count", 0),
                "url": f"https://www.youtube.com/shorts/{v['id']}",
            })

    return channel_stats, shorts


# ── LLM categorisation ────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = f"""You are a content classifier for a Telugu YouTube Shorts channel.

Classify each video title into EXACTLY ONE of these 16 content areas:
{', '.join(AREAS)}

Classification hints:
- "hidden mechanisms" → how everyday objects/engineering systems ACTUALLY work
  (ship anchors, PAPI runway lights, aircraft carrier wire, nuclear reactor water, GPS/Einstein)
- "psychology studies" → a named psychology experiment (Milgram, Stanford Prison, Rosenhan…)
- "psychology" → general human behaviour, mental health, cognitive topics
- "latest science news" → recent discoveries/breakthroughs announced in the news
- "science" → physics, chemistry, biology explained
- "world cultures" → cultural practices, rituals, traditions from around the world
- "inspiring people" → one person's extraordinary story or achievement
- "interesting events" → dramatic events, records, unusual occurrences that don't fit elsewhere

Return ONLY a JSON array — no markdown, no explanation:
[{{"title": "...", "area": "exact area from the list"}}]"""


def categorize_shorts(shorts: list[dict]) -> list[dict]:
    if not shorts:
        return []

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        temperature=0.05,
    )

    categorized: list[dict] = []
    for i in range(0, len(shorts), 50):
        batch = shorts[i:i+50]
        titles_text = "\n".join(f"{j+1}. {v['title']}" for j, v in enumerate(batch))
        response = llm.invoke([
            SystemMessage(content=_CLASSIFY_SYSTEM),
            HumanMessage(content=f"Classify these {len(batch)} titles:\n{titles_text}"),
        ])
        raw = response.content
        if isinstance(raw, list):
            raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
        try:
            if "```" in raw:
                raw = raw.split("```json")[1].split("```")[0] if "```json" in raw else raw.split("```")[1].split("```")[0]
            parsed = json.loads(raw[raw.find("["):raw.rfind("]")+1])
            for j, item in enumerate(parsed):
                if j < len(batch):
                    area = item.get("area", "interesting events").lower().strip()
                    if area not in AREAS:
                        area = "interesting events"
                    categorized.append({**batch[j], "area": area})
        except Exception:
            for v in batch:
                categorized.append({**v, "area": "interesting events"})
        print(f"[Channel Analysis] Classified batch {i//50 + 1} / {(len(shorts)-1)//50 + 1}")

    return categorized


# ── Stats aggregation ────────────────────────────────────────────────────────

def compute_area_stats(categorized: list[dict]) -> dict:
    buckets: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "total_views": 0, "total_likes": 0,
        "total_comments": 0, "videos": [],
    })
    for s in categorized:
        area = s.get("area", "interesting events")
        b = buckets[area]
        b["count"]          += 1
        b["total_views"]    += s.get("view_count", 0)
        b["total_likes"]    += s.get("like_count", 0)
        b["total_comments"] += s.get("comment_count", 0)
        b["videos"].append(s)

    result = {}
    for area, b in buckets.items():
        n = b["count"] or 1
        result[area] = {
            "count":          b["count"],
            "total_views":    b["total_views"],
            "avg_views":      round(b["total_views"] / n),
            "total_likes":    b["total_likes"],
            "avg_likes":      round(b["total_likes"] / n),
            "total_comments": b["total_comments"],
            "avg_comments":   round(b["total_comments"] / n),
            "top_videos":     sorted(b["videos"], key=lambda x: x.get("view_count", 0), reverse=True)[:5],
        }
    return result


# ── Main node ────────────────────────────────────────────────────────────────

def analyze_channel_node(state: dict) -> dict:
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return {
            "channel_analysis": None,
            "error": "NO_API_KEY",
        }

    try:
        print(f"\n[Channel Analysis] Fetching videos from @{CHANNEL_HANDLE}...")
        channel_stats, shorts = fetch_channel_data(api_key)
        print(f"[Channel Analysis] {len(shorts)} Shorts found")

        print(f"[Channel Analysis] Categorising with LLM...")
        categorized = categorize_shorts(shorts)

        area_stats = compute_area_stats(categorized)
        top_overall = sorted(categorized, key=lambda x: x.get("view_count", 0), reverse=True)[:10]

        result = {
            "channel_url":    CHANNEL_URL,
            "channel_stats":  channel_stats,
            "total_shorts":   len(categorized),
            "area_stats":     area_stats,
            "top_overall":    top_overall,
            "analyzed_at":    datetime.now(timezone.utc).isoformat(),
        }
        ANALYSIS_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Channel Analysis] Saved → {ANALYSIS_FILE.name}")
        return {"channel_analysis": result, "error": None}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"channel_analysis": None, "error": str(e)}


def load_channel_analysis() -> dict | None:
    if not ANALYSIS_FILE.exists():
        return None
    try:
        return json.loads(ANALYSIS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
