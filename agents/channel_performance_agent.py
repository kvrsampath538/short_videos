import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ollama_client import get_haiku_llm
from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
from vectordb import get_db
from settings import get_channel_handle

N_RECENT = 25
N_TOP    = 25


def _fmt(n: int) -> str:
    if n >= 1_000_000: return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:     return f"{n / 1_000:.1f}K"
    return str(n)


def _metrics(shorts: list[dict]) -> dict:
    if not shorts:
        return {}
    views    = [s.get("view_count",    0) for s in shorts]
    likes    = [s.get("like_count",    0) for s in shorts]
    comments = [s.get("comment_count", 0) for s in shorts]
    avg_v = sum(views)    / len(views)
    avg_l = sum(likes)    / len(likes)
    avg_c = sum(comments) / len(comments)
    eng   = ((avg_l + avg_c) / avg_v * 100) if avg_v > 0 else 0
    area_counts: dict[str, int] = {}
    for s in shorts:
        a = s.get("area", "other")
        area_counts[a] = area_counts.get(a, 0) + 1
    return {
        "count":           len(shorts),
        "avg_views":       round(avg_v),
        "avg_likes":       round(avg_l),
        "avg_comments":    round(avg_c),
        "engagement_rate": round(eng, 2),
        "max_views":       max(views),
        "min_views":       min(views),
        "area_breakdown":  dict(sorted(area_counts.items(), key=lambda x: x[1], reverse=True)),
    }


def _short_lines(shorts: list[dict], n: int = 25) -> str:
    lines = []
    for s in shorts[:n]:
        lines.append(
            f"  • {s.get('title', '?')} | area={s.get('area', '?')}"
            f" | views={_fmt(s.get('view_count', 0))}"
            f" | likes={_fmt(s.get('like_count', 0))}"
            f" | comments={s.get('comment_count', 0)}"
            f" | date={s.get('published_at', '')[:10]}"
        )
    return "\n".join(lines)


def run_performance_analysis() -> dict:
    db = get_db()
    meta = db.load_config("channel_analysis")
    if not meta:
        raise FileNotFoundError(
            "No channel analysis found. Run 'My Channel Analysis' first."
        )
    all_shorts: list[dict] = db.get_all("channel")
    if not all_shorts:
        raise ValueError(
            "Channel analysis is outdated — individual video data is missing. "
            "Please re-run 'My Channel Analysis' to refresh."
        )

    # Recent N: sorted newest-first
    dated = [s for s in all_shorts if s.get("published_at")]
    dated.sort(key=lambda x: x["published_at"], reverse=True)
    recent = dated[:N_RECENT]

    # Top performers: highest views, excluding recent
    recent_keys = {s.get("url", s.get("title", "")) for s in recent}
    by_views    = sorted(all_shorts, key=lambda x: x.get("view_count", 0), reverse=True)
    top         = [s for s in by_views if s.get("url", s.get("title", "")) not in recent_keys][:N_TOP]

    rm = _metrics(recent)
    tm = _metrics(top)

    channel_stats = meta.get("channel_stats", {})
    subs          = channel_stats.get("subscriber_count", 0)
    total         = meta.get("total_shorts", 0)

    views_gap_pct = round((tm["avg_views"] / max(rm["avg_views"], 1) - 1) * 100)

    prompt = f"""You are a YouTube Shorts analytics expert. Analyze performance for a Telugu educational Shorts channel.

CHANNEL: {get_channel_handle()} · {subs:,} subscribers · {total} total Shorts

━━━ RECENT {len(recent)} SHORTS (newest first) ━━━
{_short_lines(recent)}

RECENT METRICS:
  Avg views: {_fmt(rm['avg_views'])}  |  Avg likes: {_fmt(rm['avg_likes'])}  |  Engagement rate: {rm['engagement_rate']}%
  Best: {_fmt(rm['max_views'])} views  |  Worst: {_fmt(rm['min_views'])} views
  Area breakdown: {', '.join(f"{a}={c}" for a, c in list(rm['area_breakdown'].items())[:6])}

━━━ TOP {len(top)} ALL-TIME HIGH PERFORMERS ━━━
{_short_lines(top)}

HIGH PERFORMER METRICS:
  Avg views: {_fmt(tm['avg_views'])}  |  Avg likes: {_fmt(tm['avg_likes'])}  |  Engagement rate: {tm['engagement_rate']}%
  Best: {_fmt(tm['max_views'])} views  |  Worst: {_fmt(tm['min_views'])} views
  Area breakdown: {', '.join(f"{a}={c}" for a, c in list(tm['area_breakdown'].items())[:6])}

PERFORMANCE GAP: Top performers average {views_gap_pct}% more views than recent shorts.

Write your analysis under these exact headings (## heading):

## Overall Performance
2-3 sentences. Are recent shorts performing well or poorly vs historical top performers? Quote exact numbers.

## What's Working in High Performers
3-5 specific patterns — which content areas, title styles, and topic types drive the most views. Reference actual video titles from the data.

## What's Missing in Recent Shorts
3-5 specific gaps — content areas underused in recent vs high performers, title approaches absent, topics being avoided. Be specific with numbers.

## Numbers at a Glance
Bullet list comparing: avg views, avg likes, engagement rate, top area for both groups.

## 5 Actionable Recommendations
Numbered list. Each must be specific and cite numbers (e.g. "Produce more 'hidden mechanisms' videos — they average {_fmt(tm['avg_views'])} views vs your recent {_fmt(rm['avg_views'])} average"). No generic advice.

Be direct and data-driven. Every point must reference actual titles or numbers from the data above."""

    print("[Channel Performance] Running LLM analysis...")
    llm = get_haiku_llm(temperature=0.3)
    resp = llm.invoke([HumanMessage(content=prompt)])
    text = resp.content
    if isinstance(text, list):
        text = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)

    result = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "channel_stats":   channel_stats,
        "recent":          recent,
        "top_performers":  top,
        "recent_metrics":  rm,
        "top_metrics":     tm,
        "views_gap_pct":   views_gap_pct,
        "analysis":        text,
    }
    get_db().save_config("channel_performance", result)
    print("[Channel Performance] Saved to VectorDB (channel_performance)")
    return result


def load_performance() -> dict | None:
    return get_db().load_config("channel_performance")
