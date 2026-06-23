import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tools import search_youtube_shorts

# ── Bulk pre-filter (runs before ideas are shown to user) ─────────────────────

_FILTER_SYSTEM_PROMPT = """You are a YouTube Shorts content saturation analyst.

You receive a list of YouTube Shorts ideas. For EACH idea:
1. Use search_youtube_shorts to search for that idea's specific angle on YouTube
2. Assign a saturation score:
   - 1 (LOW): Very few Shorts on this specific angle, or existing ones have low views — great gap
   - 2 (MEDIUM): Some coverage exists but the unique angle still has room to compete
   - 3 (HIGH): Many popular Shorts (100k+ views) already cover this exact concept thoroughly

Search the SPECIFIC ANGLE, not just the broad topic.
Good search: "ant queen enslaves other species" not just "ants".

Return valid JSON only — no markdown, no extra text:
{
  "results": [
    {"title": "exact idea title", "saturation": 1, "reason": "one sentence"},
    {"title": "exact idea title", "saturation": 3, "reason": "one sentence"},
    ...
  ]
}"""


def bulk_saturation_filter(ideas: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Search YouTube for each idea, score saturation 1-3.
    Returns (fresh_ideas, all_scored_ideas):
      - fresh_ideas: only LOW (1) and MEDIUM (2) ideas
      - all_scored_ideas: every idea with saturation_score and saturation_reason added
    """
    if not ideas:
        return [], []

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        temperature=0.1,
    )
    llm_with_tools = llm.bind_tools([search_youtube_shorts])

    ideas_text = "\n".join(
        f"{i+1}. **{idea.get('title', '')}**: {idea.get('concept', '')}"
        for i, idea in enumerate(ideas)
    )
    user_message = (
        f"Check YouTube saturation for each of these {len(ideas)} Shorts ideas.\n"
        f"Search YouTube for each idea's specific angle, score 1-3, return results JSON.\n\n"
        f"{ideas_text}"
    )

    messages = [
        SystemMessage(content=_FILTER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    print(f"\n[Saturation Filter] Checking {len(ideas)} ideas...")
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    while response.tool_calls:
        for tc in response.tool_calls:
            args = tc.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"query": args}
            query = args.get("query", "") if isinstance(args, dict) else ""
            print(f"[Saturation Filter] YouTube search: '{query[:70]}'")
            tool_result = search_youtube_shorts.invoke(args)
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc.get("id", "")))
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    raw = response.content
    if isinstance(raw, list):
        raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)

    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        parsed  = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        results = parsed.get("results", [])
    except Exception:
        print("[Saturation Filter] JSON parse failed — passing all ideas as MEDIUM")
        all_scored = [dict(idea, saturation_score=2, saturation_reason="") for idea in ideas]
        return all_scored, all_scored

    # Match LLM results back to original ideas by title (fuzzy: first 40 chars)
    score_map: dict[str, dict] = {}
    for r in results:
        key = (r.get("title") or "")[:40].lower()
        score_map[key] = r

    def _lookup(title: str) -> dict:
        key = title[:40].lower()
        if key in score_map:
            return score_map[key]
        for k, v in score_map.items():
            if k[:25] in key or key[:25] in k:
                return v
        return {}

    labels = {1: "🟢 LOW", 2: "🟡 MEDIUM", 3: "🔴 HIGH"}
    filtered   = []
    all_scored = []
    for idea in ideas:
        title  = idea.get("title", "")
        result = _lookup(title)
        score  = int(result.get("saturation", 2))
        reason = result.get("reason", "")
        print(f"[Saturation Filter] {labels.get(score,'?')}  '{title}'")
        if reason:
            print(f"                    {reason}")
        scored = dict(idea, saturation_score=score, saturation_reason=reason)
        all_scored.append(scored)
        if score <= 2:
            filtered.append(scored)
        else:
            print(f"[Saturation Filter] ✗ REMOVED (HIGH saturation)")

    print(f"[Saturation Filter] {len(filtered)}/{len(ideas)} ideas passed\n")
    return filtered, all_scored

SYSTEM_PROMPT = """You are a YouTube content strategist and saturation analyst.

You receive 3 YouTube Shorts ideas. For each one, use the search_youtube_shorts tool to check how
saturated that topic is on YouTube.

Saturation scoring:
- Score 1 (LOW): Very few videos, or existing ones have low views, or the angle is unique/niche
- Score 2 (MEDIUM): Some coverage but not overwhelming; room for a fresh take
- Score 3 (HIGH): Many popular videos (100k+ views) already cover this exact concept thoroughly

After checking all 3 ideas:
1. If ALL 3 score HIGH saturation → reject all, return approved=false so new ideas are generated
2. Otherwise → pick the SINGLE BEST idea (lowest saturation + highest interestingness) and return it

Respond with valid JSON in exactly one of these formats:

Rejection (all saturated):
{"approved": false, "reason": "All topics are heavily covered: [brief explanation per topic]"}

Approval (best idea selected):
{
  "approved": true,
  "best_idea": {
    "title": "the idea title",
    "concept": "the full concept explanation",
    "why_best": "why this idea stands out vs the others (saturation + uniqueness)"
  }
}"""


def freshness_check_agent_node(state: dict) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        temperature=0.2,
    )
    llm_with_tools = llm.bind_tools([search_youtube_shorts])

    ideas = state.get("research_ideas", [])
    if not ideas:
        return {
            "freshness_approved": False,
            "best_idea": None,
            "rejection_reason": "No ideas received from research agent.",
        }

    ideas_text = "\n".join(
        f"{i+1}. **{idea.get('title', 'Untitled')}**: {idea.get('concept', '')}"
        for i, idea in enumerate(ideas)
    )

    user_message = (
        f"Check YouTube saturation for each of these 3 ideas, then pick the single best one:\n\n"
        f"{ideas_text}\n\n"
        f"Search YouTube for each idea individually, score their saturation, then select the winner."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    print(f"\n[Freshness Agent] Checking saturation for {len(ideas)} ideas...")

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    while response.tool_calls:
        for tool_call in response.tool_calls:
            query = tool_call["args"].get("query", "")
            print(f"[Freshness Agent] Searching YouTube: '{query[:70]}'")
            tool_result = search_youtube_shorts.invoke(tool_call["args"])
            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    raw = response.content
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        approved = parsed.get("approved", False)
        best_idea = parsed.get("best_idea", None)
        reason = parsed.get("reason", "")
    except Exception:
        # Conservative fallback: pick first idea
        approved = True
        best_idea = ideas[0] if ideas else None
        reason = ""

    if approved and best_idea:
        print(f"[Freshness Agent] BEST IDEA SELECTED: \"{best_idea.get('title', '')}\"")
        print(f"  Reason: {best_idea.get('why_best', '')}")
    else:
        print(f"[Freshness Agent] ALL REJECTED — {reason}")

    return {
        "freshness_approved": approved,
        "best_idea": best_idea if approved else None,
        "rejection_reason": reason if not approved else "",
        "messages": messages,
    }
