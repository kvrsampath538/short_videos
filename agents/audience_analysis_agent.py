import json
import os
from datetime import datetime, timezone
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tools import get_tavily_search_tool, search_youtube_shorts

AREAS = [
    "history", "science", "latest science news", "geography", "nature", "space",
    "psychology", "psychology studies", "technology", "interesting events", "inspiring people",
    "archaeology news", "aviation reports", "research papers",
    "world cultures", "hidden mechanisms",
]

INSIGHTS_FILE = Path(__file__).parent.parent / "audience_insights.json"

CHANNEL_URL = "https://www.youtube.com/@Worldaffairs0138"

SYSTEM_PROMPT = f"""You are a YouTube audience intelligence analyst specialising in Telugu-language
short-form content.

MISSION: Analyse current public interest on YouTube Shorts for a Telugu-speaking audience and score
each content area so the idea generator can prioritise the most attractive topics.

CHANNEL: {CHANNEL_URL}
This is a Telugu educational/informational Shorts channel covering world affairs, science,
history, psychology, hidden mechanisms, and cultural facts.

THE 16 CONTENT AREAS YOU MUST SCORE:
  history · science · latest science news · geography · nature · space
  psychology · psychology studies · technology · interesting events · inspiring people
  archaeology news · aviation reports · research papers
  world cultures · hidden mechanisms

SCORING (1–10 for each area):
  10  = Viral demand right now among Telugu Shorts viewers; existing videos get 500k+ views
  8–9 = Strong consistent demand; multiple recent videos performing well
  6–7 = Moderate interest; good niche
  4–5 = Average; competitive or declining
  1–3 = Low current interest or very saturated

RESEARCH APPROACH — search in this sequence:
1. Search YouTube for trending Telugu educational/informational Shorts RIGHT NOW
2. Search for what content the channel {CHANNEL_URL} covers (look at titles and topics)
3. Search for which specific topic angles are getting most views on similar Telugu channels
4. Search for topics that are trending in India on social media / news that map to the 19 areas
5. Search for YouTube Shorts performance patterns for Telugu-language science/history/psychology

WHAT RESONATES WITH TELUGU AUDIENCES (use as scoring context):
• Hidden systems and engineering explained simply — high curiosity pull
• Psychology experiments with shocking human-nature implications
• India-specific government/audit stories (high local relevance)
• Space and science discoveries explained in simple language
• World cultures and shocking practices unknown outside their country
• "Did you know?" style counterintuitive facts
• Inspiring individual stories (underdog overcomes impossible odds)
• Historical events 1900-onwards with irony or dark twist

OUTPUT FORMAT — return valid JSON only (no markdown):
{{
  "area_scores": {{
    "history": <1-10>,
    "science": <1-10>,
    "latest science news": <1-10>,
    "geography": <1-10>,
    "nature": <1-10>,
    "space": <1-10>,
    "psychology": <1-10>,
    "psychology studies": <1-10>,
    "technology": <1-10>,
    "interesting events": <1-10>,
    "inspiring people": <1-10>,
    "archaeology news": <1-10>,
    "aviation reports": <1-10>,
    "research papers": <1-10>,
    "world cultures": <1-10>,
    "hidden mechanisms": <1-10>
  }},
  "top_areas": ["area1", "area2", "area3", "area4", "area5"],
  "trending_angles": [
    "specific trending angle or topic with strong current interest",
    ...up to 8 angles...
  ],
  "channel_pattern": "2-3 sentences describing what topics this channel covers and what style works for them",
  "audience_insights": "3-4 sentences about what Telugu YouTube Shorts audiences want right now — be specific",
  "search_evidence": "Key data points from your searches that support the scores"
}}"""


def load_audience_insights() -> dict | None:
    """Load saved audience insights, or return None if not yet analysed."""
    if not INSIGHTS_FILE.exists():
        return None
    try:
        return json.loads(INSIGHTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def analyze_audience_interest_node(state: dict) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        temperature=0.3,
    )
    web_tool     = get_tavily_search_tool(max_results=5)
    yt_tool      = search_youtube_shorts
    llm_with_tools = llm.bind_tools([web_tool, yt_tool])

    user_message = (
        "Analyse current YouTube Shorts trends for Telugu-speaking audiences. "
        "Run these searches in sequence:\n"
        f"1. YouTube search: 'Telugu shorts viral trending educational 2025'\n"
        f"2. YouTube search: 'Telugu shorts psychology hidden facts science 2025'\n"
        f"3. Web search: 'worldaffairs0138 youtube channel Telugu shorts topics'\n"
        f"4. Web search: 'Telugu YouTube Shorts highest views science history psychology 2025'\n"
        f"5. Web search: 'India trending topics YouTube Shorts Telugu audience interest 2025'\n\n"
        "Then score all 19 content areas (1–10) for current Telugu audience interest."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    print(f"\n[Audience Analysis] Starting Telugu YouTube audience analysis...")

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
            tool_name = tc.get("name", "")
            print(f"[Audience Analysis] {tool_name}: '{query[:70]}'")
            if tool_name == "search_youtube_shorts":
                tool_result = yt_tool.invoke(args)
            else:
                tool_result = web_tool.invoke(args)
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
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception as e:
        print(f"[Audience Analysis] JSON parse error: {e} — using defaults")
        parsed = {
            "area_scores": {a: 5 for a in AREAS},
            "top_areas": ["hidden mechanisms", "psychology", "space", "history", "world cultures"],
            "trending_angles": [],
            "channel_pattern": "Analysis failed — defaults applied.",
            "audience_insights": "Could not complete analysis. Defaults applied.",
            "search_evidence": raw[:300] if raw else "",
        }

    # Fill any missing area scores with 5
    scores = parsed.setdefault("area_scores", {})
    for area in AREAS:
        scores.setdefault(area, 5)

    result = {
        **parsed,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "channel": CHANNEL_URL,
    }

    INSIGHTS_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Audience Analysis] Saved to {INSIGHTS_FILE.name}")
    print(f"[Audience Analysis] Top areas: {result.get('top_areas', [])}")

    return {"audience_insights": result}
