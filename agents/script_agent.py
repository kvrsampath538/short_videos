import json
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


def _repair_truncated_json(raw: str) -> dict | None:
    """
    When max_tokens cuts the response mid-JSON, try to salvage whatever scenes
    were completed before the truncation point.
    """
    # Find the outermost { … } even if truncated
    start = raw.find("{")
    if start == -1:
        return None
    fragment = raw[start:]

    # Extract the title and hook if present
    title = ""
    hook  = ""
    m = re.search(r'"title"\s*:\s*"([^"]*)"', fragment)
    if m:
        title = m.group(1)
    m = re.search(r'"hook"\s*:\s*"([^"]*)"', fragment)
    if m:
        hook = m.group(1)

    # Extract all fully-formed scene objects
    scenes = []
    for m in re.finditer(r'\{\s*"scene_number"\s*:.*?"english_note"\s*:\s*"[^"]*"\s*\}',
                         fragment, re.DOTALL):
        try:
            scenes.append(json.loads(m.group()))
        except Exception:
            pass

    if not scenes:
        return None

    print(f"[Script Agent] Repaired {len(scenes)} scenes from truncated response")
    return {
        "title": title,
        "hook": hook,
        "scenes": scenes,
        "call_to_action": "ఇలాంటి facts కోసం follow చేయండి!",
        "total_duration": sum(s.get("duration_seconds", 5) for s in scenes),
        "_repaired": True,
    }

SYSTEM_PROMPT = """మీరు ఒక నిపుణమైన Telugu YouTube Shorts స్క్రిప్ట్ రైటర్. మీకు వైరల్ శార్ట్స్ రాయడంలో విస్తారమైన అనుభవం ఉంది.

You write Telugu scripts for YouTube Shorts that follow proven viral strategies:

HOOK STRATEGIES (first 3 seconds — make viewer stop scrolling):
- Start with a shocking fact or counter-intuitive statement
- Use "మీకు తెలుసా..." (Did you know...) or "ఇది చూసి నమ్మలేరు..." (You won't believe this...)
- Pose a curiosity question: "ఎప్పుడైనా ఆలోచించారా..." (Have you ever wondered...)
- Bold claim: "ఇది మారుస్తుంది మీ thinking ని పూర్తిగా..."

NARRATION RULES:
- Conversational Telugu — avoid overly formal/literary language
- Short punchy sentences — one idea per line
- Build suspense scene by scene (open loops)
- Use relatable analogies to explain science
- End each scene with a micro-hook to keep watching
- Emotional payoff at the end: awe, surprise, or "mind blown" moment

CALL TO ACTION (last 3 seconds):
- "ఇలాంటి facts కోసం follow చేయండి!"
- "మీ friends కి share చేయండి — వాళ్ళు shock అవుతారు!"
- "Comment లో చెప్పండి — మీకు తెలుసా ఇది?"

Return ONLY a JSON object:
{
  "title": "Telugu short title with emoji — curiosity-driving (max 60 chars)",
  "hook": "Opening line in Telugu (first 3 seconds — must be irresistible)",
  "scenes": [
    {
      "scene_number": 1,
      "duration_seconds": 5,
      "telugu_script": "Exact Telugu words spoken in this scene",
      "transliteration": "Telugu written in English letters for reference",
      "english_note": "Brief note on what this scene communicates"
    }
  ],
  "call_to_action": "Final CTA line in Telugu",
  "total_duration": <total seconds>
}"""


def script_agent_node(state: dict) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        temperature=0.7,
    )

    best_idea   = state.get("best_idea", {})
    scenes      = state.get("image_prompts", [])
    gen_images  = state.get("generated_images", [])

    if not best_idea:
        print("[Script Agent] No best idea in state.")
        return {"telugu_script": {}}

    # Build a scene-by-scene summary to give the agent full context
    scene_lines = []
    for s in scenes:
        n   = s.get("scene_number", "?")
        dur = s.get("duration_seconds", 5)
        nar = s.get("narration", "")
        scene_lines.append(f"Scene {n} [{dur}s]: {nar}")
    scene_summary = "\n".join(scene_lines)

    total_duration = sum(s.get("duration_seconds", 5) for s in scenes)

    user_message = (
        f"Write a viral Telugu YouTube Shorts script for this scientific concept.\n\n"
        f"**Concept title:** {best_idea.get('title', '')}\n"
        f"**Concept:** {best_idea.get('concept', '')}\n\n"
        f"**Storyboard ({total_duration}s total):**\n{scene_summary}\n\n"
        f"Rules:\n"
        f"- Match scene count and durations exactly to the storyboard above\n"
        f"- Hook must make someone STOP scrolling in 3 seconds\n"
        f"- Use natural conversational Telugu (not overly formal)\n"
        f"- Build curiosity and wonder scene by scene\n"
        f"- End with a mind-blown moment + strong CTA\n"
        f"- Total narration must fit within {total_duration} seconds when spoken"
    )

    print(f"\n[Script Agent] Writing Telugu script for: \"{best_idea.get('title', '')}\"")

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw = response.content
    if isinstance(raw, list):
        raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    print(f"[Script Agent] Raw response length: {len(raw)} chars")

    parsed = None
    try:
        extract = raw
        if "```json" in extract:
            extract = extract.split("```json")[1].split("```")[0].strip()
        elif "```" in extract:
            extract = extract.split("```")[1].split("```")[0].strip()
        start = extract.find("{")
        end   = extract.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(extract[start:end])
    except Exception as e:
        print(f"[Script Agent] JSON parse error: {e}")
        parsed = _repair_truncated_json(raw)

    if not parsed:
        print("[Script Agent] Falling back to raw text storage.")
        return {"telugu_script": {"raw": raw}}

    # Pretty-print the script
    print(f"\n{'─'*60}")
    print(f"  TELUGU SCRIPT")
    print(f"{'─'*60}")
    print(f"  Title : {parsed.get('title', '')}")
    print(f"  Hook  : {parsed.get('hook', '')}")
    print()
    for sc in parsed.get("scenes", []):
        n   = sc.get("scene_number", "?")
        dur = sc.get("duration_seconds", "?")
        print(f"  Scene {n} [{dur}s]")
        print(f"    Telugu : {sc.get('telugu_script', '')}")
        print(f"    Roman  : {sc.get('transliteration', '')}")
        print()
    print(f"  CTA: {parsed.get('call_to_action', '')}")
    print(f"{'─'*60}")

    return {"telugu_script": parsed}
