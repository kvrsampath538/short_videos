import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are a visual storytelling expert creating storyboards for Google Flow's
Nano Banana Pro model — a cinematic AI image/video generation model that excels at rich,
motion-aware, cinematically directed prompts.

Given a scientific concept, create a storyboard of scene prompts that explain the idea
visually in a YouTube Short of NO MORE than 60 seconds.

Structure rules:
- Total duration must not exceed 60 seconds
- Each scene is 4–7 seconds long (aim for 8–12 scenes)
- Scene 1: Hook — immediately grab attention with the surprising element
- Middle scenes: Build the explanation step by step, one concept per scene
- Final scene: The "wow" payoff / mind-blowing conclusion + call to action

Nano Banana Pro prompt style — each image_prompt MUST include ALL of these elements:
1. SUBJECT: What is shown (object, person, phenomenon)
2. ACTION/MOTION: What is moving or happening (camera pan, zoom, particle movement)
3. CAMERA: Shot type and movement (extreme close-up, aerial drone shot, slow push-in)
4. LIGHTING: Quality and color (golden hour, dramatic side lighting, neon glow, bioluminescent)
5. STYLE: Visual treatment (photorealistic, cinematic 4K, high-speed slow motion, scientific illustration)
6. ATMOSPHERE/MOOD: Emotional tone (tense, awe-inspiring, mysterious, triumphant)

Example of a good Nano Banana Pro prompt:
"Extreme close-up of a starter pistol muzzle firing, thick white smoke billowing outward in
ultra slow motion, visible sound wave rings expanding through the air like ripples on water,
dramatic side lighting with deep shadows and electric blue highlights, cinematic 4K photorealistic,
tense and electric atmosphere, slow camera push-in toward the barrel"

Return ONLY a JSON object:
{
  "title": "YouTube Short title (max 60 chars, punchy and curiosity-driving)",
  "hook": "The opening spoken line — must create instant curiosity in 3 seconds",
  "total_duration": <sum of all scene durations in seconds>,
  "scenes": [
    {
      "scene_number": 1,
      "duration_seconds": 5,
      "narration": "Concise punchy spoken text for this scene",
      "image_prompt": "Full Nano Banana Pro optimized prompt with all 6 elements"
    }
  ]
}"""


def image_prompt_agent_node(state: dict) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        temperature=0.8,
    )

    best_idea = state.get("best_idea", {})
    if not best_idea:
        print("[Image Prompt Agent] No best idea found in state.")
        return {"image_prompts": []}

    title = best_idea.get("title", "Unknown")
    concept = best_idea.get("concept", "")

    user_message = (
        f"Create a YouTube Shorts storyboard (max 60 seconds) for this scientific concept.\n\n"
        f"**Title:** {title}\n"
        f"**Concept:** {concept}\n\n"
        f"Each image_prompt must be fully optimized for Google Flow's Nano Banana Pro model — "
        f"include subject, action/motion, camera direction, lighting, style, and atmosphere. "
        f"Make every scene visually stunning and self-explanatory even without narration."
    )

    print(f"\n[Image Prompt Agent] Generating Nano Banana Pro storyboard for: \"{title}\"")

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    raw = response.content
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        scenes = parsed.get("scenes", [])
        total = parsed.get("total_duration", sum(s.get("duration_seconds", 5) for s in scenes))
        short_title = parsed.get("title", title)
        hook = parsed.get("hook", "")
    except Exception as e:
        print(f"[Image Prompt Agent] JSON parse error: {e}")
        return {"image_prompts": [{"raw": raw}]}

    print(f"[Image Prompt Agent] {len(scenes)} scenes — {total}s total")
    print(f"  Title : \"{short_title}\"")
    print(f"  Hook  : \"{hook}\"")
    for scene in scenes:
        n = scene.get("scene_number", "?")
        dur = scene.get("duration_seconds", "?")
        narration = scene.get("narration", "")
        prompt = scene.get("image_prompt", "")
        print(f"\n  Scene {n} [{dur}s]: {narration}")
        print(f"    PROMPT: {prompt[:120]}...")

    return {"image_prompts": scenes}
