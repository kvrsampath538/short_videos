import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── Intent detection ──────────────────────────────────────────────────────────

_INTENT_SYSTEM = """You are a routing agent for a Telugu YouTube Shorts AI assistant.

Read the user's message, the recent chat history, and the currently active idea (if any),
then return a JSON object identifying the user's intent.

AVAILABLE ACTIONS:
  chat               — Answer a question, explain, discuss strategy or creative direction
  generate_ideas     — Generate viral YouTube Shorts ideas (no specific topic, or "ideas about X")
  research_topic     — Deep-research a SPECIFIC topic to find the most viral angles
  check_saturation   — Check if a topic or idea is already saturated on YouTube Shorts
  generate_storyboard — Create a visual storyboard + Telugu script for an idea
  analyze_audience   — Run public interest analysis for Telugu-speaking YouTube audience

DECISION RULES:
  • "generate ideas", "find ideas", "suggest", "what should I make" → generate_ideas
  • "research", "dig into", "find angles on", "explore", "what's interesting about" → research_topic
  • "saturated", "check saturation", "covered on youtube", "already exists", "how competitive" → check_saturation
  • "storyboard", "create video", "make video", "script", "scenes for", "build video" → generate_storyboard
  • "trending", "audience interest", "analyze audience", "what performs well" → analyze_audience
  • Questions, opinions, explanations, "what do you think", general discussion → chat
  • When user says "this" / "it" / "the idea" → resolve from ACTIVE IDEA in context

Return ONLY valid JSON (no markdown, no explanation):
{
  "action": "<one of the 6 actions above>",
  "topic": "<extracted topic/subject, or null>",
  "idea_title": "<exact idea title if identifiable, or null>",
  "confidence": <0.0-1.0>
}"""


def detect_intent(
    user_message: str,
    chat_history: list[dict],
    current_idea: dict | None = None,
) -> dict:
    """Lightweight LLM call to classify user intent and extract topic."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        temperature=0.05,
    )

    # Summarise recent history (last 4 turns) for context
    history_lines = []
    for msg in chat_history[-8:]:
        role = msg.get("role", "")
        if role == "user":
            history_lines.append(f"User: {msg['content']}")
        elif role == "assistant":
            history_lines.append(f"Assistant: {msg.get('content', '')[:120]}")
    history_text = "\n".join(history_lines) or "(none)"

    idea_text = ""
    if current_idea:
        idea_text = f"\nACTIVE IDEA: {current_idea.get('title', '')} — {current_idea.get('concept', '')[:100]}"

    context = f"Recent chat history:\n{history_text}{idea_text}\n\nUser message: {user_message}"

    response = llm.invoke([
        SystemMessage(content=_INTENT_SYSTEM),
        HumanMessage(content=context),
    ])
    raw = response.content
    if isinstance(raw, list):
        raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)

    try:
        if "```" in raw:
            raw = raw.split("```")[1] if "json" not in raw else raw.split("```json")[1].split("```")[0]
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        return parsed
    except Exception:
        return {"action": "chat", "topic": None, "idea_title": None, "confidence": 0.5}


# ── Conversational LLM response ───────────────────────────────────────────────

_CHAT_SYSTEM = """You are a creative assistant for a Telugu YouTube Shorts channel.

CHANNEL CONTEXT:
The channel (https://www.youtube.com/@Worldaffairs0138) creates educational Telugu YouTube Shorts
covering: history, science, psychology experiments, hidden engineering mechanisms, world cultures,
India government stories, space, archaeology, aviation incidents, inspiring people, and more.

YOUR ROLE:
Help the creator think, refine, and strategise. You can:
• Discuss why certain topics go viral on YouTube Shorts
• Critique an idea's hook, angle, or title
• Suggest alternative framings for a concept
• Explain background context on a topic
• Give feedback on structure, pacing, emotional payoff
• Discuss content strategy for Telugu-speaking audiences

WHAT MAKES TELUGU SHORTS VIRAL:
• The HOOK must create a gap in knowledge the viewer must close ("I can't believe I didn't know this")
• COUNTERINTUITIVE facts — "everyone knows X, but actually Y"
• HIDDEN SYSTEMS — "this invisible mechanism has been running in plain sight"
• SPECIFIC verifiable numbers, names, dates — not vague claims
• Stories with an EMOTIONAL payoff — awe, outrage, or "mind blown"
• Short punchy sentences — one revelation per scene

When the user asks you to DO something (generate storyboard, check saturation, research), tell them
you're routing to the specialist agent. Keep your conversational answers concise and actionable."""


def chat_response(
    user_message: str,
    chat_history: list[dict],
    current_idea: dict | None = None,
) -> str:
    """Generate a conversational LLM response using full chat history."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        temperature=0.7,
    )

    messages: list = [SystemMessage(content=_CHAT_SYSTEM)]

    if current_idea:
        ctx = (
            f"The user is currently discussing this idea: "
            f"**{current_idea.get('title', '')}** — {current_idea.get('concept', '')}"
        )
        messages.append(SystemMessage(content=ctx))

    # Replay full chat history (up to last 20 messages for cost control)
    for msg in chat_history[-20:]:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            # Flatten rich message types to plain text for context
            text = msg.get("content", "")
            if msg.get("type") == "ideas" and msg.get("ideas"):
                idea_titles = ", ".join(i.get("title", "") for i in msg["ideas"])
                text = f"{text} Ideas: {idea_titles}"
            elif msg.get("type") == "saturation":
                status = "✅ fresh" if msg.get("approved") else "⚠️ saturated"
                text = f"{text} (result: {status})"
            messages.append(AIMessage(content=text))

    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)
    raw = response.content
    if isinstance(raw, list):
        raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    return raw
