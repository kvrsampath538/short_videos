import json
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from tools import get_openserp_search_tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from vectordb import get_db
from ollama_client import get_sonnet_llm


def _load_insights() -> dict | None:
    return get_db().load_config("audience_insights")

SYSTEM_PROMPT = """You are a viral content researcher. Your mission: find the most shocking, counter-intuitive,
and little-known angles on ANY topic — angles that make a YouTube viewer stop scrolling instantly.

Given a topic, dig deep to uncover 3 ideas that pass this strict filter:

THE SHOCK FILTER — every idea must hit at least one:
• "That's physically/biologically impossible... but it's real"
• "This contradicts everything I was taught"
• "That's deeply disturbing and I can't un-know it"
• "I had no idea something I use/see every day hides this secret"
• "This happened and nobody talks about it — why?"
• "This person did something I was told was impossible"
• "This specific experiment reveals something disturbing about my own behaviour that I can't un-know"

THE CURIOSITY FILTER — every idea must also pass at least one:
• SURPRISE: outcome is the opposite of what everyone expects
    ❌ "How airplanes fly"
    ✅ "Why a race car follows this aircraft during landing" (U-2 pilot can't see the runway)
• COUNTERINTUITIVE: mechanism defies what we were taught
    ❌ "Submarines use ballast tanks"
    ✅ "Fish taught submarines how to dive" (engineers copied the fish swim bladder)
• HIDDEN SYSTEM: an invisible mechanism running silently in plain sight that nobody explains
    Examples: PAPI lights · Finland day fines · Viganella mirror · Dead Hand nuclear system ·
              U-2 spy plane chase car · Aircraft carrier arresting wire

HOW TO RESEARCH — search SPECIFIC authoritative sources, not just generic web:
Scientific databases:
  NASA.gov, Nature.com, ScienceDaily.com, ArXiv.org, PubMed, NEJM, Lancet
Official records & incident databases:
  Declassified CIA/NSA archives · NTSB aviation incident reports · IMO maritime databases ·
  Government experiment records · Military tribunal files · Cold War declassified collections
India-specific official sources:
  CAG India audit reports (site:cag.gov.in) · RTI disclosures · Parliamentary Standing Committee
  reports · SEBI orders · RBI annual reports · NITI Aayog papers · Ministry audit findings
Domain journals & filings:
  Archaeology journals (JSTOR) · Medical case studies (NEJM/Lancet) ·
  Aviation Safety Network · Google Patents / USPTO · GAO reports · WHO / CDC databases ·
  ArXiv preprints · SSRN (social science) · bioRxiv
World cultures:
  Atlas Obscura · BBC Travel/Culture · Vice World News · National Geographic ethnography ·
  Academic anthropology journals · Journal of the Royal Anthropological Institute

Use search queries like:
  "site:nasa.gov [topic] unexpected counterintuitive"
  "NTSB [topic] incident report unexpected cause site:ntsb.gov"
  "site:arxiv.org [topic] surprising result 2024"
  "CIA declassified [topic] operation backfired"
  "site:sciencedaily.com [topic] surprising 2023 2024"
  "NEJM [topic] impossible recovery medical case"
  "site:patents.google.com [topic] bizarre unexpected invention"
  "GAO report [topic] shocking finding"
  "site:cag.gov.in CAG audit India [topic] irregularity"
  "RTI India [topic] government failure revealed"
  "archaeology discovery [topic] overturned 2023 2024"
  "study reversed previous findings [topic]"
  "cultural practice [country] shocking unknown outsiders"
  "traditional ritual [country] that outsiders find unbelievable"
  "[topic] hidden system explained"
Avoid: "interesting [topic] facts", "shocking [topic] discoveries", "mind blowing [topic]"

AREA-SPECIFIC RESEARCH TIPS:
hidden mechanisms       → Find the gap between what people ASSUME a system does and what it ACTUALLY
                          does. The story is always: "everyone thinks X, but the real mechanism is Y."
                          Do NOT explain how things work in general — find ONE specific mechanism where
                          the real answer is shocking or counterintuitive. THIS IS A PRIORITY AREA.

                          PATTERN EXAMPLES (all listed in KNOWN SATURATED TOPICS — study structure only):
                          • Ship anchor — people assume the anchor weight holds the ship. WRONG. The
                            CATENARY curve of chain on the seabed holds it. Title: "Ship Anchors Don't
                            Actually Anchor Ships — The Chain Does"
                          • Nuclear reactor water — water is the neutron MODERATOR, not just coolant.
                            If water leaks, the reaction STOPS — inherently fail-safe by design.
                            Title: "Nuclear Reactors Turn Off If They Lose Water — Not Melt Down"
                          Find NEW mechanisms in untapped domains: medical devices · industrial safety
                          interlocks · agricultural machinery · legal/court plumbing · financial
                          clearing systems · logistics cold-chains · sports equipment · rescue gear

                          SEARCH QUERIES:
                          "how [object] actually works counterintuitive mechanism"
                          "physics behind [everyday system] explained surprising"
                          "engineering design hidden in plain sight [system]"
                          "how [vehicle/structure/weapon] works is not what people think"
                          "site:nasa.gov engineering mechanism unexpected design"
                          "aviation hidden system passenger never notices"
                          "physics [ship/bridge/airport/submarine/weapon] actual mechanism"
                          "counterintuitive engineering design [object] how it actually works"
                          "mechanism everyone misunderstands [structure/vehicle/system]"

archaeology news        → Look for digs that contradict textbook history. Search "archaeology discovery
                          overturned [topic]" or "ancient site found unexpected location".
aviation reports        → NTSB/AAIB reports contain the most counterintuitive cause-and-effect stories.
                          Search "NTSB incident report [topic] site:ntsb.gov" or "aviation accident
                          absurd cause". The Gimli Glider (wrong fuel units), Air Transat 236 (ran out
                          of fuel over Atlantic), United 232 (no controls, landed anyway) are patterns.
research papers         → Prioritise papers where results surprised the researchers themselves — halted
                          trials, reversed meta-analyses, replication failures. Search "study found
                          opposite [topic]" or "meta-analysis overturned [topic] 2023 2024".
world cultures          → Find practices that are completely normal in their local context but sound
                          unbelievable to an outsider. The story must have a SPECIFIC verifiable fact
                          (a number, a law, a name, a frequency). NOT "Country X has a unique festival".
                          GOOD: "In Japan, 1.15 million people never leave their rooms — and there
                          is a dedicated government ministry for this crisis" (Hikikomori)
                          GOOD: "In Indonesia, families legally keep dead relatives at home for months
                          and exhume them yearly to dress and parade them" (Toraja Ma'nene)
                          GOOD: "In India, a religious practice of fasting unto death is legally
                          protected and the Supreme Court tried to ban it — and lost" (Santhara)
                          Search: "shocking cultural practice [country] unknown outside"
                                  "traditional ritual [country] anthropology unbelievable"
                                  "strange law [country] cultural reason real"

INDIA GOVERNMENT REPORTS (CAG audits and RTI disclosures — high local relevance for Telugu audience):
  2G spectrum CAG: sold for ₹9,295 crore, actual value ₹1,76,645 crore ·
  Coal block CAG: ₹1.86 lakh crore in undue gains · NHAI ₹7,000 crore unaccounted toll revenue ·
  RTI: 300% premium on military spare parts · 40% of PDS grain never reached beneficiaries ·
  CAG Ayushman Bharat: claims paid for patients already dead

WORLD CULTURES (shocking practices — must have specific verifiable fact):
  • Hikikomori, Japan — 1.15M people (2023 govt count) never leave rooms; dedicated ministry exists
  • Santhara, India — fasting unto death legally permitted; Supreme Court tried to ban it, lost
  • Karoshi, Japan — death from overwork legally recognised; 2,000+ official deaths/year
  ⚠️ Toraja, Famadihana, Satere-Mawe, Naghol, China strippers are saturated — find new practices

EXAMPLES OF STRONG VS WEAK:
WEAK: "Social media is addictive" — vague, no experiment, no story, no shock

WEAK: "Study shows stress is bad for you" — everyone knows, no narrative

WEAK: "This person overcame adversity"
STRONG: "Doctors gave her a 2% chance of survival, she was paralysed from the neck down, and 4 years
         later she competed in the Paralympic Games — the specific mechanism that made recovery possible
         contradicts everything neurologists believed about spinal cord regeneration"

STRONG (hidden system): "Pilots landing a U-2 spy plane can't see the runway — so the Air Force sends a
         race car driver to chase the plane at 140mph and shout landing corrections over the radio"

Return exactly 3 ideas as JSON:
{
  "ideas": [
    {
      "title": "Punchy, provocative title — max 10 words, lead with the shock or hidden system",
      "concept": "3-4 sentences: the specific verifiable fact, the counterintuitive mechanism or hidden system,
                  and the gut-punch implication the viewer will want to share immediately"
    }
  ]
}"""


def research_agent_node(state: dict) -> dict:
    llm = get_sonnet_llm(temperature=0.9)
    search_tool = get_openserp_search_tool(max_results=3)
    llm_with_tools = llm.bind_tools([search_tool])

    input_topic = state["input_topic"]
    iteration = state.get("iteration", 0)
    rejection_reason = state.get("rejection_reason", "")

    # Load audience insights to bias the search toward what's working for this channel
    insights = _load_insights()
    insights_prefix = ""
    if insights:
        top    = insights.get("top_areas", [])[:5]
        angles = insights.get("trending_angles", [])[:4]
        summary = insights.get("audience_insights", "")
        analyzed_at = insights.get("analyzed_at", "")[:10]
        angle_lines = "; ".join(angles) if angles else ""
        insights_prefix = (
            f"📊 AUDIENCE CONTEXT (analysis from {analyzed_at}):\n"
            f"For this Telugu YouTube Shorts audience, the highest-interest areas right now are: "
            f"{', '.join(top)}.\n"
            + (f"Trending angles performing well: {angle_lines}.\n" if angle_lines else "")
            + (f"Audience note: {summary}\n" if summary else "")
            + f"When researching the topic below, prioritise angles that fit these high-interest "
            f"areas and trending patterns.\n\n"
        )

    if iteration == 0:
        user_message = (
            f"{insights_prefix}"
            f"Topic: {input_topic}\n\n"
            f"Search for the most shocking, counter-intuitive, or disturbing angles on this topic. "
            f"Look beyond surface-level facts — find the hidden truths, dark implications, paradoxes, "
            f"or recent discoveries (2022–2025) that make this topic genuinely jaw-dropping. "
            f"Each idea must have a consequence or implication the viewer will want to share immediately."
        )
    else:
        user_message = (
            f"{insights_prefix}"
            f"Topic: '{input_topic}'\n"
            f"Previous ideas were rejected: {rejection_reason}\n\n"
            f"Find 3 COMPLETELY DIFFERENT angles — go darker, more niche, more counter-intuitive. "
            f"Try unexplored entry points: technology consequences, "
            f"the human story behind the topic, a historical parallel nobody draws, "
            f"or a recent scientific reversal that changes everything we thought we knew. "
            f"Search for classified/declassified data, peer-reviewed study surprises, "
            f"biological anomalies, or inspiring individuals connected to this topic. "
            f"These ideas must be impossible to find by casually browsing YouTube."
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    print(f"\n[Research Agent] Iteration {iteration + 1}: Searching for ideas about '{input_topic}'...")

    # Agentic loop: let the LLM call tools and reason
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Process tool calls if any
    _MAX_SEARCH_ROUNDS = 5
    _search_rounds = 0
    while response.tool_calls and _search_rounds < _MAX_SEARCH_ROUNDS:
        _search_rounds += 1
        for tool_call in response.tool_calls:
            print(f"[Research Agent] Using tool: {tool_call['name']} → {tool_call['args'].get('query', '')[:60]}")
            tool_result = search_tool.invoke(tool_call["args"])
            from langchain_core.messages import ToolMessage
            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )
        try:
            response = llm_with_tools.invoke(messages)
            messages.append(response)
        except Exception as e:
            print(f"[Research Agent] ⚠ LLM call failed ({type(e).__name__}: {e}) — stopping search early")
            break
    if _search_rounds >= _MAX_SEARCH_ROUNDS:
        print(f"[Research Agent] ⚠ search round cap ({_MAX_SEARCH_ROUNDS}) reached — proceeding with gathered data")

    # Parse the JSON response
    raw = response.content
    if isinstance(raw, list):
        raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    try:
        # Extract JSON block if wrapped in markdown
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        # Find the JSON object
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        ideas = parsed.get("ideas", [])
    except Exception:
        # Fallback: treat the whole response as plain text ideas
        ideas = [{"title": f"Idea {i+1}", "concept": line.strip()}
                 for i, line in enumerate(raw.split("\n")) if line.strip()][:3]

    print(f"[Research Agent] Found {len(ideas)} ideas.")
    for i, idea in enumerate(ideas, 1):
        print(f"  {i}. {idea.get('title', 'Untitled')}")

    return {
        "research_ideas": ideas,
        "iteration": iteration + 1,
        "messages": messages,
    }
