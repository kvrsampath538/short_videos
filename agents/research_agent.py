import json
import os
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tools import get_tavily_search_tool

_INSIGHTS_FILE = Path(__file__).parent.parent / "audience_insights.json"


def _load_insights() -> dict | None:
    if not _INSIGHTS_FILE.exists():
        return None
    try:
        return json.loads(_INSIGHTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None

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

                          STRONG EXAMPLES TO USE AS TEMPLATES:
                          • Ship anchor — people assume the anchor weight holds the ship. WRONG. The
                            CATENARY (the draped curve of chain on the seabed) holds it.
                          • Nuclear reactor water — water is the neutron MODERATOR, not just coolant.
                            If water leaks, the reaction STOPS. The reactor is inherently fail-safe.
                          • GPS and Einstein — GPS satellites' clocks run 38 microseconds fast per
                            day (relativity). Without correction, navigation drifts 11km per day.
                          • PAPI lights — 4 runway lights guide every airliner to land. No passenger
                            knows this system exists.
                          • Seatbelt pre-tensioner — explosive charge detonates BEFORE collision force
                            reaches the occupant.
                          • GravityLight — 12kg bag + geared crank powers LED (cuckoo clock mechanism).
                          • Runway numbers — magnetic heading ÷ 10, painted on every runway worldwide.
                          • Bulbous bow — the underwater bulge on ship hulls creates a wave that cancels
                            the bow wave, reducing drag by 15%.
                          • Airplane window bleed hole — the tiny hole in the inner pane equalises
                            pressure; the outer pane bears all the stress alone.
                          • Car crumple zones — deliberately DESIGNED TO FAIL so the cabin doesn't.
                          • Elevator brake — triggered by overspeeding, NOT by cable snap (centrifugal
                            governor, invented in 1852, same principle as a flywheel governor).
                          • Dead Hand (Perimeter) — Soviet system that auto-launches nuclear weapons
                            if leadership is killed. It is still active.
                          • Trebuchet — the falling counterweight (not a spring) makes it the most
                            energy-efficient siege weapon ever built.
                          • Sonar silence — submarines go silent AFTER pinging because the ping
                            reveals their own position to every listener.
                          • Dam flip bucket — water shot 60m into the air to dissipate kinetic energy
                            safely. Looks like a design flaw; it's intentional.

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

CATEGORY INSPIRATION — rich veins to mine:
AWE (scale and cosmic power):
  NASA Parker Solar Probe (touching the sun at 430 miles/sec) · Artemis mission anomalies ·
  Krakatoa 1883 (heard 5,000 km away, pressure wave circled Earth 7 times) · Voyager 1 interstellar data

MYSTERY (hidden systems, unexplained mechanics):
  Dead Hand nuclear auto-launch system · Fingerprint uniqueness (why every person differs — unknown) ·
  Labyrinth navigation biology · U-2 spy plane chase car · PAPI approach lights geometry

HUMAN STORIES (one person, impossible specific outcome):
  Desmond Doss — saved 75 men at Hacksaw Ridge, refused to ever carry a weapon
  Grigori Perelman — solved the $1M Poincaré Conjecture, declined the prize and vanished
  George Dantzig — solved two "unsolvable" stat problems as homework thinking they were assignments

HIDDEN MECHANISMS (everyday objects and systems nobody understands):
  Ship anchor catenary — chain shape holds the ship, not the anchor weight ·
  PAPI approach lights — 4 lights guide every airliner to land; no passenger knows ·
  Aircraft carrier arresting wire — 3-inch cable stops 30-ton jet from 150mph in 2 seconds ·
  Aircraft carrier catapult / EMALS — 0 to 165mph in 2 seconds on a 100m track ·
  U-2 spy plane chase car — pilot can't see runway; chase car at 140mph shouts altitude ·
  Nuclear reactor water = neutron moderator (water loss = reactor OFF, not meltdown) ·
  GPS needs Einstein's relativity — 38μs/day clock correction; without it, 11km drift daily ·
  GravityLight — 12kg bag descending 20min powers LED via geared crank (cuckoo clock principle) ·
  Runway numbers — magnetic heading ÷ 10 painted on every runway ·
  Seatbelt pre-tensioner — controlled explosive detonates before collision force reaches you ·
  Submarine swim bladder origin — fish invented ballast tank buoyancy 400 million years ago ·
  Dead Hand system — Soviet nuclear auto-launch if leadership is killed ·
  Bridge expansion joints — Brooklyn Bridge grows 4 feet longer in summer ·
  Fire hydrant unpressurised — the fire truck provides the pressure, not the hydrant ·
  Viganella mirror — giant computer-controlled mountain mirror reflects sun into a sunless valley

INDIA GOVERNMENT REPORTS (CAG audits and RTI disclosures):
  2G spectrum CAG report — spectrum sold for ₹9,295 crore; actual value ₹1,76,645 crore ·
  Coal block CAG audit — ₹1.86 lakh crore in undue gains to private companies ·
  CAG found NHAI (highways authority) couldn't account for ₹7,000 crore in toll revenue ·
  RTI revealed government paid 300% premium on military spare parts bought indirectly ·
  Parliamentary committee found 40% of PDS grain never reached intended beneficiaries ·
  SEBI found manipulation in 100+ listed stocks linked to entities connected to officials ·
  CAG audit of Ayushman Bharat found claims paid for patients who were already dead

WORLD CULTURES (shocking practices that are completely normal locally):
  MUST have a SPECIFIC verifiable fact — a number, a law, a frequency, a government response.

  • Toraja, Indonesia — dead family members kept at home for weeks/months, mummified for years;
    exhumed annually in "Ma'nene" ceremony (washed, dressed, photographed, paraded through village)
  • Famadihana, Madagascar — ancestors dug up every 7 years, rewrapped in silk, danced with;
    the number of people attending determines how much honour the family receives
  • Satere-Mawe, Brazil — boys wear gloves packed with bullet ants (30× more painful than a bee)
    for 10 solid minutes. Must do this 20 times across years to be considered a man.
  • Santhara, Jain India — voluntary fasting unto death; legally permitted; Rajasthan High Court
    banned it in 2015, Supreme Court overturned the ban; thousands have died this way
  • Hikikomori, Japan — 1.15 million people (2023 government count) who never leave their rooms;
    Japan's Cabinet Office publishes annual statistics; there is a dedicated government ministry
  • Karoshi, Japan — death from overwork is a legally recognised cause of death; government tracks
    it; companies compensate families; 2,000+ official karoshi deaths recorded annually
  • China funeral strippers — professionally hired to draw large crowds (more people = more honour
    for deceased); government has banned them 4 separate times; practice persists in rural areas
  • Naghol, Vanuatu — men jump from 30m towers with only vines around their ankles; vine must
    brush their head against the earth; preceded bungee jumping by centuries; first recorded
    jump filmed in 1950 was performed by a woman to escape an abusive husband
  • Fa'afafine, Samoa — fully socially accepted third gender (born male, raised as female);
    not a modern concept — centuries-old, mainstream, supported by families and employers
  • Iceland everyone is listed equally — phone directory lists all citizens by first name + job;
    no surname hierarchy; the President is findable the same way as any other citizen

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
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        temperature=0.9,
    )
    search_tool = get_tavily_search_tool(max_results=5)
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
    while response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"[Research Agent] Using tool: {tool_call['name']} → {tool_call['args'].get('query', '')[:60]}")
            tool_result = search_tool.invoke(tool_call["args"])
            from langchain_core.messages import ToolMessage
            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )
        response = llm_with_tools.invoke(messages)
        messages.append(response)

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
