import json
import re
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from tools import get_tavily_search_tool, search_youtube_shorts

AREAS = [
    "history", "science", "latest science news", "geography", "nature", "space",
    "technology", "interesting events", "inspiring people",
    "archaeology news", "aviation reports", "research papers",
    "world cultures", "hidden mechanisms",
]

HISTORY_FILE   = Path(__file__).parent.parent / "ideas_history.json"
BLACKLIST_FILE = Path(__file__).parent.parent / "blacklist.json"
FAVORITES_FILE = Path(__file__).parent.parent / "favorites.json"
BOOKMARKS_FILE = Path(__file__).parent.parent / "bookmarks.json"
INSIGHTS_FILE  = Path(__file__).parent.parent / "audience_insights.json"
TARGET_IDEAS  = 5
MAX_RETRIES   = 4   # each call yields up to 5 candidates; 4 retries = 20 shots at finding 5 unique

_STOPWORDS = {
    # Articles / pronouns / prepositions
    "a","an","the","this","that","its","your","our","their","it","is","are","was","were",
    "be","been","has","have","had","just","for","with","from","into","onto","than","more",
    "most","all","every","some","any","can","will","may","might","could","would","should",
    "did","does","and","but","or","nor","so","yet","how","why","what","when","where","who",
    "which","we","they","he","she","you","me","him","her","us","them","my","his","of","to",
    "in","on","at","by","up","out","off","over","under","about","after","before","never",
    "ever","not","no","even","one","two","three","first","last","new","old","own","same",
    "other","each","both","few","many","much","such","only","also","there","here","now",
    "then","too","very","real","still","back","find","give","keep","left","let","look",
    "move","need","part","show","turn","work","way","time","year","long","days","next",
    "high","large","small","used","those","these","been","being","right","since","until",
    "around","another","billion","faster","wrong","already","actually","suddenly","almost",
    # Generic verbs that appear in many topics (not topic-specific)
    "get","got","make","made","goes","going","went","come","came","take","took","know",
    "knew","found","find","killed","born","died","die","visited","visit","lived","live",
    "exist","existed","discovered","discover","revealed","reveal","created","create",
    "started","start","stopped","stop","changed","change","happened","happen","caused",
    "cause","built","build","made","used","proved","proved","showed","showed","began",
    "began","became","become",
    # Generic nouns / adjectives that appear everywhere (not topic-specific)
    "people","human","thing","way","world","life","place","time","fact","reason","secret",
    "nobody","everyone","someone","everything","nothing","something","anything","anywhere",
    "inside","outside","through","beyond","across","around","between","behind","beneath",
    "east","west","north","south","expanding","dying","faster","closer","bigger","smaller",
    "longer","older","younger","higher","lower","deeper","wider","stranger","stronger",
    "friend","victim","wear","force","power","energy","level","size","shape","color",
    "speed","heat","light","sound","weight","mass","age","distance","temperature",
    # Generic research/academic language — appears in almost every concept regardless of topic
    "scientist","researcher","study","research","experiment","result","evidence","data",
    "finding","suggest","confirm","confirm","publish","journal","paper","team","test",
    "model","analysis","observation","measure","method","process","theory","conclude",
    # Generic filler words common in concept descriptions
    "entire","single","mean","meaning","call","contain","near","like","known","given",
    "million","across","within","between","using","making","doing","going","taking",
    "allow","help","lead","even","just","well","also","then","now","here","there",
    # Template artifacts — "gut-punch implication" phrase used in research_agent prompt
    "gut","punch","implication",
}


# ── History helpers ────────────────────────────────────────────────────────────

def _load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_to_history(ideas: list[dict]) -> None:
    history = _load_history()
    timestamp = datetime.now(timezone.utc).isoformat()
    for idea in ideas:
        history.append({**idea, "generated_at": timestamp})
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    _compact_history()


def _compact_history() -> int:
    """
    Remove duplicate/near-duplicate and blacklisted ideas from ideas_history.json.
    Iterates oldest-to-newest: when a later idea is too similar to an already-kept idea
    the later one is removed (oldest version is preserved).
    Returns the number of entries removed.
    """
    history = _load_history()
    if len(history) < 2:
        return 0

    blacklisted_titles = {
        b.get("title", "").lower().strip()
        for b in _load_blacklist()
    }

    kept: list[dict] = []
    removed = 0

    for idea in history:
        title = idea.get("title", "").lower().strip()

        if title in blacklisted_titles:
            print(f"[History Compact] ✗ BLACKLISTED '{idea.get('title', '')}'")
            removed += 1
            continue

        too_sim, matched = _is_too_similar(idea, kept, set())
        if too_sim:
            print(f"[History Compact] ✗ DUPLICATE  '{idea.get('title', '')}' ~ '{matched}'")
            removed += 1
        else:
            kept.append(idea)

    if removed > 0:
        HISTORY_FILE.write_text(
            json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[History Compact] {len(history)} → {len(kept)} entries (removed {removed})")

    return removed


def load_audience_insights() -> dict | None:
    """Load saved audience analysis insights, or None if not yet run."""
    if not INSIGHTS_FILE.exists():
        return None
    try:
        return json.loads(INSIGHTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_blacklist() -> list[dict]:
    if not BLACKLIST_FILE.exists():
        return []
    try:
        return json.loads(BLACKLIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_to_blacklist(idea: dict) -> None:
    blacklist = _load_blacklist()
    timestamp = datetime.now(timezone.utc).isoformat()
    blacklist.append({**idea, "blacklisted_at": timestamp})
    BLACKLIST_FILE.write_text(json.dumps(blacklist, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_favorites() -> list[dict]:
    if not FAVORITES_FILE.exists():
        return []
    try:
        return json.loads(FAVORITES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_to_favorites(idea: dict) -> None:
    favorites = _load_favorites()
    # Avoid exact duplicates
    existing_titles = {f.get("title", "") for f in favorites}
    if idea.get("title", "") in existing_titles:
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    favorites.append({**idea, "favorited_at": timestamp})
    FAVORITES_FILE.write_text(json.dumps(favorites, indent=2, ensure_ascii=False), encoding="utf-8")


def load_bookmarks() -> list[dict]:
    if not BOOKMARKS_FILE.exists():
        return []
    try:
        return json.loads(BOOKMARKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_to_bookmarks(idea: dict, source: str = "") -> bool:
    """Save idea to bookmarks. Returns True if saved, False if already bookmarked."""
    bookmarks = load_bookmarks()
    existing = {b.get("title", "") for b in bookmarks}
    if idea.get("title", "") in existing:
        return False
    timestamp = datetime.now(timezone.utc).isoformat()
    bookmarks.append({**idea, "bookmarked_at": timestamp, "bookmark_source": source})
    BOOKMARKS_FILE.write_text(json.dumps(bookmarks, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def remove_from_bookmarks(title: str) -> None:
    bookmarks = load_bookmarks()
    bookmarks = [b for b in bookmarks if b.get("title", "") != title]
    BOOKMARKS_FILE.write_text(json.dumps(bookmarks, indent=2, ensure_ascii=False), encoding="utf-8")


def update_history_saturation(all_scored: list[dict]) -> None:
    """Update history entries with saturation scores after bulk_saturation_filter runs."""
    if not all_scored:
        return
    history = _load_history()
    score_by_title = {i.get("title", ""): i for i in all_scored}
    updated = False
    for entry in history:
        title = entry.get("title", "")
        if title in score_by_title and "saturation_score" not in entry:
            src = score_by_title[title]
            entry["saturation_score"] = src.get("saturation_score")
            entry["saturation_reason"] = src.get("saturation_reason", "")
            updated = True
    if updated:
        HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Similarity / frequency helpers ────────────────────────────────────────────

def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    result = set()
    for w in words:
        if len(w) < 3 or w in _STOPWORDS:
            continue
        # Normalize plurals: "ants"→"ant", "moons"→"moon", "holes"→"hole"
        stem = w[:-1] if (w.endswith("s") and len(w) >= 4 and len(w[:-1]) >= 3) else w
        # Also filter the stemmed form — "years"→"year" must be caught even though
        # the raw word "years" is not in _STOPWORDS (only "year" is)
        if stem in _STOPWORDS:
            continue
        result.add(stem)
    return result


def _idea_text(idea) -> str:
    """Full text of an idea for keyword extraction (title + concept)."""
    if isinstance(idea, dict):
        return f"{idea.get('title', '')} {idea.get('concept', '')}"
    return str(idea)


def _idea_title(idea) -> str:
    """Display title of an idea."""
    if isinstance(idea, dict):
        return idea.get("title", str(idea))
    return str(idea)


def _pick_target_areas(used_ideas: list[dict], n: int = 5) -> list[str]:
    """
    Pick the n most deserving areas, balancing coverage count against audience interest.

    Sort key: coverage_count * 10 - audience_score
      - Primarily picks least-covered areas (low count = low sort key = picked first)
      - Among similarly-covered areas, audience score breaks the tie (higher score = picked first)
      - A score of 10 effectively reduces one full coverage unit of weight, so a highly
        interesting area that was covered once competes with a less interesting uncovered area
    """
    counts: Counter = Counter(
        idea.get("area", "").lower().strip()
        for idea in used_ideas
        if idea.get("area")
    )
    insights   = load_audience_insights()
    scores     = insights.get("area_scores", {}) if insights else {}

    def _sort_key(area: str) -> float:
        coverage       = counts.get(area, 0)
        audience_score = float(scores.get(area, 5))   # default 5/10 if no insights
        # hidden mechanisms appears in nearly every batch — subtract 40 to force priority
        # (needs 5 uses before it stops being picked ahead of fresh areas)
        priority_boost = -40 if area == "hidden mechanisms" else 0
        return coverage * 10 - audience_score + priority_boost

    return sorted(AREAS, key=_sort_key)[:n]


def _overused_words(ideas: list, min_count: int = 3) -> set[str]:
    """Topic-specific words that appear in min_count+ idea TITLES (titles only, not concepts,
    so generic concept words like 'study', 'researcher', 'published' stay off the banned list)."""
    counts: Counter = Counter()
    for idea in ideas:
        for w in _keywords(_idea_title(idea)):
            counts[w] += 1
    return {w for w, c in counts.items() if c >= min_count}


def _is_too_similar(new_idea, used_ideas: list, high_freq: set[str]) -> tuple[bool, str]:
    """
    Two-level duplicate check:
    1. Title-level: 1 high-freq topic word shared with a specific used title → reject.
                    OR 3+ title keywords shared → reject.
    2. Concept-level: only triggered when titles already share ≥1 keyword (same domain).
                      Requires 10+ full-text content words shared → reject.
                      Threshold is high because genuine duplicates (same study, same event)
                      share 15-26 specific words, while false positives from different topics
                      that share research language only reach 5-8 even before our expanded
                      stopwords strip out words like 'scientist', 'million', 'entire', etc.
    """
    new_title_kw = _keywords(_idea_title(new_idea))
    new_full_kw  = _keywords(_idea_text(new_idea))
    if not new_full_kw:
        return False, ""

    for used in used_ideas:
        display       = _idea_title(used)
        used_title_kw = _keywords(_idea_title(used))
        used_full_kw  = _keywords(_idea_text(used))

        # ── Level 1: title keywords ──────────────────────────────────────────
        title_overlap = new_title_kw & used_title_kw
        if title_overlap & high_freq:
            return True, display
        if len(title_overlap) >= 3:
            return True, display

        # ── Level 2: concept keywords (same study / same event) ─────────────
        # Gate: only compare concepts when titles already share ≥1 topic word.
        # This prevents cross-domain false positives like "Dying Ants" vs "Snack Dye"
        # from being compared at concept level (they share zero title keywords).
        if not title_overlap:
            continue
        full_overlap = new_full_kw & used_full_kw
        if len(full_overlap) >= 10:
            return True, display

    return False, ""


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a viral YouTube Shorts idea machine. Your ideas combine REAL FACTS with
IRONY, ABSURDITY, or DARK HUMOR to create content that stops the scroll.

━━━ WHAT MAKES AN IDEA GO VIRAL ━━━

Study these PROVEN viral structures:

1. THE PUNCHLINE REVERSAL — setup implies one outcome, reality delivers the opposite:
   "Napoleon's Greatest Defeat Was Rabbits" (not a military rival — rabbits)
   "Australia Declared War On Birds — And Lost" (the army lost to emus)
   "The CIA Spent $20M Spying With A Cat" (literal surveillance cat, got hit by a taxi)

2. THE ABSURD UNDERSTATEMENT — catastrophic event described with deadpan specificity:
   "He Survived Both Atomic Bombs — And Told His Boss" (Tsutomu Yamaguchi, Monday morning)
   "One Man's Gut Feeling Stopped World War III" (Petrov, 1983, single Soviet officer)
   "A Pope Dug Up A Corpse — To Put It On Trial" (Cadaver Synod, 897 AD)

3. THE BELIEF FLIP — something everyone 'knows' is provably wrong:
   "Everything You Know About Carthage Is Roman Propaganda"
   "Fish Taught Submarines How To Dive" (not ballast tanks — fish swim bladders did it first)

4. THE HIDDEN CATASTROPHE — a trusted system secretly caused harm:
   "Your Boss's AI Is Making You Dumber"
   "Our Rivers Are Accidentally Drugging Wild Salmon"
   "The Snack Dye That Makes Your Skin Disappear"

5. THE IMPOSSIBLE REAL STORY — specific, verifiable, sounds made up:
   "She Fell 10,000 Feet — And Walked Out Of The Jungle"
   "D-Day's Secrets Leaked — In A Crossword Puzzle"
   "He Solved Two 'Impossible' Problems — As Homework" (George Dantzig mistook them for assignments)

6. THE HIDDEN SYSTEM — an invisible mechanism running silently in plain sight:
   "Pilots Land By 4 Coloured Lights Nobody Ever Explains" (PAPI lights)
   "A Race Car Chases Every U-2 Spy Plane Landing" (pilot can't see runway — chase car talks them down)
   "Finland Charges Speeding Fines Based On Your Salary" (Nokia exec paid $103,000 for 15mph over)
   "A Mirror Saved A Town That Gets No Sunlight For Months" (Viganella, Italy)
   "The USSR Built A Machine To Start Nuclear War Without Anyone Pressing A Button" (Dead Hand)

━━━ CURIOSITY FILTERS — every idea MUST pass at least one ━━━

✅ SURPRISE — the outcome is the opposite of what everyone expects:
   ❌ "How airplanes fly"
   ✅ "Why a race car follows this aircraft during landing" (U-2 pilot can't see the runway)

✅ COUNTERINTUITIVE — the mechanism defies what we were taught:
   ❌ "Submarines use ballast tanks"
   ✅ "Fish taught submarines how to dive" (engineers copied the swim bladder)

✅ HIDDEN SYSTEM — an invisible mechanism running silently in the background:
   People love discovering the hidden machinery of the world. Look for:
   PAPI lights · Finland day fines · Viganella mirror · Dead Hand system ·
   Aircraft carrier arresting wire · U-2 chase car · CERN safety interlocks

━━━ WHERE TO SEARCH — specific authoritative sources, NOT generic web ━━━

Scientific: NASA.gov, Nature.com, ScienceDaily.com, ArXiv.org, PubMed
Official records: Declassified CIA/NSA archives, NTSB aviation incident reports,
                  IMO maritime databases, government experiment records, military tribunal files
Domain journals: Archaeology journals (JSTOR), NEJM / Lancet medical case studies,
                 Aviation Safety Network, Cold War declassified document collections
Patents & filings: Google Patents, USPTO.gov, EPO (European Patent Office)
Government reports (Global): GAO reports, Congressional Research Service, WHO reports, IPCC findings,
                             FDA adverse event database, CDC unusual outbreak reports
Government reports (India): CAG (Comptroller & Auditor General) audit reports (site:cag.gov.in),
                             RTI disclosures, Parliamentary Standing Committee reports,
                             SEBI orders, RBI annual reports, NITI Aayog policy papers,
                             Ministry audit findings, PIB (Press Information Bureau)
Research papers: ArXiv preprints, PubMed, SSRN (social science), bioRxiv
World cultures: Anthropological journals, BBC Travel/Culture, Atlas Obscura, Vice World News,
                National Geographic, academic ethnography databases

GOOD search queries targeting these sources:
  "site:nasa.gov [topic] unexpected discovery"
  "NTSB aviation incident report [topic] unexpected cause"
  "site:arxiv.org [topic] counterintuitive 2024"
  "CIA declassified [topic] operation backfired"
  "site:sciencedaily.com [topic] surprising result 2023 2024"
  "NEJM medical case impossible recovery [topic]"
  "site:patents.google.com [topic] bizarre invention"
  "GAO report [topic] unexpected finding"
  "site:cag.gov.in CAG audit India [topic] irregularity"
  "RTI disclosure India [topic] government failure"
  "archaeology discovery overturned [topic] site:jstor.org"
  "shocking cultural practice [country] unknown outside"
  "[topic] hidden system explained"

BAD search queries: "interesting [topic] facts", "shocking [topic] discoveries", "mind blowing [topic]"

━━━ AREA GUIDANCE — what each area means ━━━

archaeology news     → Recent digs that overturn history textbooks. Not "ancient Egyptians built X" —
                       find the DIG that proved experts wrong, the artefact in the wrong place, the
                       civilisation nobody knew existed. Search: "archaeology discovery overturned 2023 2024"
aviation reports     → NTSB/AAIB incident reports with absurd or counterintuitive causes. The bird
                       strike that grounded a fleet, the checklist step that saves 400 lives every
                       year, the pilot who landed blind. Search: "NTSB report unexpected cause site:ntsb.gov"
research papers      → Peer-reviewed results that shocked the researchers themselves — the study that
                       found the opposite of its hypothesis, the drug trial stopped early because it
                       worked too well, the meta-analysis that invalidated 20 years of advice.
                       Search: "site:arxiv.org [topic] unexpected result 2024", "study reversed findings"
hidden mechanisms    → Invisible systems and engineering principles hiding in plain sight that almost
                       nobody understands — the physical laws, clever hacks, and counterintuitive
                       designs behind everyday objects and systems. The STORY is the gap between what
                       people ASSUME the mechanism does and what it ACTUALLY does.

                       THE PATTERN TO FOLLOW:
                       Most people think a ship anchor works because it's heavy and digs into the seabed.
                       WRONG. What actually holds a ship is the CATENARY — the U-shaped curve of the
                       chain draping along the seabed, acting as a shock absorber and holding the chain
                       nearly horizontal. A lighter chain with a better catenary holds better than a
                       heavier anchor dragged upright.
                       Title: "Ship Anchors Don't Actually Anchor Ships — The Chain Does"

                       MORE EXAMPLES OF THIS PATTERN:
                       • PAPI lights (Precision Approach Path Indicator) — 4 lights on every runway.
                         2 red + 2 white = perfect glide path. All red = dangerously low. All white =
                         too high. Every civilian airliner uses this to land, and almost no passenger
                         knows it exists. Title: "The 4 Lights That Every Plane Lands By"
                       • Aircraft carrier arresting wire — a 3-inch steel cable strung across a 300m
                         deck stops a 30-ton jet traveling at 150mph in under 2 seconds. The cable
                         runs into hydraulic cylinders that absorb the force gradually. The pilot has
                         to apply FULL THROTTLE on touchdown in case the wire snaps.
                         Title: "Carrier Pilots Slam Throttle OPEN When They Land — Here's Why"
                       • Aircraft carrier catapult — jets launch from 0 to 165mph in 2 seconds. Early
                         carriers used steam pistons; modern carriers use EMALS (electromagnetic). The
                         pilot experiences 4G in 2 seconds. The launch track is only 100m long.
                         Title: "How A Navy Jet Goes From 0 To 165mph In 2 Seconds"
                       • U-2 spy plane landing — the plane has no rear visibility from the cockpit and
                         stalls at nearly the same speed it cruises. Landing requires a chase car to
                         drive alongside at 140mph shouting altitude readings over radio.
                         Title: "The Race Car That Chases Every U-2 Spy Plane Landing"
                       • Nuclear reactor water — water in most reactors is NOT primarily a coolant.
                         It's a NEUTRON MODERATOR — it slows neutrons enough to sustain fission. If
                         the water leaks out, the reaction STOPS. The reactor is inherently fail-safe
                         by design because the coolant is also the ON switch.
                         Title: "Nuclear Reactors Turn Off If They Lose Water — Not Melt Down"
                       • GPS and Einstein's relativity — GPS satellites' clocks tick faster by
                         38 microseconds/day because of lower gravity (general relativity) and slower
                         orbital speed (special relativity). If uncorrected, GPS would drift 11km/day.
                         Einstein's equations are what make your phone's navigation accurate.
                         Title: "Your Phone's GPS Only Works Because Einstein Was Right"
                       • GravityLight — a lamp that runs on no electricity, no batteries, no solar.
                         A 12kg bag of sand, slowly lowered over 20 minutes, drives a gear system
                         that powers an LED light — exactly like a cuckoo clock mechanism.
                         Title: "This Lamp Is Powered By Gravity — Like A Cuckoo Clock"
                       • Runway numbers — every runway has a two-digit number painted at each end.
                         It's the magnetic compass heading divided by 10. Runway 27 = 270° = due west.
                         The same runway from the other end is Runway 09. Navigation baked into paint.
                         Title: "The Hidden Navigation Code Painted On Every Runway"
                       • Submarine buoyancy — submarines don't sink by flooding tanks with water.
                         They vent AIR out of ballast tanks and let water in passively. To surface,
                         they blow compressed air back in. Fish invented this first — the swim bladder
                         is an ancient biological version of the same system.
                         Title: "Fish Invented Submarine Technology 400 Million Years Before We Did"
                       • Seatbelt pre-tensioner — modern car seatbelts contain a small explosive charge.
                         In a crash, it detonates in milliseconds — before the collision force even
                         reaches you — pulling the belt tight. Airbags work the same way: sodium azide
                         explodes to fill the bag in 30ms, faster than a human blink.
                         Title: "Your Seatbelt Has An Explosive Inside It"
                       • Viganella mirror — the Italian village of Viganella sits in a valley so deep
                         it gets NO direct sunlight for 83 days each winter. In 2006, engineers
                         installed a giant computer-controlled mirror on the mountain above that
                         tracks the sun and reflects it onto the village square all day.
                         Title: "This Village Built A Giant Mirror To Steal Sunlight From A Mountain"

                       WHAT TO SEARCH FOR:
                       "hidden system how [object] actually works counterintuitive mechanism"
                       "engineering design principle behind [system] explained"
                       "how [everyday object] works is different from what people think"
                       "physics behind [vehicle/structure] explained counterintuitive"
                       "simple mechanism nobody understands [device] how it actually works"
                       "clever engineering solution hidden in plain sight"
                       "site:nasa.gov engineering solution unexpected mechanism"
                       "aviation system hidden mechanism passenger doesn't know"

                       BAD (DO NOT generate these):
                       ✗ "How engines work" — too broad, no specific hidden element
                       ✗ "The physics of flight explained" — educational lecture, not a story
                       ✗ "Top 5 engineering facts" — list format, no single irony
                       GOOD: pick ONE specific mechanism with a gap between the assumption and reality

world cultures       → Shocking, counterintuitive, or completely unknown cultural practices from
                       countries around the world — things that sound made-up but are real and normal
                       in their local context. NOT "this country has a unique festival" — find the
                       practice that makes an outsider's jaw drop: the legal ritual, the social norm,
                       the government policy, the belief system nobody outside knows exists.

                       PATTERN TO FOLLOW:
                       "In Indonesia, families keep deceased relatives at home for WEEKS before burial
                       — some mummify them and keep them for YEARS, bringing out the corpse yearly
                       to dress and parade it through the village."
                       Title: "In This Country, You Live With Your Dead Family For Years"

                       MORE EXAMPLES (use as templates, not topics):
                       • Toraja death ritual (Indonesia) — dead kept at home, mummified, exhumed
                         yearly for "Ma'nene" ceremony, dressed in new clothes and walked around
                       • Famadihana (Madagascar) — ancestors dug up every 7 years, rewrapped in silk,
                         danced with, then reburied. A celebration, not mourning.
                       • Satere-Mawe bullet ant ritual (Brazil) — boys wear gloves full of bullet ants
                         for 10 minutes. 20 times. This is how you become a man.
                       • Santhara (Jain, India) — voluntary fasting unto death, legally permitted and
                         religiously sanctioned. Supreme Court tried to ban it, community refused.
                       • Hikikomori (Japan) — 1.15 million people never leave their rooms. Japan has
                         a government-funded task force dedicated solely to this problem.
                       • Karoshi (Japan) — death from overwork is a legally recognised cause of death
                         with official government statistics. Companies pay compensation.
                       • Fa'afafine (Samoa) — a fully accepted third gender in mainstream Samoan
                         society, not a modern construct — traditional and centuries old.
                       • China funeral strippers — hired to attract larger crowds to rural funerals
                         (larger crowd = more honour to the deceased). Government tried to ban it.
                       • Denmark age 30 pepper ritual — if still unmarried at 30, friends cover you
                         in pepper publicly. At 25, cinnamon. Both are ancient traditions.
                       • Iceland phone book — lists every citizen by first name with profession.
                         No surname hierarchy. The president is listed the same as everyone else.
                       • Tonga beauty standard — obesity is culturally prized as beauty and wealth.
                         The royal family actively exemplified this for centuries.
                       • Naghol (Vanuatu) — men jump from 30m towers with only vines tied to their
                         ankles. The vine MUST touch their head to the ground. This preceded
                         bungee jumping by centuries. The first recorded jump was 1950 by a woman.

                       SEARCH: "cultural practice [country] shocking unknown outside"
                               "traditional ritual [country] that outsiders find unbelievable"
                               "strange law [country] cultural reason"
                               "unusual social norm [country] anthropology"

━━━ CATEGORY INSPIRATION — rich veins to mine ━━━

AWE (scale and cosmic power):
  NASA Parker Solar Probe touching the sun at 430 miles/sec · Artemis mission anomalies ·
  Krakatoa 1883 (heard 5,000 km away, pressure wave circled Earth 7 times) ·
  Voyager 1 data from interstellar space

MYSTERY (hidden systems, unexplained mechanics):
  Dead Hand nuclear auto-launch system · Fingerprint uniqueness (no one knows why they differ) ·
  Labyrinth navigation biology · U-2 spy plane chase car procedure · PAPI approach lights

PSYCHOLOGY EXPERIMENTS (specific story + shocking result + implication about YOU):
  Richter rat experiment — rescued rats swam 240x longer (hope is physically measurable)
  Milgram experiment — 65% would fatally shock a stranger if authority told them to
  Stanford Prison — ordinary students became sadistic guards in 6 days
  Rosenhan — sane people checked into asylums, nobody detected them (except other patients)
  Good Samaritan — seminary students rushing to preach about helping ignored a dying man
  Harlow monkey — baby monkeys chose comfort over food, proving love matters more than feeding
  Asch conformity — 75% of people denied what their own eyes saw to agree with a group
  Bystander effect (Kitty Genovese) — 38 witnesses, nobody called police
  The marshmallow test reversal — kids who waited weren't more successful; family stability was
  Broken windows theory — disproven by a 2019 study that controlled for actual policing levels

HUMAN STORIES (one person, impossible outcome):
  Desmond Doss — saved 75 men at Hacksaw Ridge, refused to carry a weapon
  Grigori Perelman — solved a $1M Millennium Problem, declined the prize and disappeared
  George Dantzig — solved two "unsolvable" problems as homework, thought they were assignments

HIDDEN MECHANISMS (everyday objects and systems nobody understands) — PRIORITY AREA:
  Ship anchor catenary (the CHAIN holds the ship, not the anchor weight) ·
  PAPI approach lights (4 lights tell every pilot their glide angle; no passenger knows this) ·
  Aircraft carrier arresting wire (3-inch cable stops 30-ton jet from 150mph in 2 seconds) ·
  Aircraft carrier catapult / EMALS (0 to 165mph in 2 seconds on a 100m track) ·
  U-2 spy plane chase car (pilot can't see runway — chase car at 140mph shouts altitude) ·
  Nuclear reactor water = neutron moderator NOT coolant (water loss = reactor OFF, not meltdown) ·
  GPS Einstein correction (satellites' clocks drift 38μs/day — uncorrected, navigation fails by 11km) ·
  GravityLight (12kg bag descending for 20min powers LED — cuckoo clock mechanism) ·
  Runway numbers (magnetic heading ÷ 10, baked into the paint) ·
  Submarine ballast tanks (fish invented this first — the swim bladder) ·
  Seatbelt pre-tensioner explosive charge (detonates BEFORE the collision reaches you) ·
  Viganella mirror (computer-controlled mirror reflects sunlight into a sunless valley) ·
  Dead Hand nuclear auto-launch system (built to fire if Soviet leadership was killed) ·
  ILS / Instrument Landing System (invisible radio beams guide planes in zero-visibility fog) ·
  Bridge expansion joints (Brooklyn Bridge grows 1.2m longer in summer — built-in thermal gap) ·
  Fire hydrant not pressurised (truck provides the pump; the hydrant itself is passive and dry) ·
  Freeway on-ramp metering lights (timed to prevent downstream shockwave congestion, not merge) ·
  Trebuchet counterweight (a falling weight, not a spring — the most energy-efficient war machine ever built) ·
  Ship hull bulbous bow (the underwater bulge creates a wave that cancels the bow wave — saves 15% fuel) ·
  Autopilot disconnect (most commercial pilots' hands touch the controls for under 7 min per flight) ·
  Traffic light green arrow vs full green (different right-of-way rules almost nobody knows) ·
  Aeroplane window inner pane has a bleed hole (pressure equalisation — the outer pane bears all stress) ·
  Bascule bridge counterweight (Tower Bridge lifts 1,000-ton spans with 400-ton counterweights in 60 seconds) ·
  Elevator emergency brake (not triggered by cable snap — triggered by overspeeding, like a centrifugal governor) ·
  Concrete creep (the Sydney Harbour Bridge has moved 18cm since 1932 — steel expands, concrete flows) ·
  Sonar ping silence (submarines go silent AFTER pinging because the ping reveals their own position) ·
  Car crumple zones (designed to FAIL on purpose — energy absorbed by deformation, not by strength) ·
  Dam spillway flip bucket (water shot 60m into the air to dissipate energy — not a design flaw)

ARCHAEOLOGY (digs that broke history):
  The Bronze Age collapse — entire civilisations vanished in 50 years, cause still unknown ·
  Göbekli Tepe — built 6,000 years before Stonehenge, by "hunter-gatherers" who weren't supposed to
  be capable of it · Antikythera mechanism — 2,000-year-old analogue computer found in a shipwreck

AVIATION INCIDENTS (absurd causes, impossible outcomes):
  United 232 — hydraulic failure, crew invented a technique in the air to land a plane with no controls ·
  Gimli Glider — Boeing 767 ran out of fuel mid-flight because ground crew used wrong units ·
  Air Canada 143 — same story, different flight, both landed safely

WORLD CULTURES (shocking/unknown practices that are completely normal locally):
  Toraja, Indonesia — families keep deceased relatives at home for weeks/months, exhume them
  yearly for Ma'nene (wash, dress, parade the mummified body through the village) ·
  Famadihana, Madagascar — dig up ancestors every 7 years, rewrap in silk, dance with the corpse ·
  Satere-Mawe, Brazil — bullet ant initiation gloves (10 minutes, 20 times, to become a man) ·
  Santhara, India — legally permitted Jain practice of fasting unto death; Supreme Court tried
  to ban it in 2015, the community won ·
  Hikikomori, Japan — 1.15 million people who never leave their rooms; Japan has a government
  ministry dedicated solely to this crisis ·
  Karoshi, Japan — death from overwork is a legally recognised cause of death with official
  government compensation statistics ·
  China funeral strippers — hired to ensure large crowds (= honour to deceased); government
  banned them in 2006, 2015, 2018, 2023 — they keep coming back ·
  Naghol land diving, Vanuatu — men leap from 30m towers with only vines tied to ankles,
  head must brush the ground. Preceded bungee jumping by centuries.

RESEARCH PAPERS (results that broke their own field):
  The study that found placebos work even when patients know they're placebos ·
  The paper that proved a common painkiller reduces empathy

━━━ BAD PATTERNS TO AVOID ━━━
✗ "Ancient X civilisation did Y" — dry educational trivia, no emotional hook
✗ "Study shows X might cause Y" — weak hedging language, not a story
✗ "The shocking truth about X" — vague, no specific punchline
✗ "X from history nobody talks about" — generic framing, no irony
✗ Any idea where the title is a straight fact rather than a punchline or reversal

━━━ YOUR PROCESS ━━━

PHASE 1 — Study ONE YouTube search for FORMAT only:
  Use search_youtube_shorts once to see what title formats and emotional hooks work.
  Extract the STRUCTURE (e.g. "He did X — The result was Y"), not the topics.
  Topics you find are already covered — do not use them.

PHASE 2 — Hunt with SPECIFIC queries targeting the authoritative sources above:
  Search for STORIES and SYSTEMS, not lists of facts. Target irony, absurdity, hidden mechanisms.
  Use the good query templates above. Vary your source domains each search.

━━━ MANDATORY OUTPUT RULES ━━━
1. Generate ideas ONLY for the areas specified in the "TARGET AREAS" block below — one per area.
2. Each idea must have a PUNCHLINE TITLE — the reversal or absurdity must be IN the title.
3. Every idea must pass the CURIOSITY FILTER (surprise, counterintuitive, OR hidden system).
4. For "history" or "interesting events" area: events from 1900 onwards ONLY.
5. Real, verifiable facts only — no speculation or "might be true".

Return exactly 5 ideas as JSON:
{
  "ideas": [
    {
      "title": "Punchline title — the irony/reversal must be visible in the title itself",
      "area": "exact area from the TARGET AREAS list",
      "concept": "2-3 sentences: the specific verifiable fact, the irony/mechanism, the implication viewers share",
      "viral_hook": "ONE opening sentence so absurd/specific it stops the scroll",
      "pattern_used": "which of the 6 viral structures above this follows"
    }
  ]
}"""


# ── Per-batch search angle rotation ───────────────────────────────────────────
# Each retry batch is steered toward a different domain so the LLM doesn't
# repeat the same generic queries (e.g. "viral history shorts") across retries.
_BATCH_ANGLES = [
    (
        "Authoritative source mining — NASA, NTSB, declassified archives",
        "Search SPECIFIC authoritative databases, not generic web. "
        "Queries: 'site:nasa.gov unexpected discovery mission result counterintuitive', "
        "'NTSB aviation incident report unexpected cause absurd outcome site:ntsb.gov', "
        "'CIA declassified operation backfired site:cia.gov', "
        "'site:sciencedaily.com surprising counterintuitive result 2024', "
        "'government experiment citizens declassified archive'",
    ),
    (
        "Hidden mechanisms — everyday objects and systems nobody understands",
        "Target the gap between what people ASSUME a mechanism does and what it ACTUALLY does. "
        "The title must contain the counterintuitive reversal. "
        "Queries: 'PAPI lights aircraft landing glide path how works', "
        "'ship anchor catenary chain physics how ship actually stays put', "
        "'aircraft carrier arresting wire hydraulic physics how it stops jet', "
        "'aircraft carrier catapult EMALS launch 0 to 165mph how works', "
        "'U-2 spy plane chase car landing procedure why pilot blind', "
        "'nuclear reactor water neutron moderator not coolant counterintuitive', "
        "'GPS satellite clock drift Einstein relativity correction navigation', "
        "'seatbelt pre-tensioner explosive charge how works', "
        "'submarine ballast tank fish swim bladder buoyancy', "
        "'GravityLight gravity powered lamp how works cuckoo clock', "
        "'runway numbers magnetic heading explained', "
        "'Dead Hand nuclear automatic launch system Soviet explained', "
        "'bulbous bow ship hull underwater bulge reduces drag fuel savings', "
        "'airplane window bleed hole inner pane purpose pressure equalisation', "
        "'elevator centrifugal governor brake overspeed mechanism', "
        "'trebuchet counterweight falling mass war machine energy efficiency', "
        "'car crumple zone designed to fail energy absorption physics', "
        "'bridge expansion joint thermal movement how bridges handle heat', "
        "'fire hydrant not pressurised dry barrel truck provides pump', "
        "'dam spillway flip bucket aerator how hydraulic jump works', "
        "'autopilot commercial aviation pilot hands off controls hours per flight', "
        "'sonar submarine silence after ping reveals own position', "
        "'hidden engineering system public never notices how works counterintuitive'",
    ),
    (
        "Impossible human stories — Dantzig, Doss, Perelman pattern",
        "One person, one specific verifiable jaw-dropping outcome. "
        "Queries: 'George Dantzig solved impossible problems homework unsolvable', "
        "'Desmond Doss Hacksaw Ridge saved 75 men unarmed', "
        "'Grigori Perelman declined million dollar prize Poincare', "
        "'person turned down prize award impossible outcome real verified', "
        "'survival defied medical odds specific percentage real'",
    ),
    (
        "Scientific journals & medical case studies — counterintuitive peer-reviewed findings",
        "Mine peer-reviewed sources for results that shocked the researchers themselves. "
        "Queries: 'site:arxiv.org counterintuitive unexpected result 2024', "
        "'NEJM Lancet medical case impossible recovery outcome', "
        "'site:nature.com finding contradicts established theory 2023 2024', "
        "'PubMed study result opposite expected real', "
        "'archaeology journal discovery overturned consensus history'",
    ),
    (
        "Dark irony & institutional betrayal — official incident records",
        "Systems and institutions that did the opposite of their purpose — use official records. "
        "Queries: 'medical treatment caused the disease it was designed to prevent real', "
        "'maritime database IMO ship disaster unexpected cause official report', "
        "'intelligence agency operation spectacular backfire declassified', "
        "'corporate safety system created the hazard it was built to prevent', "
        "'Krakatoa eruption records 1883 pressure wave circled earth times'",
    ),
]


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(llm_with_tools, web_search,
              banned_words: set[str], recent_titles: list[str],
              targeted_rejects: list[dict], iteration: int,
              favorites: list[dict] | None = None,
              used_queries: list[str] | None = None,
              batch_num: int = 0,
              target_areas: list[str] | None = None,
              audience_insights: dict | None = None) -> tuple[list[dict], list[str]]:

    banned_block = ""
    if banned_words:
        words_str = ", ".join(sorted(banned_words))
        banned_block = (
            f"🚫 BANNED TOPIC WORDS — these subjects are exhausted, do NOT use any of them "
            f"as the primary subject of an idea:\n{words_str}\n\n"
            f"If a word from this list is the main noun/subject of your idea → reject that idea "
            f"and choose a completely different subject.\n\n"
        )

    recent_block = ""
    if recent_titles:
        bullets = "\n".join(f"  • {t}" for t in recent_titles[-25:])
        recent_block = (
            f"📋 RECENT IDEAS (last 25) — do NOT repeat these topics or patterns:\n"
            f"{bullets}\n\n"
        )

    reject_block = ""
    if targeted_rejects:
        lines = "\n".join(
            f"  ✗ \"{r['title']}\" — too similar to \"{r['matched']}\""
            for r in targeted_rejects
        )
        reject_block = (
            f"❌ IDEAS REJECTED THIS SESSION (too similar to history):\n{lines}\n"
            f"Generate completely different subjects — not just different wording.\n\n"
        )

    favorites_block = ""
    if favorites:
        fav_lines = []
        for fav in favorites[-10:]:   # use up to 10 most recent favorites
            title   = fav.get("title", "")
            area    = fav.get("area", "")
            hook    = fav.get("viral_hook", "")
            pattern = fav.get("pattern_used", "")
            fav_lines.append(
                f"  ★ \"{title}\" [{area}]\n"
                f"    Hook: {hook}\n"
                + (f"    Pattern: {pattern}\n" if pattern else "")
            )
        favorites_block = (
            f"❤️ USER'S LOVED IDEAS — style templates to emulate:\n"
            f"The user marked these ideas as favourites. Study the EMOTIONAL TRIGGER, "
            f"HOOK STRUCTURE, and TYPE OF SHOCK each uses, then apply those exact same "
            f"patterns to COMPLETELY DIFFERENT subjects and areas. Do NOT repeat the same "
            f"topics — extract the WHY and apply it elsewhere.\n\n"
            + "\n".join(fav_lines)
            + "\nAsk yourself: 'What emotional response does each favourite trigger, and what "
            f"totally different subject would trigger the SAME response?'\n\n"
        )

    # ── Search variety blocks ─────────────────────────────────────────────────
    # Tell the LLM which queries have already been used so it picks different ones
    searched_block = ""
    if used_queries:
        searched_list = "\n".join(f"  • {q}" for q in used_queries)
        searched_block = (
            f"🔍 ALREADY SEARCHED — do NOT repeat these queries or phrases:\n"
            f"{searched_list}\n"
            f"Reframe your searches with completely different keywords, source types, and angles.\n\n"
        )

    # Also tell the LLM to avoid banned topic words in its SEARCH QUERIES (not just idea subjects)
    search_avoid_block = ""
    if banned_words:
        avoid_str = ", ".join(sorted(banned_words))
        search_avoid_block = (
            f"🔎 AVOID IN SEARCH QUERIES — these topic words already have many existing ideas, "
            f"so searching for them will only produce more duplicates. Do NOT use any of these "
            f"words as primary keywords in your web or YouTube searches:\n{avoid_str}\n\n"
        )

    # Rotate the search STRUCTURE each batch — drives to different story patterns
    angle_label, angle_hints = _BATCH_ANGLES[batch_num % len(_BATCH_ANGLES)]
    angle_block = (
        f"🔎 THIS BATCH — Search for: {angle_label}\n"
        f"{angle_hints}\n"
        f"Use specific, niche queries. Avoid generic 'interesting facts' searches.\n\n"
    )

    # Mandatory area targets — one idea per area, enforced in code too
    areas_for_this_batch = target_areas or AREAS[:5]
    area_list = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(areas_for_this_batch))
    target_block = (
        f"🎯 TARGET AREAS — generate EXACTLY one idea per area, in any order:\n"
        f"{area_list}\n"
        f"Each idea's 'area' field must match one of these exactly. "
        f"Do NOT generate two ideas for the same area.\n\n"
    )

    # Audience insights block — injected when a recent analysis exists
    insights_block = ""
    if audience_insights:
        top    = audience_insights.get("top_areas", [])[:5]
        angles = audience_insights.get("trending_angles", [])[:5]
        summary = audience_insights.get("audience_insights", "")
        analyzed_at = audience_insights.get("analyzed_at", "")[:10]
        if top or summary:
            angle_lines = "\n".join(f"  • {a}" for a in angles) if angles else ""
            insights_block = (
                f"📊 AUDIENCE INSIGHTS (from analysis on {analyzed_at}):\n"
                f"Top areas for Telugu audience right now: {', '.join(top)}\n"
                + (f"Trending angles:\n{angle_lines}\n" if angle_lines else "")
                + (f"Audience context: {summary}\n" if summary else "")
                + f"↑ When your target areas overlap with the top areas above, pick angles "
                f"that match the trending patterns. Prioritise these styles within those areas.\n\n"
            )

    user_message = (
        f"{insights_block}"
        f"{favorites_block}"
        f"{banned_block}"
        f"{recent_block}"
        f"{reject_block}"
        f"{searched_block}"
        f"{search_avoid_block}"
        f"{target_block}"
        f"{angle_block}"
        f"PHASE 1 — ONE YouTube search to extract viral FORMAT (title structure, hook style).\n"
        f"PHASE 2 — Specific web searches to find REAL stories with irony, absurdity, or punchline reversals.\n\n"
        f"⚠️ Use the viral structures from the system prompt. "
        f"Dry educational facts are NOT acceptable. Each title must have the punchline IN IT."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    print(f"\n[Idea Generator] LLM call (batch {batch_num + 1}/{iteration + 1}, "
          f"focus={angle_label}, "
          f"banned={len(banned_words)} words, "
          f"history={len(recent_titles)} titles)...")

    new_queries: list[str] = []

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    while response.tool_calls:
        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            # args is normally a dict, but some LangChain builds pass a JSON string
            args = tc.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"query": args}
            query = args.get("query", "") if isinstance(args, dict) else ""
            if query:
                new_queries.append(query)
            if tool_name == "search_youtube_shorts":
                print(f"[Idea Generator] YouTube: {query[:60]}")
                result = search_youtube_shorts.invoke(args)
            else:
                print(f"[Idea Generator] Web search: {query[:60]}")
                result = web_search.invoke(args)
            messages.append(ToolMessage(content=str(result), tool_call_id=tc.get("id", "")))
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    raw = response.content
    # Claude sometimes returns content as a list of blocks; extract just the text
    if isinstance(raw, list):
        raw = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in raw
        )
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        ideas_raw = parsed.get("ideas", [])
        # LLM sometimes returns ideas as a dict {"1": {...}, "2": {...}}
        if isinstance(ideas_raw, dict):
            ideas_raw = list(ideas_raw.values())
        return [i for i in ideas_raw if isinstance(i, dict)], new_queries
    except Exception:
        return ([{"title": f"Idea {i+1}", "area": "science",
                  "concept": line.strip(), "viral_hook": "", "pattern_used": ""}
                 for i, line in enumerate(raw.split("\n")) if line.strip()][:3],
                new_queries)


# ── Node 1: Generate ideas ─────────────────────────────────────────────────────

def idea_generator_agent_node(state: dict) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY"),
        temperature=0.9,
    )
    web_search     = get_tavily_search_tool(max_results=6)
    llm_with_tools = llm.bind_tools([search_youtube_shorts, web_search])

    iteration  = state.get("idea_gen_iteration", 0)
    prev_ideas = state.get("generated_ideas", [])

    history          = _load_history()
    blacklist        = _load_blacklist()
    favorites        = _load_favorites()
    audience_data    = load_audience_insights()
    prev_dicts       = [i for i in prev_ideas if isinstance(i, dict)]

    # all_used_ideas: history + blacklist + session ideas, deduplicated by title
    # Used for semantic dedup (title + concept compared) so same-study ideas are caught
    seen_titles: set = set()
    all_used_ideas: list[dict] = []
    for idea in prev_dicts + history + blacklist:
        t = idea.get("title", "")
        if t not in seen_titles:
            seen_titles.add(t)
            all_used_ideas.append(idea)

    # all_used_titles: title strings only, for the LLM prompt
    all_used_titles = [i.get("title", "") for i in all_used_ideas]

    # Adaptive threshold: need more occurrences to declare a topic exhausted
    threshold    = max(3, len(all_used_ideas) // 15)
    banned_words = _overused_words(all_used_ideas, min_count=threshold)

    # Pick 5 least-covered areas to balance the distribution across sessions
    target_areas = _pick_target_areas(all_used_ideas, n=TARGET_IDEAS)

    from collections import Counter as _Counter
    area_counts = _Counter(d.get("area", "").lower() for d in all_used_ideas if d.get("area"))
    print(f"[Idea Generator] History: {len(history)} | Favorites: {len(favorites)} | "
          f"threshold={threshold} | Banned ({len(banned_words)}): {', '.join(sorted(banned_words))}")
    print(f"[Idea Generator] Target areas: {target_areas}")
    print(f"[Idea Generator] Area counts: { {a: area_counts.get(a, 0) for a in AREAS} }")

    # ── Retry loop: generate → check → replace duplicates ────────────────────
    accepted:         list[dict] = []
    accepted_areas:   set[str]   = set()  # enforce max 1 idea per area per batch
    targeted_rejects: list[dict] = []
    session_queries:  list[str]  = []     # accumulates across retries for dedup

    for attempt in range(MAX_RETRIES):
        if len(accepted) >= TARGET_IDEAS:
            break

        # Remaining areas needed for this attempt
        remaining_areas = [a for a in target_areas if a not in accepted_areas]

        candidates, new_queries = _call_llm(
            llm_with_tools, web_search,
            banned_words, all_used_titles,
            targeted_rejects, iteration,
            favorites=favorites,
            used_queries=session_queries,
            batch_num=attempt,
            target_areas=remaining_areas or target_areas,
            audience_insights=audience_data,
        )
        session_queries.extend(new_queries)

        for idea in candidates:
            if len(accepted) >= TARGET_IDEAS:
                break
            if not isinstance(idea, dict):
                continue
            title = idea.get("title", "")
            area  = idea.get("area", "").lower().strip()

            # Code-level area dedup — reject if this area already has an accepted idea
            if area in accepted_areas:
                print(f"[Idea Generator] ✗ AREA DUP  '{title}' (area '{area}' already used)")
                targeted_rejects.append({"title": title, "matched": f"area '{area}' already filled"})
                continue

            # Keyword-level dedup against history + already accepted
            too_sim, matched = _is_too_similar(
                idea, all_used_ideas + accepted, banned_words
            )
            if too_sim:
                print(f"[Idea Generator] ✗ REJECTED  '{title}'")
                print(f"                    ~ too similar to '{matched}'")
                targeted_rejects.append({"title": title, "matched": matched})
            else:
                print(f"[Idea Generator] ✓ ACCEPTED  '{title}' [{area}]")
                accepted.append(idea)
                accepted_areas.add(area)
                all_used_ideas.append(idea)
                all_used_titles.append(title)

        if len(accepted) < TARGET_IDEAS and attempt < MAX_RETRIES - 1:
            still_needed = [a for a in target_areas if a not in accepted_areas]
            print(f"[Idea Generator] {len(accepted)}/{TARGET_IDEAS} — retry "
                  f"({attempt + 2}/{MAX_RETRIES}), need areas: {still_needed}")

    print(f"\n[Idea Generator] Final: {len(accepted)} unique ideas")

    if accepted:
        _save_to_history(accepted)   # also runs _compact_history() internally
        history_size = len(_load_history())
        print(f"[Idea Generator] Saved {len(accepted)} ideas — history now {history_size} entries")

    return {
        "generated_ideas":    accepted,
        "selected_idea":      None,
        "idea_gen_iteration": iteration + 1,
        "messages":           [],
    }


# ── Node 2: Wait for user's pick ───────────────────────────────────────────────

def idea_selector_node(state: dict) -> dict:
    ideas = state.get("generated_ideas", [])

    choice = interrupt({
        "ideas":   ideas,
        "message": "Pick 1, 2, or 3 — or say 'more' for new ideas",
    })

    choice_str = str(choice).strip().lower()

    if choice_str == "more":
        print("[Idea Selector] User requested more ideas.")
        return {"selected_idea": None}

    try:
        idx = int(choice_str) - 1
        if not (0 <= idx < len(ideas)):
            idx = 0
    except ValueError:
        idx = 0

    selected    = ideas[idx]
    input_topic = f"{selected['title']} — {selected['concept']}"
    print(f"[Idea Selector] Selected: \"{selected['title']}\"")

    return {
        "selected_idea": selected,
        "input_topic":   input_topic,
    }
