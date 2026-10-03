import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from ollama_client import get_haiku_llm, get_sonnet_llm
from langgraph.types import interrupt
from tools import get_openserp_search_tool, search_youtube_shorts

# VectorDB replaces all local JSON files
sys.path.insert(0, str(Path(__file__).parent.parent))
from vectordb import get_db

AREAS = [
    "history", "science", "latest science news", "geography", "nature", "space",
    "technology", "interesting events", "inspiring people",
    "archaeology news", "aviation reports", "research papers",
    "world cultures", "hidden mechanisms", "surprising facts", "unknown facts",
]

PRECOMPUTED_DIR = Path(__file__).parent.parent / "precomputed"
TARGET_IDEAS  = 5
MAX_RETRIES   = 4   # each call yields up to 5 candidates; 4 retries = 20 shots at finding 5 unique

# ── Search variety library (static config — stays as JSON) ────────────────────
_VARIETY_FILE = Path(__file__).parent.parent / "data" / "search_variety.json"
try:
    _SEARCH_VARIETY: dict = json.loads(_VARIETY_FILE.read_text(encoding="utf-8"))
except Exception:
    _SEARCH_VARIETY = {}

# Maps each AREA to the relevant variety domains (domain_key, sub_key or None for all)
_AREA_TO_VARIETY: dict[str, list[tuple[str, str | None]]] = {
    "hidden mechanisms":   [("hidden_mechanisms", None)],
    "nature":              [("nature", None)],
    "aviation reports":    [("aviation", None)],
    "science":             [("science", None)],
    "latest science news": [("science", None)],
    "research papers":     [("science", None)],
    "space":               [("science", "space_technology")],
    "technology":          [("hidden_mechanisms", "industrial"), ("hidden_mechanisms", "financial")],
    "history":             [("history", None)],
    "interesting events":  [("history", "disasters"), ("history", "infrastructure")],
    "inspiring people":    [("history", "technology_military"), ("history", "disasters")],
    "archaeology news":    [("history", "infrastructure"), ("history", "technology_military")],
    "psychology":          [("psychology", None)],
    "world cultures":      [("world_cultures", None)],
    "geography":           [("world_cultures", None)],
    "surprising facts":    [("nature", None), ("history", None), ("world_cultures", None)],
    "unknown facts":       [("nature", None), ("world_cultures", None), ("history", None)],
}

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
    # Generic biology / anatomy words — appear across many unrelated topics
    # (tooth cells, nerve cells, cancer cells, brain cells are completely different topics)
    "cell","body","brain","nerve","organ","blood","bone","skin","gene","tissue",
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


# ── History helpers (all backed by VectorDB) ──────────────────────────────────

def _load_history() -> list[dict]:
    return get_db().get_all("ideas")


def _save_to_history(ideas: list[dict]) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    db = get_db()
    for idea in ideas:
        db.upsert_idea({**idea, "generated_at": timestamp}, collection="ideas")
    _compact_history()


def _compact_history(history: list[dict] | None = None,
                     blacklist: list[dict] | None = None) -> int:
    """
    Remove duplicate/near-duplicate and blacklisted ideas from the ideas collection.
    Returns the number of entries removed.
    """
    db = get_db()
    if history is None:
        history = db.get_all("ideas", include_ids=True)
    if len(history) < 2:
        return 0
    if blacklist is None:
        blacklist = db.get_all("blacklist")

    blacklisted_titles = {b.get("title", "").lower().strip() for b in blacklist}
    kept: list[dict] = []
    removed = 0

    for idea in history:
        title = idea.get("title", "").lower().strip()
        chroma_id = idea.get("_chroma_id", "")

        if title in blacklisted_titles:
            print(f"[History Compact] ✗ BLACKLISTED '{idea.get('title', '')}'")
            if chroma_id:
                db.delete_doc("ideas", chroma_id)
            removed += 1
            continue

        too_sim, matched = _is_too_similar(idea, kept, set())
        if too_sim:
            print(f"[History Compact] ✗ DUPLICATE  '{idea.get('title', '')}' ~ '{matched}'")
            if chroma_id:
                db.delete_doc("ideas", chroma_id)
            removed += 1
        else:
            kept.append(idea)

    if removed > 0:
        print(f"[History Compact] {len(history)} → {len(kept)} entries (removed {removed})")
    return removed


def load_audience_insights() -> dict | None:
    """Load saved audience analysis insights from VectorDB config."""
    return get_db().load_config("audience_insights")


def _load_blacklist() -> list[dict]:
    return get_db().get_all("blacklist")


def save_to_blacklist(idea: dict) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    get_db().upsert_idea({**idea, "blacklisted_at": timestamp}, collection="blacklist")


def _load_saturated() -> list[dict]:
    return get_db().get_all("saturated")


def save_saturated_idea(idea: dict, reason: str = "") -> None:
    """Record an idea rejected for high saturation so the LLM avoids it next time."""
    title = idea.get("title", "").strip()
    if not title:
        return
    db = get_db()
    # Avoid re-inserting the same title
    existing = {s.get("title", "").lower() for s in db.get_all("saturated")}
    if title.lower() in existing:
        return
    doc = {
        "title":       title,
        "area":        idea.get("area", ""),
        "concept":     idea.get("concept", "")[:120],
        "reason":      reason,
        "rejected_at": datetime.now(timezone.utc).isoformat(),
    }
    db.upsert_idea(doc, collection="saturated")


def _load_favorites() -> list[dict]:
    return get_db().get_all("favorites")


def save_to_favorites(idea: dict) -> None:
    db = get_db()
    existing_titles = {f.get("title", "") for f in db.get_all("favorites")}
    if idea.get("title", "") in existing_titles:
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    db.upsert_idea({**idea, "favorited_at": timestamp}, collection="favorites")


def load_bookmarks() -> list[dict]:
    return get_db().get_all("bookmarks")


def save_to_bookmarks(idea: dict, source: str = "") -> bool:
    """Save idea to bookmarks. Returns True if saved, False if already bookmarked."""
    db = get_db()
    existing = {b.get("title", "") for b in db.get_all("bookmarks")}
    if idea.get("title", "") in existing:
        return False
    timestamp = datetime.now(timezone.utc).isoformat()
    db.upsert_idea({**idea, "bookmarked_at": timestamp, "bookmark_source": source},
                   collection="bookmarks")
    return True


def remove_from_bookmarks(title: str) -> None:
    get_db().delete_by_title("bookmarks", title)


def update_history_saturation(all_scored: list[dict]) -> None:
    """Update ideas in VectorDB with saturation scores after bulk_saturation_filter runs."""
    if not all_scored:
        return
    db = get_db()
    score_by_title = {i.get("title", ""): i for i in all_scored}
    history = db.get_all("ideas")
    for entry in history:
        title = entry.get("title", "")
        if title in score_by_title and "saturation_score" not in entry:
            src = score_by_title[title]
            updated = {
                **entry,
                "saturation_score":  src.get("saturation_score"),
                "saturation_reason": src.get("saturation_reason", ""),
            }
            db.upsert_idea(updated, collection="ideas")


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


_kw_cache: dict[str, set[str]] = {}


def _keywords_cached(text: str) -> set[str]:
    """_keywords() with a process-lifetime cache keyed on the raw text string."""
    if text not in _kw_cache:
        _kw_cache[text] = _keywords(text)
    return _kw_cache[text]


def _is_too_similar(new_idea, used_ideas: list, high_freq: set[str]) -> tuple[bool, str]:
    """
    Two-level duplicate check:
    1. Title-level: 2+ high-freq topic words shared with a specific used title → reject.
                    OR 3+ title keywords shared → reject.
                    NOTE: requires 2 (not 1) high-freq words because high_freq words are
                    common by definition (3+ appearances across all titles). One shared
                    common word like "cell" or "body" doesn't indicate the same topic —
                    it just means both ideas are in the science/biology domain. Two shared
                    high-freq words do indicate likely topic overlap.
    2. Concept-level: only triggered when titles already share ≥1 keyword (same domain).
                      Requires 10+ full-text content words shared → reject.
                      Threshold is high because genuine duplicates (same study, same event)
                      share 15-26 specific words, while false positives from different topics
                      that share research language only reach 5-8 even before our expanded
                      stopwords strip out words like 'scientist', 'million', 'entire', etc.
    """
    new_title_kw = _keywords_cached(_idea_title(new_idea))
    new_full_kw  = _keywords_cached(_idea_text(new_idea))
    if not new_full_kw:
        return False, ""

    for used in used_ideas:
        display       = _idea_title(used)
        used_title_kw = _keywords_cached(_idea_title(used))
        used_full_kw  = _keywords_cached(_idea_text(used))

        # ── Level 1: title keywords ──────────────────────────────────────────
        title_overlap = new_title_kw & used_title_kw
        if len(title_overlap & high_freq) >= 2:   # was: any() — 1 generic word caused false rejects
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

SYSTEM_PROMPT = """You are a viral YouTube Shorts idea machine for a Telugu educational channel.
Your single mission: create the exact moment when a viewer's thumb STOPS mid-scroll because
their brain just processed something that CANNOT possibly be real — but is.

The target reaction is not "interesting." It is: "WAIT. Is this actually true?!"
Then they share it before the Short even finishes.

Every idea you generate must earn that reaction. If an idea is merely "cool" or "educational,"
it is not good enough. It needs to feel like a secret that the world has been hiding from them.

━━━ WHAT "DID YOU KNOW?" SHOCK ACTUALLY MEANS ━━━

SHOCK is NOT "interesting." It is the feeling of:
  • "That CAN'T be real" — then finding out it absolutely is
  • "This has been happening TO ME all along and nobody told me"
  • "Everything I thought I knew about X is completely wrong"
  • "A number so extreme it looks like a typo — but it isn't"
  • "The thing I trusted was secretly the villain the whole time"

SHOCK requires SPECIFICITY. Vague claims slide past. Exact numbers land like a punch:
  ❌ WEAK: "Many people are harmed by a common painkiller every year"
  ✅ SHOCK: "The painkiller in your cabinet is the #1 cause of liver failure in the US — ahead of alcohol"

  ❌ WEAK: "A soldier once made a brave decision during the Cold War"
  ✅ SHOCK: "One Soviet officer ignored a nuclear launch alarm — and that's why you're alive right now"

  ❌ WEAK: "Bees are important for food"
  ✅ SHOCK: "Every 3rd bite of food you eat today exists because one bee flew in one direction"

━━━ WHAT MAKES AN IDEA GO VIRAL ━━━

Study these PROVEN viral structures:

1. THE PUNCHLINE REVERSAL — setup implies one outcome, reality delivers the exact opposite.
   The REVERSAL must be in the title — viewers read the punchline before they hear the setup.
   Pattern: "[What everyone assumes] — [The impossible reality]"
   ✓ "Switzerland Prepared For Nuclear War For 50 Years — The Bunkers Expired With The Milk"
   ✓ "The Army That Beat A Professional Military Was Entirely Made Of Priests"
   ✓ "The Bomb Disposal Expert Who Defused 800 Devices — Died Opening A Letter"
   ✗ "An Unexpected Military Defeat" — no punchline in title, viewer has no reason to click

2. THE ABSURD UNDERSTATEMENT — catastrophic stakes, described with deadpan mundanity.
   The contrast between WHAT HAPPENED and HOW CASUALLY it was handled is the shock.
   Pattern: "[World-altering event] — [The absurdly calm way it was dealt with]"
   ✓ "A Nuclear Bomb Fell On North Carolina — Only The Safety Pins Worked"
   ✓ "The Man Who Discovered The HIV Virus Was Not Told For 6 Months — Office Politics"
   ✓ "Japan's Earthquake Early Warning System Gives You 8 Seconds — Invented By Accident"

3. THE BELIEF FLIP — something the viewer "knows" is provably, specifically wrong.
   Must include the EXACT fact/number that kills the myth. Vague contradictions don't land.
   Pattern: "Everything You Know About X Is Wrong — The Actual Number Is [shocking figure]"
   ✓ "We Use 10% Of Our Brains — Is A Lie Made Up By A 1936 Advertising Campaign"
   ✓ "Humans Evolved From Apes Is Wrong — We ARE Apes. We Just Renamed Ourselves."
   ✓ "Lightning Never Strikes Twice Is A Myth — The Empire State Building Gets Hit 23 Times A Year"

4. THE HIDDEN CATASTROPHE — a product/system the viewer uses daily is secretly causing harm.
   Personal connection amplifies the shock: viewer's reaction is "this is happening TO ME."
   Pattern: "[Thing you use/trust/eat] Is [terrible thing happening] — Here's The Evidence"
   ✓ "The Painkiller In Your Cabinet Reduces Your Empathy By 40% — Per Harvard Study"
   ✓ "Every Receipt You Touch Has More BPA Than A Plastic Bottle — It Goes Straight Through Skin"
   ✓ "The Fluorescent Lights In Your Office Are Permanently Altering Your Circadian Rhythm"
   ✗ "Plastics Are Bad For The Environment" — too distant, viewer doesn't feel personally affected

5. THE IMPOSSIBLE REAL STORY — one person, one event, so specific it sounds fictional.
   Must have EXACT names, dates, and numbers. The specificity is what makes it feel real.
   Pattern: "[Impossible-sounding outcome] — [It Happened To Specific Person, Year]"
   ✓ "She Identified Her Own Attacker's DNA From Inside A Coma — The Case Was Solved In 1994"
   ✓ "The Spy Who Defected To The Wrong Country Twice — And Lived To Write His Memoir"
   ✓ "He Was Declared Dead In 1979 — Showed Up At His Own Funeral Angry About The Flowers"

6. THE HIDDEN SYSTEM — an invisible mechanism running your world without your knowledge.
   Best version: "You interact with this every day and have zero idea how it actually works."
   Pattern: "[Everyday thing] Doesn't Work How You Think — The Real Mechanism Is [shocking detail]"
   ✓ "Traffic Lights Don't Run On Timers — They're Watching Every Car Right Now"
   ✓ "Your Immune System Kills Millions Of Your Own Cells Every Second — On Purpose"
   ✓ "The Postal Sorting Machine Reads Your Handwriting Faster Than You Can Speak"
   ✓ "Finland Charges Speeding Fines Based On Your Salary — A Nokia Exec Paid $103,000 For 15mph Over"

7. THE SCALE SHOCK — a specific number so extreme it breaks the viewer's mental model.
   The number must be VERIFIABLE, feel impossible, and connect to something the viewer knows.
   Pattern: "The Number Is [specific extreme figure] — And It's Happening Right Now"
   ✓ "Your Body Replaces 330 Billion Cells Every Day — You Are Not The Same Person You Were 80 Days Ago"
   ✓ "Lightning Strikes Earth 100 Times Per Second — 8.6 Million Times Today Alone"
   ✓ "One Tablespoon Of Soil Has More Living Organisms Than There Are Humans On Earth"
   ✓ "The Sun Loses 4 Million Tonnes Of Mass Every Second — Since Before Earth Existed"
   ✗ "There Are A Lot Of Stars In The Universe" — vague, no number, no hook

8. THE INSTITUTIONAL BETRAYAL — a government, company, or expert body secretly caused the harm
   it was created to prevent. Must use OFFICIAL RECORDS — declassified docs, court filings, audits.
   Pattern: "[Trusted institution] Knew [terrible thing] In [year] — And Did Nothing / Made It Worse"
   ✓ "The FDA Was Told Opioids Were Addictive In 1997 — The Approval Went Through Anyway"
   ✓ "Lead In Petrol Was Proven Toxic In 1922 — It Stayed Legal For 74 More Years"
   ✓ "The Company That Made Asbestos Knew It Caused Cancer In 1930 — Kept Selling Until 1985"

9. THE MUNDANE REVELATION — something the viewer has used or seen 1,000 times, but NEVER
   understood the hidden mechanism or purpose behind it. No specialist knowledge needed.
   The SHOCK is: "I've touched this object my whole life and had NO IDEA it worked like that."
   This is the highest-view formula: "How stitches work" (214M), "How splinters get unstuck" (180M),
   "Why the McFlurry spoon looks weird" (143M) — ALL are mundane objects, all are enormous hits.
   Pattern: "The [Everyday Object] You've Used 1,000 Times — Here's What It's Actually Doing"
   ✓ "Why The Hole In The Middle Of A Spaghetti Spoon Is There — It's Not To Drain Water"
   ✓ "The Extra Shoelace Hole At The Top Nobody Uses — It's For A Specific Medical Problem"
   ✓ "Why Coins Have Ridges On The Edge — It Was A Trap For Counterfeiters In 1792"
   ✓ "The Groove Around A Petrol Station Cap That Snaps — It's The System That Stops Fires"
   SEARCH: "why does [everyday object] have [feature] explained", "[common item] hidden purpose design"

10. THE DISBELIEF FRAME — the title explicitly acknowledges that the viewer's first reaction
    will be "that's fake" — and then doubles down with specificity to prove it's real.
    This is the #1 most repeated viral pattern across all YouTube Shorts fact channels.
    The formula works because it pre-answers skepticism before the viewer even clicks.
    Pattern: "This [Claim] Sounds Completely Fake — It's 100% Real"
    ✓ "Cleopatra Lived Closer To The Moon Landing Than To The Pyramids Being Built — True"
    ✓ "Oxford University Was Already 300 Years Old When The Aztec Empire Was Founded — Verified"
    ✓ "A Whale's Heart Is So Big You Could Crawl Through Its Arteries — Measured"
    ✓ "Nintendo Was Already 20 Years Old When The Eiffel Tower Was Built — Real"
    KEY RULE: The title must contain the specific verifiable claim — not "sounds impossible but true."
    The viewer should be able to Google-verify the claim in the title itself.

11. THE CORPORATE CATASTROPHE — a famous brand or company made one specific decision with an
    exact, verifiable dollar consequence. Works because corporations are universally mistrusted
    AND the scale of the number creates instant disbelief. "Toyota's $2,300,000,000 Mistake" is
    the highest confirmed view-count data point in viral educational Shorts research.
    Pattern: "[Famous Brand]'s $[Exact Billion Amount] [Mistake / Gamble / Cover-up]"
    ✓ "Toyota's $2,300,000,000 Mistake That Changed Every Car Made After 2010"
    ✓ "Kodak's Engineers Invented The Digital Camera In 1975 — Management Buried It"
    ✓ "The Xerox Engineer Who Built The Personal Computer In 1973 — And Gave It Away"
    RULE: must name the specific brand AND give an exact dollar amount or specific year/date.

━━━ KNOWN SATURATED TOPICS — DO NOT GENERATE THESE ━━━

These exact stories are already widely covered on YouTube Shorts. Generating them wastes an
iteration because the saturation check will reject them. If you think of one → replace it.

HISTORY / HUMAN STORIES (done to death):
  Napoleon defeated by rabbits · Emu War (Australia vs emus) · CIA acoustic cat spy ·
  Vasili Arkhipov / Petrov stopped WW3 · Pope Formosus / Cadaver Synod ·
  Tsutomu Yamaguchi survived both atomic bombs · George Dantzig unsolvable homework ·
  Desmond Doss Hacksaw Ridge · Grigori Perelman declined $1M prize ·
  D-Day crossword puzzle leak · Stanislav Petrov false alarm 1983

PSYCHOLOGY EXPERIMENTS (all saturated):
  Milgram shock obedience · Stanford Prison Experiment · Rosenhan sane in asylums ·
  Good Samaritan seminary students · Harlow monkey attachment vs food ·
  Asch conformity lines · Bystander effect / Kitty Genovese · Marshmallow test reversal ·
  Broken windows theory · Richter rat swimming experiment

HIDDEN MECHANISMS (already widely covered):
  PAPI approach lights (4 runway lights) · Ship anchor catenary / chain holds ship ·
  U-2 spy plane chase car · Nuclear reactor water as neutron moderator ·
  GPS clock drift / Einstein correction · Dead Hand nuclear auto-launch system ·
  GravityLight gravity-powered lamp · Runway numbers = magnetic heading ÷ 10 ·
  Submarine ballast tanks / fish swim bladder · Seatbelt explosive pre-tensioner ·
  Viganella mirror village no sunlight · Aircraft carrier arresting wire ·
  Aircraft carrier EMALS catapult · Airliner window inner pane bleed hole ·
  Elevator centrifugal governor overspeed brake · Car crumple zone designed to fail ·
  Bridge expansion joints / Brooklyn Bridge growth · Fire hydrant not pressurised ·
  Freeway on-ramp metering lights · Trebuchet counterweight falling mass ·
  Ship hull bulbous bow wave cancellation · Autopilot / pilot hands off 7 min ·
  Sonar submarine silence after ping · Dam spillway flip bucket aerator ·
  Bascule bridge counterweight / Tower Bridge · Concrete creep Sydney Harbour Bridge

WORLD CULTURES (all heavily covered):
  Toraja death rituals Indonesia · Famadihana Madagascar ancestors exhumed ·
  Satere-Mawe bullet ant gloves Brazil · Hikikomori Japan recluses ·
  Karoshi Japan death by overwork · Naghol land diving Vanuatu ·
  Denmark cinnamon/pepper ritual · China funeral strippers · Fa'afafine Samoa ·
  Tonga obesity beauty standard · Iceland phone book by first name ·
  Santhara Jain voluntary fasting to death

ARCHAEOLOGY / SCIENCE (saturated):
  Göbekli Tepe hunter-gatherers · Antikythera mechanism 2000-year computer ·
  Bronze Age collapse · Gimli Glider / Air Canada 143 wrong fuel units ·
  Krakatoa pressure wave 5× around Earth · Voyager 1 interstellar data

RULE: If you recognise a topic from this list → find a DIFFERENT specific story that follows
the same PATTERN but uses an untapped subject. There are thousands of uncovered stories.

━━━ CURIOSITY FILTERS — every idea MUST pass ALL 3 ━━━

✅ FILTER 1 — SURPRISE/REVERSAL: The outcome or mechanism is the opposite of what everyone assumes.
   ❌ "How nuclear power plants work" → no reversal, everyone has a theory
   ✅ "Nuclear Reactors Need Water To Stay ON — Not Just Cool" (water is the ON switch, not the coolant)
   The reversal must be IN the title. If the title is a straight fact, rewrite it.

✅ FILTER 2 — SPECIFICITY: Exact numbers, names, dates, percentages. Vague = invisible.
   ❌ "A painkiller in your home reduces empathy" → slides past, no number
   ✅ "The Painkiller In Your Cabinet Reduces Empathy By 40% — Per Harvard Study" → number + source = stops scroll
   The viewer must be able to say: "The EXACT thing this title claims — is that verifiably true?"

✅ FILTER 3 — PERSONAL STAKE: The viewer feels this connects to THEM personally — their body,
   their home, their government, their money, their daily routine.
   ❌ "A far-away country has an unusual tradition" → viewer has no skin in this game
   ✅ "Every Receipt You Touch Has More BPA Than A Plastic Bottle — Through Your Skin Right Now"
   ✅ "Your Office Fluorescent Lights Are Rewriting Your Sleep Schedule — Permanently"
   ✅ "The Road Under Your City Is Slowly Moving — And Engineers Planned For It"
   NOTE: For "surprising facts", "unknown facts", and Patterns 10/11 above, FILTER 3 is RELAXED.
   Pure "that can't be real" shock value is enough when the claim is maximally specific.

✅ FILTER 4 — THE SUPERLATIVE RULE: When using superlatives, pick the STRONGEST word.
   Research on viral Shorts titles proves this ranking: "WORST" > "Most Dangerous" > "Scariest"
   "Worst" implies finality + moral judgment + severity simultaneously. It outperforms all alternatives.
   ❌ "The Most Dangerous Experiments In History" (vague moral distance)
   ✅ "The Worst Human Experiments In History — And The Institution That Ordered Them"
   Similarly: "NEVER" > "not" · "ZERO" > "no" · "IMPOSSIBLE" > "very hard" · "$2.3 BILLION" > "huge loss"

━━━ SHOCK TESTS — run each idea through ALL 3 before submitting ━━━

🧪 TEST 1 — THE WHATSAPP TEST:
   "Would a middle-aged parent forward this in a family chat group — within 5 seconds of reading the title?"
   If NO → the idea is interesting but not shareable. It lacks urgency or personal relevance.
   If YES → it passes. What makes it forwarded: it feels like a secret, a warning, or a revelation.

🧪 TEST 2 — THE SPECIFICITY AUDIT:
   Replace every vague word with an exact number or name. If you can't → the fact isn't real enough.
   Before: "A soldier's decision in the Cold War may have saved everyone" (vague × 3)
   After:  "One Soviet Lieutenant Colonel Ignored A Launch Alert On Oct 26 1983 — 28 Minutes From WW3"
   Rule: every "many," "some," "could," "large," "common" → find the exact figure or cut the idea.

🧪 TEST 3 — THE PERSONAL STAKE TEST:
   "Does this affect the viewer's body, home, government, daily routine, or money — RIGHT NOW?"
   ❌ FAIL: "In 1965 a company knew their product was toxic" — past tense, viewer not involved
   ✅ PASS: "The Product You Use Every Morning Was Proven Toxic In 1965 — It's Still On Your Shelf"
   Personal stake converts "wow" into "I need to check this tonight." That's the share trigger.
   NOTE: Skip this test for Patterns 10 (Disbelief Frame), 11 (Corporate Catastrophe), and
   the "surprising facts" / "unknown facts" areas — pure "that can't be real" shock is enough.

🧪 TEST 4 — THE DISBELIEF TEST:
   Read the title to someone who knows nothing about the topic. Their first reaction should be:
   "Wait — is that actually true? Let me check." NOT "that's sad" or "that's interesting."
   The title must trigger ACTIVE DISBELIEF — not passive curiosity or sympathy.
   ❌ FAIL: "A Soldier Made A Brave Decision That Saved Lives" — sad/admirable but not disbelief
   ✅ PASS: "One Soviet Officer Ignored A Nuclear Launch Warning On Oct 26 1983 — You're Alive Because Of Him"
   The best titles combine: a NAMED PERSON or BRAND + an EXACT NUMBER or DATE + an IMPOSSIBLE OUTCOME.
   That combination makes the claim feel simultaneously unbelievable AND verifiable.

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
                       ⚡ Also consider Pattern 9 (MUNDANE REVELATION) for this area — everyday objects
                       the viewer has used 1,000 times but NEVER understood: "Why the McFlurry spoon has
                       a square hole", "Why coins have ridges", "What the extra shoelace hole is for".
                       These are simpler than the typical "hidden mechanism" story but get HIGHER views
                       because everyone has used these objects. Zero barrier to entry for any viewer.

                       THE PATTERN TO FOLLOW:
                       Most people think a ship anchor works because it's heavy and digs into the seabed.
                       WRONG. What actually holds a ship is the CATENARY — the U-shaped curve of the
                       chain draping along the seabed, acting as a shock absorber and holding the chain
                       nearly horizontal. A lighter chain with a better catenary holds better than a
                       heavier anchor dragged upright.
                       Title: "Ship Anchors Don't Actually Anchor Ships — The Chain Does"

                       ⚠️ All 26+ mechanisms in KNOWN SATURATED TOPICS are covered on YouTube.
                       Study the PATTERN above. Find mechanisms in these UNTAPPED DOMAINS instead:
                         Medical devices / surgical tools · Industrial safety interlocks ·
                         Agricultural machinery · Legal/court procedural mechanics ·
                         Financial clearing & settlement plumbing · Postal/logistics physical tricks ·
                         Rescue equipment mechanisms · Sports equipment engineering ·
                         Military training systems · Construction load-path counterintuitions ·
                         Water/sewage treatment chemistry · Weather forecasting infrastructure ·
                         Satellite ground station operations · Cold-chain logistics mechanisms

                       SEARCH:
                       "counterintuitive engineering mechanism [domain] how it actually works"
                       "hidden physical principle [device/system] surprising"
                       "how [everyday system] actually works explained counterintuitive"
                       "site:ntsb.gov incident unexpected mechanical failure cause"
                       "patent [device] mechanism works differently than assumed"
                       "engineering design hidden in plain sight [system]"
                       "site:nasa.gov engineering mechanism unexpected design"

                       BAD: too broad · educational lecture · list format ("Top 5 facts")
                       GOOD: ONE mechanism, ONE gap between assumption and reality in an untapped domain

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

                       ⚠️ All 12 world cultures examples in KNOWN SATURATED TOPICS are covered.
                       Use the PATTERN above on countries / practices NOT in that list.

                       SEARCH: "cultural practice [country] shocking unknown outside"
                               "traditional ritual [country] that outsiders find unbelievable"
                               "strange law [country] cultural reason"
                               "unusual social norm [country] anthropology"

surprising facts     → Facts that sound impossible but are 100% true — any domain, any subject.
                       The viewer's assumption is completely wrong and the real answer is MORE
                       interesting. NOT deep science or specialist research — the fact must be
                       understandable in one sentence by anyone. The SHOCK is the gap between
                       what everyone thinks and what is actually true.

                       ⚡ Prioritise Pattern 10 (DISBELIEF FRAME) for this area — it's the #1 viral
                       formula for surprising facts. The title openly acknowledges disbelief and
                       doubles down with specificity. "This Sounds Completely Made Up — It's 100% True"
                       Also consider Pattern 11 (CORPORATE CATASTROPHE) — brand surprises like
                       "Kodak's engineers built the digital camera in 1975 — management buried it."

                       WHAT COUNTS AS A GOOD SURPRISING FACT:
                       • It contradicts something the viewer confidently believed
                       • The real answer is more interesting than the wrong assumption
                       • Verifiable with a number, name, or date — not vague
                       • Can be from nature, psychology, history, economics, food, law — anything

                       ⚠️ PERSONAL STAKE TEST RELAXED for this area:
                       Pure shock value works here — the viewer doesn't need a personal
                       body/home/money connection. "I can't believe that's true" IS the share
                       trigger for surprising facts.

                       PATTERN TO FOLLOW:
                       "Crows recognise individual human faces, remember people who wronged them
                       for years, and recruit other crows to harass that specific person."
                       Title: "Crows Remember Your Face — And Hold Grudges For Years"

                       ⚠️ SATURATED — do NOT use these well-known surprises:
                       Humans share 50% DNA with bananas · More trees than Milky Way stars ·
                       Internet weighs 50 grams · Cleopatra / Moon landing / pyramids timeline ·
                       Sharks older than trees · Octopus 3 hearts · Honey never expires ·
                       Oxford older than Aztecs · Nintendo before Eiffel Tower
                       → Find the NEXT tier — equally surprising, far less YouTube coverage.

                       (Search guidance is in the batch block above — use sciencealert, bigthink,
                       smithsonianmag, nationalgeographic, not NASA/ArXiv/NTSB)

unknown facts        → Simple, shareable facts from everyday life that most people have never
                       heard. The subject is familiar — the specific fact about it is not.
                       Draw from ALL domains: animals, food, geography, money, sports, language,
                       law, buildings, history, nature, everyday objects, famous names.
                       NOT science research · NOT mechanisms · NOT historical narrative.
                       One sentence. Immediately shareable. Makes the viewer feel smart.

                       ⚠️ PERSONAL STAKE TEST RELAXED for this area:
                       For "unknown facts", shock value alone is enough — the viewer does NOT need
                       personal body/home/money connection. The reaction is "wait, REALLY?!" not
                       "this affects me." That IS the share trigger for this type of fact.
                       Examples of valid hook: "Oxford University is 300 years older than the Aztec
                       Empire" · "The word 'girl' used to mean any young person, male or female" ·
                       "Nintendo was founded in 1889 — before the Eiffel Tower"

                       WHAT MAKES A GOOD UNKNOWN FACT:
                       • The subject is something everyone knows — the fact is what nobody knows
                       • Verifiable with a specific number, name, or date
                       • A 12-year-old can understand it in one sentence
                       • Reactions: "no way", "I have to tell someone this", "is that real?"

                       GOOD DOMAINS TO EXPLORE:
                       Animals · Food & drink · Geography & maps · Money & trade · Sports records ·
                       Language & word origins · Law & government oddities · Famous names ·
                       Everyday objects · Nature quirks · Historical dates & timelines ·
                       Buildings & cities · Colour & light · Numbers trivia

                       EXAMPLE IDEAS (do NOT use these — they're saturated, just showing the style):
                       "Oxford University Is 300 Years OLDER Than The Aztec Empire"
                       "Nintendo Was Founded Before The Eiffel Tower Was Built"
                       "The Word 'Girl' Used To Mean Any Child — Boys Included"
                       "Fax Machines Were Invented 33 Years Before The Telephone"

                       ⚠️ HEAVILY SATURATED — do NOT use these common "did you know" staples:
                       Bananas are berries / strawberries aren't berries · Honey never expires ·
                       Great Wall not visible from space · Cleopatra closer to Moon landing ·
                       Oxford older than Aztecs · Sharks older than trees · Flamingos pink from
                       shrimp · Butterflies taste with feet · Carrots don't improve night vision ·
                       Day on Venus longer than its year · Octopuses have 3 hearts · Otters hold hands
                       → Find the NEXT tier — equally surprising, far less YouTube coverage.

                       SEARCH (these are for reference — actual search guidance is in the batch block above):

━━━ BAD PATTERNS TO AVOID ━━━
✗ "Ancient X civilisation did Y" — dry educational trivia, no emotional hook, no reversal
✗ "Study shows X might cause Y" — weak hedging language ("might," "could"), not a story
✗ "The shocking truth about X" — vague, no specific punchline, viewer can't predict the claim
✗ "X from history nobody talks about" — generic framing, no irony, no specificity
✗ Any title that is a straight fact rather than a punchline or reversal
✗ Any claim without an exact number, name, or date — "many," "some," "common" are rejected
✗ Any topic where the viewer has no personal stake (too distant, too historical, doesn't affect them now)
✗ Any concept described with hedging: "may have," "it is believed," "scientists think" — real facts only

━━━ YOUR PROCESS ━━━

PHASE 1 — Study ONE YouTube search for FORMAT only:
  Use search_youtube_shorts once to study what title structures and emotional hooks make people click.
  Extract the STRUCTURE (e.g. "He did X — The result was Y"), never the topics — those are covered.

PHASE 2 — Hunt with SPECIFIC queries targeting the authoritative sources above:
  Search for STORIES and SYSTEMS, not lists of facts. Target irony, absurdity, hidden mechanisms.
  Use the good query templates above. Vary your source domains each search.

PHASE 3 — SHOCK VERIFICATION before finalising each idea:
  For each idea, run the 4 tests silently:
  ① WhatsApp Test: Would a parent forward this in a family chat within 5 seconds? If no → rewrite or replace.
  ② Specificity Audit: Replace every vague word with the exact number/name/date. If you can't → cut the idea.
  ③ Personal Stake Test: Does this affect the viewer's body/home/money/government right now? If no → reframe.
     (Skip for "surprising facts", "unknown facts", Pattern 10/11 — shock alone is enough there.)
  ④ Disbelief Test: Does the title trigger "wait, is that actually REAL?" — not just "that's sad/interesting"?
     The title must name a SPECIFIC person/brand + an EXACT figure/date + an IMPOSSIBLE-SOUNDING OUTCOME.
  Only submit an idea that passes all four. If it fails one → fix it or swap it.

PHASE 4 — PATTERN MIX CHECK:
  At least 2 of your 5 ideas should use Patterns 9, 10, or 11 (Mundane Revelation, Disbelief Frame,
  Corporate Catastrophe). If all 5 use Patterns 1-8 only, you are over-indexing on "dramatic history"
  and under-indexing on the format that actually generates the highest view counts.
  shock_score minimum is 8 — if any idea scores 7 or below, rewrite its title before submitting.

━━━ MANDATORY OUTPUT RULES ━━━
1. Generate ideas ONLY for the areas specified in the "TARGET AREAS" block below — one per area.
2. Each idea must have a PUNCHLINE TITLE — the reversal or impossible reality must be IN the title.
3. Every idea must pass ALL 4 CURIOSITY FILTERS (Surprise/Reversal, Specificity, Personal Stake, Superlative Rule).
4. Every idea must pass ALL 4 SHOCK TESTS (WhatsApp, Specificity Audit, Personal Stake, Disbelief Test).
5. For "history" or "interesting events" area: events from 1900 onwards ONLY.
6. Real, verifiable facts only — no speculation, no "might be true," no hedging language.
7. shock_score must be 8 or above — if it's below 8, rewrite the title before submitting.
   A score of 7 means "educational but not viral." 8+ means "thumb-stopping disbelief."
8. SATURATION SELF-CHECK — before finalising each idea ask: "Is this topic/story ALREADY
   covered extensively on YouTube Shorts?" Famous experiments, pop-history events, and
   the mechanisms listed in KNOWN SATURATED TOPICS above → HIGH. Replace before submitting.
   Only LOW and MEDIUM ideas should appear in your output.
9. AIM FOR PATTERNS 9, 10, or 11 for at least 2 of your 5 ideas — these are the highest-view
   formats from actual viral data: Mundane Revelation (everyday objects), Disbelief Frame
   ("sounds fake but true"), and Corporate Catastrophe (brand + exact dollar amount).
   These consistently outperform the "dry historical fact" format.

Return exactly 5 ideas as JSON:
{
  "ideas": [
    {
      "title": "Punchline title — the reversal or impossible reality must be IN the title, not teased",
      "area": "exact area from the TARGET AREAS list",
      "concept": "EXACT verifiable fact with specific numbers/names/dates. WHY it contradicts the common assumption. HOW it connects to the viewer's daily life or body right now. ONE 'did you know?' opening sentence that lands like a punch.",
      "viral_hook": "ONE sentence with a SPECIFIC shocking number or impossible-sounding real fact. Must pass the WhatsApp Test alone — a stranger should want to share it before finishing the sentence.",
      "pattern_used": "which of the 8 viral structures above this follows",
      "shock_score": "1-10 — honest rating: how many out of 10 people would stop scrolling at this exact title? 7 = educational. 8+ = viral. Below 8 = rewrite the title before submitting.",
      "saturation_self_check": "LOW/MEDIUM/HIGH — one sentence: why this specific story/mechanism is or is not already widely covered on YouTube Shorts"
    }
  ]
}"""


# ── Search variety sampling ───────────────────────────────────────────────────

def _pick_variety_queries(
    target_areas: list[str],
    used_queries: list[str],
    n: int = 8,
    seed: int = 0,
) -> list[str]:
    """
    Return n specific sub-topic search angles sampled from search_variety.json.
    Primary pool is biased toward target_areas; secondary pool adds cross-domain variety.
    Seed by batch_num so each retry batch gets a different but reproducible selection.
    """
    if not _SEARCH_VARIETY:
        return []

    rng = random.Random(seed)
    used_set = {q.lower() for q in (used_queries or [])}

    def _collect(domain_key: str, sub_key: str | None, bucket: list[str]) -> None:
        domain = _SEARCH_VARIETY.get(domain_key, {})
        subs = {sub_key: domain[sub_key]} if (sub_key and sub_key in domain) else domain
        for items in subs.values():
            for t in items:
                if t.lower() not in used_set:
                    bucket.append(t)

    primary: list[str] = []
    for area in target_areas:
        for domain_key, sub_key in _AREA_TO_VARIETY.get(area, []):
            _collect(domain_key, sub_key, primary)

    # Secondary: everything else for cross-domain breadth
    secondary: list[str] = []
    primary_set = set(primary)
    for domain_key, domain in _SEARCH_VARIETY.items():
        for items in domain.values():
            for t in items:
                if t not in primary_set and t.lower() not in used_set:
                    secondary.append(t)

    rng.shuffle(primary)
    rng.shuffle(secondary)

    # Take up to 6 area-relevant, fill remainder from cross-domain
    area_count = min(len(primary), max(n - 2, n // 2 + 1))
    result = primary[:area_count] + secondary[: n - area_count]
    return result[:n]


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
        "Hidden mechanisms — find NEW untapped mechanisms not yet on YouTube",
        "⚠️ DO NOT search for PAPI lights, ship anchors, U-2 chase car, GPS Einstein, "
        "Dead Hand, seatbelt explosive, submarine ballast, elevator governor, car crumple zones, "
        "or any mechanism listed in KNOWN SATURATED TOPICS. Those are done. Find something NEW. "
        "Target the gap between what people ASSUME a mechanism does and what it ACTUALLY does — "
        "but in UNTAPPED domains: medical devices, industrial safety interlocks, agricultural "
        "machinery, legal/financial plumbing, logistics cold-chains, rescue equipment, sports gear. "
        "Queries: 'counterintuitive engineering mechanism medical device how it actually works', "
        "'industrial safety system failure mode surprising how works', "
        "'agricultural machinery hidden mechanism farmers know but public doesn't', "
        "'financial system clearing settlement mechanism how money actually moves between banks', "
        "'cold chain logistics surprising mechanism food safety', "
        "'rescue equipment mechanism surprising how it physically works', "
        "'construction technique hidden load path counterintuitive engineers', "
        "'water treatment process mechanism surprising chemistry public doesn't know', "
        "'satellite ground station operation surprising mechanism', "
        "'sports equipment physics mechanism counterintuitive', "
        "'military equipment mechanism works opposite to what civilians think', "
        "'legal procedure mechanism public completely misunderstands', "
        "'everyday system hidden physical principle nobody explains'",
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
    (
        "Mundane Revelation — everyday objects with hidden mechanisms or purposes",
        "Find the hidden design intention or physical mechanism behind ordinary objects the viewer "
        "has used thousands of times but never understood. This is the Zack D. Films formula — "
        "'How stitches work' (214M views), 'How splinters get unstuck' (180M views), "
        "'Why the McFlurry spoon looks weird' (143M views). Target the mundane, not the exotic. "
        "⚠️ Do NOT use mechanisms already in KNOWN SATURATED TOPICS. "
        "Search queries: "
        "'[everyday object] hidden purpose design explained', "
        "'why does [common item] have [specific feature] real reason', "
        "'[food packaging/clothing/tool] feature purpose most people don't know', "
        "'[object you see daily] designed that way because history', "
        "'engineering design hidden in plain sight [object] how it works', "
        "'[car/kitchen/coin/clothing] feature hidden meaning explained', "
        "site:mentalfloss.com 'everyday object hidden purpose', "
        "site:smithsonianmag.com 'design history surprising reason'",
    ),
    (
        "Disbelief Frame & Corporate Catastrophe — sounds-fake-but-true + brand + exact dollar",
        "Two highest-confirmed viral patterns from actual view-count data. "
        "DISBELIEF FRAME: find facts that sound made-up but are verifiable — the title acknowledges "
        "disbelief and doubles down. 'Sounds Fake But Is 100% True' is the #1 repeated viral structure. "
        "CORPORATE CATASTROPHE: one famous brand, one exact billion-dollar consequence. "
        "'Toyota's $2,300,000,000 Mistake' is the single highest view-count data point in this research. "
        "Search queries: "
        "'[common belief] sounds impossible but is scientifically true verified', "
        "'[historical timeline] fact sounds wrong but is accurate dates', "
        "'[brand] billion dollar mistake decision real history', "
        "'[famous company] invention suppressed sold wrong reason real', "
        "'[major corporation] cover-up proven court documents', "
        "'[brand] product liability lawsuit settlement exact amount', "
        "'corporate decision that backfired exact financial cost real', "
        "site:smithsonianmag.com 'sounds fake but true', "
        "site:mentalfloss.com 'corporate mistake history'",
    ),
]


# ── Search plan loader ────────────────────────────────────────────────────────

def _load_search_plan() -> dict | None:
    """Load precomputed search plan if it exists and is < 6 hours old."""
    f = PRECOMPUTED_DIR / "search_plan.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        generated_at = datetime.fromisoformat(data["generated_at"])
        age_h = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
        if age_h > 6:
            print(f"[Idea Generator] Search plan is {age_h:.1f}h old — falling back to agentic search")
            return None
        if not data.get("batches"):
            return None
        print(f"[Idea Generator] Using precomputed search plan ({age_h:.1f}h old, "
              f"{len(data['batches'])} batches)")
        return data
    except Exception as e:
        print(f"[Idea Generator] Could not load search plan: {e}")
        return None


# ── Prompt builder (shared between agentic and plan-based paths) ──────────────

def _build_search_prompt(
    banned_words: set[str], recent_titles: list[str],
    targeted_rejects: list[dict], iteration: int,
    favorites: list[dict] | None = None,
    used_queries: list[str] | None = None,
    batch_num: int = 0,
    target_areas: list[str] | None = None,
    audience_insights: dict | None = None,
    saturated_titles: list[str] | None = None,
) -> tuple[str, str]:
    """Build user_message for idea generation. Returns (user_message, angle_label)."""

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
        bullets = "\n".join(f"  • {t}" for t in recent_titles[-15:])
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
        for fav in favorites[-10:]:
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

    searched_block = ""
    if used_queries:
        searched_list = "\n".join(f"  • {q}" for q in used_queries[-20:])
        searched_block = (
            f"🔍 ALREADY SEARCHED — do NOT repeat these queries or phrases:\n"
            f"{searched_list}\n"
            f"Reframe your searches with completely different keywords, source types, and angles.\n\n"
        )

    search_avoid_block = ""
    if banned_words:
        avoid_str = ", ".join(sorted(banned_words))
        search_avoid_block = (
            f"🔎 AVOID IN SEARCH QUERIES — these topic words already have many existing ideas, "
            f"so searching for them will only produce more duplicates. Do NOT use any of these "
            f"words as primary keywords in your web or YouTube searches:\n{avoid_str}\n\n"
        )

    areas_for_this_batch = target_areas or AREAS[:5]
    area_list = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(areas_for_this_batch))
    _sole_area = areas_for_this_batch[0] if len(areas_for_this_batch) == 1 else None

    if len(areas_for_this_batch) == 1:
        target_block = (
            f"🎯 TARGET AREA — ALL 5 ideas must be in this single area: **{_sole_area}**\n"
            f"Every idea's 'area' field must be exactly '{_sole_area}'.\n"
            f"Cover 5 COMPLETELY DIFFERENT specific stories, mechanisms, or facts within '{_sole_area}'.\n"
            f"Variety of sub-topics is required — do not repeat the same angle twice.\n\n"
        )
    else:
        target_block = (
            f"🎯 TARGET AREAS — generate EXACTLY one idea per area, in any order:\n"
            f"{area_list}\n"
            f"Each idea's 'area' field must match one of these exactly. "
            f"Do NOT generate two ideas for the same area.\n\n"
        )

    # "unknown facts" and "surprising facts" need their own search guidance — the science-heavy
    # _BATCH_ANGLES make the LLM default to NASA/PubMed/NTSB even for general-interest content.
    if _sole_area == "surprising facts":
        _sf_domains = [
            "animal behaviour & biology", "human body surprising truths", "psychology & behaviour",
            "nature & ecology", "food science & agriculture", "geography & planet Earth",
            "history surprising reversals", "technology & invention dates",
            "economics & money surprises", "medicine surprising findings",
            "physics & light everyday surprises", "law & politics oddities",
        ]
        _sf_rotated = _sf_domains[batch_num % len(_sf_domains):][:4] + _sf_domains[:batch_num % len(_sf_domains)][:2]
        _domain_str = " · ".join(_sf_rotated[:5])
        angle_label = "Surprising facts — shocking truths across all domains"
        angle_block = (
            f"🔎 THIS BATCH — Surprising facts that defy common assumptions (wide domain spread)\n"
            f"Focus domains this batch: {_domain_str}\n\n"
            f"⛔ DO NOT default to NASA/ArXiv/NTSB/PubMed — those skew everything toward deep science.\n"
            f"   This area needs broadly shocking facts, NOT specialist research findings.\n\n"
            f"✅ GOOD SOURCES:\n"
            f"   site:sciencealert.com · site:bigthink.com · site:smithsonianmag.com\n"
            f"   site:mentalfloss.com · site:nationalgeographic.com · site:livescience.com\n"
            f"   site:psychologytoday.com · site:britannica.com · Wikipedia 'counterintuitive'\n\n"
            f"✅ GOOD SEARCH PATTERNS:\n"
            f"   '[common belief] is actually wrong — what really happens'\n"
            f"   '[animal/plant/body part] surprising fact defies expectation'\n"
            f"   '[everyday thing] works completely differently than people think'\n"
            f"   'counterintuitive truth [topic] verified'\n"
            f"   '[food/country/object] fact that sounds fake but is true'\n"
            f"   'site:sciencealert.com [topic] surprising'\n\n"
        )
        variety_block = ""  # skip variety — science-domain subjects pull it off track

    elif _sole_area == "unknown facts":
        _uf_domains = [
            "animals & insects", "food & plants", "geography & maps",
            "language & words", "money & trade", "sports records",
            "famous people & names", "everyday objects", "human body quirks",
            "nature oddities", "historical dates & timelines", "buildings & cities",
            "colour & light", "numbers & mathematics trivia", "law & government oddities",
        ]
        _uf_rotated = _uf_domains[batch_num % len(_uf_domains):][:5] + _uf_domains[:batch_num % len(_uf_domains)][:3]
        _domain_str = " · ".join(_uf_rotated[:6])
        angle_label = "Unknown facts — everyday domains, no science sources"
        angle_block = (
            f"🔎 THIS BATCH — Unknown facts from everyday life (NOT science articles)\n"
            f"Explore these domains this batch: {_domain_str}\n\n"
            f"⛔ DO NOT USE these sources — they produce science/research ideas, not unknown facts:\n"
            f"   NASA · ArXiv · PubMed · NTSB · NEJM · ScienceDaily · declassified CIA archives\n\n"
            f"✅ USE THESE SOURCES INSTEAD:\n"
            f"   site:mentalfloss.com · site:smithsonianmag.com · site:britannica.com\n"
            f"   site:worldatlas.com · Wikipedia 'List of common misconceptions'\n"
            f"   Wikipedia 'Unusual articles' · site:atlasobscura.com\n\n"
            f"✅ GOOD SEARCH PATTERNS:\n"
            f"   '[animal] surprising fact most people don't know'\n"
            f"   '[everyday object] counterintuitive truth'\n"
            f"   '[country/city] unknown fact verified'\n"
            f"   'common misconception [topic] what actually happens'\n"
            f"   '[food/drink] surprising true fact'\n"
            f"   'world record [category] you didn't know'\n"
            f"   'site:mentalfloss.com [subject] fact'\n\n"
        )
        variety_block = ""  # skip variety — those are science-domain subjects
    else:
        angle_label, angle_hints = _BATCH_ANGLES[batch_num % len(_BATCH_ANGLES)]
        angle_block = (
            f"🔎 THIS BATCH — Search for: {angle_label}\n"
            f"{angle_hints}\n"
            f"Use specific, niche queries. Avoid generic 'interesting facts' searches.\n\n"
        )
        variety_block = ""
        if _SEARCH_VARIETY:
            picks = _pick_variety_queries(
                target_areas=areas_for_this_batch,
                used_queries=used_queries or [],
                n=8,
                seed=batch_num,
            )
            if picks:
                bullet_list = "\n".join(f"  • {p}" for p in picks)
                variety_block = (
                    "🎲 SPECIFIC SEARCH ANGLES FOR THIS BATCH:\n"
                    "Use these as SEARCH SUBJECTS — not final topics. Build queries around the specific "
                    "entity or mechanism, not the broad category:\n"
                    f"{bullet_list}\n"
                    "  ✓ SPECIFIC: 'mantis shrimp 16-color vision counterintuitive mechanism'\n"
                    "  ✗ GENERIC: 'shocking animal facts' (finds only saturated content)\n\n"
                )

    insights_block = ""
    if audience_insights:
        top        = audience_insights.get("top_areas", [])[:5]
        angles     = audience_insights.get("trending_angles", [])[:5]
        summary    = audience_insights.get("audience_insights", "")
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

    saturated_block = ""
    if saturated_titles:
        bullets = "\n".join(f"  ✗ {t}" for t in saturated_titles)
        saturated_block = (
            f"🔴 PREVIOUSLY REJECTED (YouTube already saturated with these topics):\n"
            f"{bullets}\n"
            f"Do NOT generate these ideas or close variants — they were already checked and rejected.\n\n"
        )

    user_message = (
        f"{insights_block}"
        f"{favorites_block}"
        f"{banned_block}"
        f"{recent_block}"
        f"{reject_block}"
        f"{saturated_block}"
        f"{searched_block}"
        f"{search_avoid_block}"
        f"{target_block}"
        f"{angle_block}"
        f"{variety_block}"
        f"⛔ SATURATION REMINDER — the following are ALREADY widely covered on YouTube Shorts. "
        f"Do NOT generate these or close variants:\n"
        f"Napoleon/rabbits · Emu War · CIA cat · Petrov/Arkhipov · Cadaver Synod · "
        f"Tsutomu Yamaguchi · George Dantzig · Desmond Doss · Grigori Perelman · "
        f"Milgram · Stanford Prison · Rosenhan · Bystander effect · Marshmallow test · "
        f"PAPI lights · Ship anchor chain · U-2 chase car · GPS Einstein · Dead Hand · "
        f"GravityLight · Runway numbers · Seatbelt explosive · Submarine ballast/fish · "
        f"Viganella mirror · Carrier wire · Crumple zone · Elevator governor · "
        f"Göbekli Tepe · Antikythera · Bronze Age collapse · Toraja rituals · "
        f"Famadihana · Bullet ant gloves · Hikikomori · Karoshi · Naghol diving\n\n"
        f"PHASE 1 — ONE YouTube search to extract viral FORMAT only (title hook structure).\n"
        f"PHASE 2 — Specific web searches for REAL stories. Aim for at least 2 ideas using Patterns\n"
        f"  9 (Mundane Revelation — everyday objects), 10 (Disbelief Frame — sounds-fake-but-true),\n"
        f"  or 11 (Corporate Catastrophe — brand + exact dollar). These are the highest-view formats.\n"
        f"PHASE 3 — Before submitting, silently run ALL 4 SHOCK TESTS on every idea:\n"
        f"  ① WhatsApp Test: would a parent forward this in 5 seconds?\n"
        f"  ② Specificity Audit: every vague word → exact number/name/date\n"
        f"  ③ Personal Stake Test: does this affect the viewer's body/home/money RIGHT NOW?\n"
        f"     (Skip for 'surprising facts', 'unknown facts', Patterns 10/11)\n"
        f"  ④ Disbelief Test: does the title trigger 'wait, is that actually REAL?' — name + number + impossible outcome\n"
        f"PHASE 4 — Pattern Mix Check: verify at least 2 of 5 ideas use Patterns 9/10/11.\n\n"
        f"⚠️ Use the viral structures from the system prompt. "
        f"Dry educational facts are NOT acceptable. Each title must have the punchline IN IT.\n"
        f"shock_score must be 8+ — a 7 means 'educational but not viral'. Rewrite the title if below 8.\n"
        f"'WORST' outperforms 'Most Dangerous' or 'Scariest' — use it when applicable.\n"
        f"Each idea's saturation_self_check must be LOW or MEDIUM — replace HIGH ideas before submitting."
    )

    return user_message, angle_label


def _parse_ideas(raw: str, new_queries: list[str]) -> tuple[list[dict], list[str]]:
    """Parse LLM response into ideas list, filtering self-flagged HIGH saturation."""
    if isinstance(raw, list):
        raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        parsed    = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        ideas_raw = parsed.get("ideas", [])
        if isinstance(ideas_raw, dict):
            ideas_raw = list(ideas_raw.values())
        clean = []
        for idea in ideas_raw:
            if not isinstance(idea, dict):
                continue
            self_check = idea.get("saturation_self_check", "").strip().upper()
            if self_check.startswith("HIGH"):
                print(f"[Idea Generator] ✗ SELF-FLAGGED HIGH SAT: '{idea.get('title', '')}'")
            else:
                clean.append(idea)
        return clean, new_queries
    except Exception:
        return ([{"title": f"Idea {i+1}", "area": "science",
                  "concept": line.strip(), "viral_hook": "", "pattern_used": ""}
                 for i, line in enumerate(raw.split("\n")) if line.strip()][:3],
                new_queries)


# ── Plan-based idea generation (Anthropic synthesis only, no agentic loop) ────

def _call_llm_from_plan(
    llm, web_search,
    plan_batch: dict,
    banned_words: set[str], recent_titles: list[str],
    targeted_rejects: list[dict], iteration: int,
    favorites: list[dict] | None = None,
    used_queries: list[str] | None = None,
    batch_num: int = 0,
    target_areas: list[str] | None = None,
    audience_insights: dict | None = None,
    saturated_titles: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Execute pre-planned searches in parallel, then send everything to Anthropic
    in ONE synthesis call. No agentic loop — much faster and more predictable.
    """
    import concurrent.futures

    queries_by_area = plan_batch.get("queries_by_area", {})
    youtube_queries = plan_batch.get("youtube_queries", [])
    areas_to_cover  = target_areas or AREAS[:5]

    # Build the consolidated search task list — web only.
    # YouTube queries from the plan are topic-specific (e.g. "anchoring effect shorts")
    # which finds *existing* videos → synthesis LLM copies those topics → all saturated.
    # Viral format patterns are already embedded in SYSTEM_PROMPT; no YouTube step needed.
    tasks: list[tuple[str, str]] = []   # (kind, query)
    for area in areas_to_cover:
        area_qs = queries_by_area.get(area, [])
        if not area_qs:
            # Try fuzzy match
            for k, v in queries_by_area.items():
                if area.lower() in k.lower() or k.lower() in area.lower():
                    area_qs = v
                    break
        for q in area_qs[:4]:   # was 3 — more web context per area
            tasks.append(("web", q))

    angle_label = plan_batch.get("angle", "precomputed")
    print(f"\n[Idea Generator] Plan-based call (batch {batch_num + 1}, "
          f"focus={angle_label[:50]}, {len(tasks)} searches, "
          f"history={len(recent_titles)} titles)...")

    # Execute all searches in parallel (up to 4 concurrent)
    gathered: list[str] = []
    new_queries: list[str] = []

    def _clean_query(q: str) -> str:
        """Remove language-specific prefixes injected by older precomputed plans."""
        import re
        return re.sub(r'\btelugu\b\s*', '', q, flags=re.IGNORECASE).strip()

    def _run(kind: str, q: str) -> str:
        q = _clean_query(q)
        new_queries.append(q)
        if kind == "youtube":
            print(f"[Idea Generator] YouTube (plan): {q[:65]}")
            return f"**YouTube search:** {q}\n{search_youtube_shorts.invoke({'query': q})}"
        print(f"[Idea Generator] Web (plan): {q[:65]}")
        return f"**Web search [{kind}]:** {q}\n{web_search.invoke({'query': q})}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_run, kind, q): (kind, q) for kind, q in tasks}
        for fut in concurrent.futures.as_completed(futures):
            try:
                gathered.append(fut.result())
            except Exception as e:
                print(f"[Idea Generator] Search error: {e}")

    search_context = "\n\n---\n\n".join(gathered)

    user_message, _ = _build_search_prompt(
        banned_words, recent_titles, targeted_rejects, iteration,
        favorites=favorites, used_queries=used_queries,
        batch_num=batch_num, target_areas=target_areas,
        audience_insights=audience_insights,
        saturated_titles=saturated_titles,
    )

    # Replace the PHASE 1/2/3 instructions with a direct synthesis directive.
    # Key framing: web results are SOURCE MATERIAL — find the NICHE SPECIFIC detail,
    # not the headline topic (which is usually already on YouTube).
    synthesis_suffix = (
        "\n\n══════════════════════════════════════════════════════\n"
        "RESEARCH IS PRE-GATHERED BELOW — do NOT call any search tools.\n\n"
        "HOW TO USE THIS RESEARCH:\n"
        "• Each web result contains a NICHE SPECIFIC FACT buried inside a broader topic.\n"
        "• Build your idea around that NICHE DETAIL — NOT the broad topic headline.\n"
        "  Example: a result about 'how vaccines work' might mention a specific protein-folding\n"
        "  mechanism nobody knows about. The IDEA is about THAT mechanism, not 'vaccines'.\n"
        "• SKIP any result whose main topic is already on the KNOWN SATURATED TOPICS list\n"
        "  (Dunning-Kruger, anchoring effect, Milgram, Stanford Prison, Marshmallow test,\n"
        "  Tacoma Narrows, Dunning Kruger, placebo effect, etc.).\n"
        "  If a result is about one of those — read the SUPPORTING DETAILS for a less-known angle.\n"
        "• The viral FORMAT patterns are already in your system prompt — apply them to the\n"
        "  novel TOPICS you extract from the research below.\n"
        "══════════════════════════════════════════════════════\n\n"
        + search_context
    )
    # Strip the PHASE instructions from the end of user_message and add synthesis directive
    phase_marker = "PHASE 1 —"
    if phase_marker in user_message:
        user_message = user_message[:user_message.index(phase_marker)] + synthesis_suffix
    else:
        user_message = user_message + synthesis_suffix

    print(f"[Idea Generator] Synthesizing {len(gathered)} results → Anthropic (single call)...")
    try:
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)])
        return _parse_ideas(response.content, new_queries)
    except Exception as e:
        print(f"[Idea Generator] Synthesis call failed: {e}")
        return [], new_queries


# ── Agentic LLM call (fallback when no precomputed plan) ─────────────────────

def _call_llm(llm_with_tools, web_search,
              banned_words: set[str], recent_titles: list[str],
              targeted_rejects: list[dict], iteration: int,
              favorites: list[dict] | None = None,
              used_queries: list[str] | None = None,
              batch_num: int = 0,
              target_areas: list[str] | None = None,
              audience_insights: dict | None = None,
              saturated_titles: list[str] | None = None) -> tuple[list[dict], list[str]]:

    user_message, angle_label = _build_search_prompt(
        banned_words, recent_titles, targeted_rejects, iteration,
        favorites=favorites, used_queries=used_queries,
        batch_num=batch_num, target_areas=target_areas,
        audience_insights=audience_insights,
        saturated_titles=saturated_titles,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    print(f"\n[Idea Generator] Agentic call (batch {batch_num + 1}/{iteration + 1}, "
          f"focus={angle_label}, "
          f"banned={len(banned_words)} words, "
          f"history={len(recent_titles)} titles)...")

    new_queries: list[str] = []

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    _MAX_SEARCH_ROUNDS = 6
    _search_rounds = 0
    while response.tool_calls and _search_rounds < _MAX_SEARCH_ROUNDS:
        _search_rounds += 1
        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
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
        try:
            response = llm_with_tools.invoke(messages)
            messages.append(response)
        except Exception as e:
            print(f"[Idea Generator] ⚠ LLM call failed ({type(e).__name__}: {e}) — stopping early")
            break
    if _search_rounds >= _MAX_SEARCH_ROUNDS:
        print(f"[Idea Generator] ⚠ search round cap ({_MAX_SEARCH_ROUNDS}) reached")
    print(f"[Idea Generator] Synthesizing {_search_rounds} search rounds...")

    return _parse_ideas(response.content, new_queries)


# ── Node 1: Generate ideas ─────────────────────────────────────────────────────

def idea_generator_agent_node(state: dict) -> dict:
    llm_synth  = get_sonnet_llm(temperature=0.9)
    llm_agent  = get_sonnet_llm(temperature=0.9)
    web_search     = get_openserp_search_tool(max_results=5)
    llm_with_tools = llm_agent.bind_tools([search_youtube_shorts, web_search])

    iteration  = state.get("idea_gen_iteration", 0)
    prev_ideas = state.get("generated_ideas", [])

    history          = _load_history()
    blacklist        = _load_blacklist()
    favorites        = _load_favorites()
    audience_data    = load_audience_insights()
    prev_dicts       = [i for i in prev_ideas if isinstance(i, dict)]
    saturated_log    = _load_saturated()

    # all_used_ideas: history + blacklist + session ideas, deduplicated by title
    # Used for semantic dedup (title + concept compared) so same-study ideas are caught
    seen_titles: set = set()
    all_used_ideas: list[dict] = []
    for idea in prev_dicts + history + blacklist:
        t = idea.get("title", "")
        if t not in seen_titles:
            seen_titles.add(t)
            all_used_ideas.append(idea)

    # Also include saturated-rejected ideas in keyword dedup so near-duplicates are caught
    # before generation, not just after the saturation filter
    for s in saturated_log:
        t = s.get("title", "")
        if t and t not in seen_titles:
            seen_titles.add(t)
            all_used_ideas.append({"title": t, "area": s.get("area", ""), "concept": s.get("concept", "")})

    # all_used_titles: title strings only, for the LLM prompt
    all_used_titles = [i.get("title", "") for i in all_used_ideas]

    # Saturated titles for explicit prompt block (most recent 30)
    saturated_titles = [s.get("title", "") for s in saturated_log[-30:] if s.get("title")]

    # Adaptive threshold: need more occurrences to declare a topic exhausted
    threshold    = max(3, len(all_used_ideas) // 15)
    banned_words = _overused_words(all_used_ideas, min_count=threshold)

    # Pick target areas — use caller-supplied list if present (single-domain button)
    forced_areas = state.get("forced_areas")
    if forced_areas:
        target_areas = [a for a in forced_areas if a in AREAS] or forced_areas
        print(f"[Idea Generator] Forced areas: {target_areas}")
    else:
        target_areas = _pick_target_areas(all_used_ideas, n=TARGET_IDEAS)

    # When user forces a single area, allow multiple ideas from it (skip area-dedup)
    single_area_mode = bool(forced_areas and len(target_areas) == 1)

    from collections import Counter as _Counter
    area_counts = _Counter(d.get("area", "").lower() for d in all_used_ideas if d.get("area"))
    print(f"[Idea Generator] History: {len(history)} | Favorites: {len(favorites)} | "
          f"threshold={threshold} | Banned ({len(banned_words)}): {', '.join(sorted(banned_words))}")
    print(f"[Idea Generator] Target areas: {target_areas}")
    print(f"[Idea Generator] Area counts: { {a: area_counts.get(a, 0) for a in AREAS} }")

    search_plan  = _load_search_plan()
    use_plan     = bool(search_plan and search_plan.get("batches"))
    plan_batches = search_plan["batches"] if use_plan else []
    if use_plan:
        print(f"[Idea Generator] Using precomputed search plan — Anthropic does synthesis only")
    else:
        print(f"[Idea Generator] No precomputed plan — using agentic search (fallback)")

    # ── Retry loop: generate → check → replace duplicates ────────────────────
    accepted:         list[dict] = []
    accepted_areas:   set[str]   = set()
    targeted_rejects: list[dict] = []
    session_queries:  list[str]  = []

    for attempt in range(MAX_RETRIES):
        if len(accepted) >= TARGET_IDEAS:
            break

        remaining_areas = [a for a in target_areas if a not in accepted_areas]

        if use_plan:
            plan_batch = plan_batches[attempt % len(plan_batches)]
            candidates, new_queries = _call_llm_from_plan(
                llm_synth, web_search, plan_batch,
                banned_words, all_used_titles,
                targeted_rejects, iteration,
                favorites=favorites,
                used_queries=session_queries,
                batch_num=attempt,
                target_areas=remaining_areas or target_areas,
                audience_insights=audience_data,
                saturated_titles=saturated_titles,
            )
        else:
            candidates, new_queries = _call_llm(
                llm_with_tools, web_search,
                banned_words, all_used_titles,
                targeted_rejects, iteration,
                favorites=favorites,
                used_queries=session_queries,
                batch_num=attempt,
                target_areas=remaining_areas or target_areas,
                audience_insights=audience_data,
                saturated_titles=saturated_titles,
            )
        session_queries.extend(new_queries)

        for idea in candidates:
            if len(accepted) >= TARGET_IDEAS:
                break
            if not isinstance(idea, dict):
                continue
            title = idea.get("title", "")
            area  = idea.get("area", "").lower().strip()

            # Code-level area dedup — skip when all ideas intentionally share one area
            if not single_area_mode and area in accepted_areas:
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
