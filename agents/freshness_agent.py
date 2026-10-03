import json
import concurrent.futures
from ollama_client import get_haiku_llm
from langchain_core.messages import HumanMessage
from tools import search_youtube_shorts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _llm():
    return get_haiku_llm(temperature=0.1)


def _parse_results(result_str: str) -> tuple[int, list[str]]:
    """Count videos returned by search_youtube_shorts and extract their titles."""
    videos = []
    for line in str(result_str).strip().split("\n"):
        line = line.strip()
        if line.startswith("- "):
            videos.append(line[2:])
    return len(videos), videos


def _generate_queries(title: str, concept: str, n: int = 5) -> tuple[str, list[str]]:
    """
    Ask the LLM to understand the concept behind the idea and generate
    n diverse search queries covering different angles of the same topic.
    """
    prompt = f"""You are a YouTube research analyst. Given a YouTube Shorts idea, extract the core concept and generate {n} distinct search queries to measure how saturated that topic is on YouTube.

IDEA TITLE: {title}
IDEA CONCEPT: {concept}

Think about what this idea is REALLY about underneath the specific framing, then generate queries that cover:
1. The exact specific angle of this idea (most specific)
2. The broader topic / category it belongs to
3. A "did you know / interesting facts about X" framing viewers would search
4. A "how X works / why X happens" framing
5. An alternative keyword variation someone might use

Rules:
- Queries should be 3-7 words each
- Think like a viewer, not a content creator
- Do NOT just copy the idea title — explore the concept space
- Queries should genuinely test if this concept is covered on YouTube

Return JSON only, no markdown:
{{"core_concept": "one sentence describing what this idea is fundamentally about", "queries": ["q1", "q2", "q3", "q4", "q5"]}}"""

    try:
        resp = _llm().invoke([HumanMessage(content=prompt)])
        raw = resp.content
        if isinstance(raw, list):
            raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
        if "```" in raw:
            raw = raw.split("```json", 1)[-1].split("```")[0] if "```json" in raw else raw.split("```")[1].split("```")[0]
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        core   = parsed.get("core_concept", concept[:120])
        q_list = [q for q in parsed.get("queries", []) if q][:n]
        if not q_list:
            q_list = [title]
        return core, q_list
    except Exception as e:
        print(f"[Saturation] query generation failed: {e}")
        return concept[:120], [title]


def _run_searches(queries: list[str]) -> list[dict]:
    """Run searches in parallel and return per-query result dicts."""
    def _one(q: str) -> dict:
        print(f"[Saturation] Search: '{q}'")
        try:
            raw = search_youtube_shorts.invoke({"query": q})
            count, videos = _parse_results(raw)
            print(f"[Saturation]   → {count} result(s)")
            return {"query": q, "results_found": count, "videos": videos}
        except Exception as e:
            print(f"[Saturation]   → error: {e}")
            return {"query": q, "results_found": 0, "videos": [], "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as ex:
        return list(ex.map(_one, queries))


def _score_local(search_data: list[dict], total: int) -> dict | None:
    """
    Skip LLM scoring for unambiguous cases based on result counts alone.
    Returns a scoring dict if the result is clear-cut, None if LLM judgment is needed.
    """
    n = len(search_data)
    max_possible = n * 5
    # Every search returned 0 or 1 result — clearly not saturated
    if total <= n:
        return {
            "score": 1, "label": "LOW",
            "reasoning": f"Only {total} result(s) across {n} searches — topic is largely uncovered.",
            "directly_competing_videos": [],
            "content_gap": "This specific angle is largely absent on YouTube.",
        }
    # Near-maximum results — clearly saturated (≥90% of max)
    if total >= max_possible * 0.9:
        all_vids = [v for s in search_data for v in s.get("videos", [])[:2]]
        return {
            "score": 3, "label": "HIGH",
            "reasoning": f"{total}/{max_possible} results found — topic is heavily covered on YouTube.",
            "directly_competing_videos": all_vids[:6],
            "content_gap": "",
        }
    return None  # ambiguous — use LLM


def _score(title: str, core_concept: str, search_data: list[dict], total: int) -> dict:
    """
    LLM scores saturation based on aggregated search evidence.
    Must justify using actual numbers and video titles found.
    """
    max_possible = len(search_data) * 5  # Tavily caps at 5 per search

    evidence = []
    for i, s in enumerate(search_data, 1):
        evidence.append(f"Search {i}: \"{s['query']}\" → {s['results_found']} result(s)")
        for v in s["videos"][:4]:
            evidence.append(f"    • {v}")

    prompt = f"""You are a strict YouTube saturation analyst. Assess how saturated this topic is based ONLY on the search evidence below.

IDEA: {title}
CORE CONCEPT: {core_concept}

━━━ SEARCH EVIDENCE ━━━
{chr(10).join(evidence)}

━━━ SUMMARY ━━━
Total searches run: {len(search_data)}
Total results found: {total} out of a maximum possible {max_possible}

━━━ SCORING RULES ━━━
HIGH  (score=3): Topic is well-covered. Total results ≥ {max(10, max_possible//2)} OR multiple searches each returned 4-5 results, especially if video titles are directly on this topic.
MEDIUM (score=2): Some coverage exists. Total results {max(4, max_possible//4)}–{max(9, max_possible//2)-1} OR fewer results but titles are directly competitive.
LOW   (score=1): Very little coverage. Total results ≤ {max(3, max_possible//4)-1} AND no search returned more than 1-2 results on-topic.

IMPORTANT: Do NOT default to LOW. If searches found real videos on this topic, be honest about it.
Also consider: if results are tangential / off-topic, that lowers the effective saturation vs same count of directly competing videos.

Return JSON only:
{{
  "score": 1 or 2 or 3,
  "label": "LOW" or "MEDIUM" or "HIGH",
  "reasoning": "2-3 sentences citing specific search numbers and video titles from the evidence above. State why you chose this score over the others.",
  "directly_competing_videos": ["title1", "title2"],
  "content_gap": "specific angle NOT found in results that this idea could own"
}}"""

    try:
        resp = _llm().invoke([HumanMessage(content=prompt)])
        raw  = resp.content
        if isinstance(raw, list):
            raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
        if "```" in raw:
            raw = raw.split("```json", 1)[-1].split("```")[0] if "```json" in raw else raw.split("```")[1].split("```")[0]
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        return parsed
    except Exception as e:
        print(f"[Saturation] scoring LLM failed: {e}")
        # Fallback via raw count
        if total <= 3:
            s, l = 1, "LOW"
        elif total <= 9:
            s, l = 2, "MEDIUM"
        else:
            s, l = 3, "HIGH"
        return {
            "score": s, "label": l,
            "reasoning": f"Fallback: {total} results across {len(search_data)} searches.",
            "directly_competing_videos": [],
            "content_gap": "",
        }


def _build_report(title, core_concept, search_data, scoring) -> dict:
    total = sum(s["results_found"] for s in search_data)
    return {
        "title":           title,
        "core_concept":    core_concept,
        "searches":        search_data,
        "total_searches":  len(search_data),
        "total_results":   total,
        "score_num":       scoring.get("score", 1),
        "score":           scoring.get("label", "LOW"),
        "reasoning":       scoring.get("reasoning", ""),
        "competitors":     scoring.get("directly_competing_videos", []),
        "content_gap":     scoring.get("content_gap", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# User-facing full check (5 searches, detailed report)
# ══════════════════════════════════════════════════════════════════════════════

def freshness_check_agent_node(state: dict) -> dict:
    ideas = state.get("research_ideas", [])
    if not ideas:
        return {
            "freshness_approved": False,
            "best_idea":          None,
            "rejection_reason":   "No ideas received.",
            "saturation_report":  None,
        }

    idea    = ideas[0]
    title   = idea.get("title", "")
    concept = idea.get("concept", title)

    print(f"\n[Saturation] Analysing: '{title}'")

    # Step 1 — understand concept, generate 5 search angles
    print("[Saturation] Step 1: extracting concept + generating search queries...")
    core_concept, queries = _generate_queries(title, concept, n=5)
    print(f"[Saturation] Core concept: {core_concept}")
    print(f"[Saturation] Queries: {queries}")

    # Step 2 — run all searches
    print("[Saturation] Step 2: running searches...")
    search_data   = _run_searches(queries)
    total_results = sum(s["results_found"] for s in search_data)
    print(f"[Saturation] Total results across {len(search_data)} searches: {total_results}")

    # Step 3 — score (local fast-path for clear-cut cases, LLM for ambiguous)
    scoring = _score_local(search_data, total_results)
    if scoring:
        print(f"[Saturation] Step 3: local threshold → {scoring['label']} (skipped LLM)")
    else:
        print("[Saturation] Step 3: LLM scoring...")
        scoring = _score(title, core_concept, search_data, total_results)
    label   = scoring.get("label", "LOW")
    score_n = scoring.get("score", 1)

    print(f"[Saturation] ► {label} (score={score_n}) — {scoring.get('reasoning','')[:80]}")

    approved  = score_n <= 2          # LOW and MEDIUM pass
    report    = _build_report(title, core_concept, search_data, scoring)
    best_idea = {
        "title":    title,
        "concept":  concept,
        "why_best": scoring.get("reasoning", ""),
    } if approved else None

    if not approved:
        from agents.idea_generator_agent import save_saturated_idea
        save_saturated_idea(
            {"title": title, "area": "", "concept": concept},
            reason=scoring.get("reasoning", ""),
        )

    return {
        "freshness_approved": approved,
        "best_idea":          best_idea,
        "rejection_reason":   scoring.get("reasoning", "") if not approved else "",
        "saturation_report":  report,
        "messages":           [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Bulk pre-filter (3 searches per idea — leaner, runs before ideas shown to user)
# ══════════════════════════════════════════════════════════════════════════════

def check_idea_saturation(idea: dict) -> dict:
    """
    Check saturation for a single idea (2 parallel searches, no LLM).
    Returns the idea dict with saturation_score / saturation_reason / saturation_report added.
    """
    labels  = {1: "🟢 LOW", 2: "🟡 MEDIUM", 3: "🔴 HIGH"}
    title   = idea.get("title", "")
    queries = _fast_queries(title)

    def _search(q: str) -> dict:
        try:
            raw = search_youtube_shorts.invoke({"query": q})
            count, videos = _parse_results(raw)
            return {"query": q, "results_found": count, "videos": videos}
        except Exception as e:
            return {"query": q, "results_found": 0, "videos": [], "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as ex:
        search_data = list(ex.map(_search, queries))

    total   = sum(s["results_found"] for s in search_data)
    local_s = _score_local(search_data, total)
    if local_s:
        score_n, score_l = local_s["score"], local_s["label"]
    elif total <= 4:
        score_n, score_l = 1, "LOW"
    elif total <= 8:
        score_n, score_l = 2, "MEDIUM"
    else:
        score_n, score_l = 3, "HIGH"

    detail = ", ".join(f'"{s["query"]}"={s["results_found"]}' for s in search_data)
    reason = f"{total} results across {len(search_data)} searches ({detail})"
    print(f"[Sat Check] {labels[score_n]} '{title}' — {reason}")

    scored = dict(
        idea,
        saturation_score=score_n,
        saturation_reason=reason,
        saturation_report=_build_report(title, title, search_data, {
            "score": score_n, "label": score_l,
            "reasoning": reason,
            "directly_competing_videos": [],
            "content_gap": "",
        }),
    )
    if score_n > 2:
        from agents.idea_generator_agent import save_saturated_idea
        save_saturated_idea(idea, reason=reason)
    return scored


def _fast_queries(title: str) -> list[str]:
    """
    Derive 2 topic-focused queries from an idea title WITHOUT an LLM call.
    Extracts CONTENT KEYWORDS only — strips framing words like 'actually', 'your', 'the truth'.
    Using the full title as a query causes DuckDuckGo to keyword-match on domain words
    ("painkiller", "empathy", "Harvard") and return 8-10 unrelated videos → false HIGH.
    Topic words like "painkiller empathy reduces 40%" give specific saturation evidence.
    """
    import re
    _FRAMING = {
        "the","a","an","your","our","you","i","we","they","he","she","it","its","my","their",
        "this","that","these","those","why","how","what","when","where","who","which",
        "dont","doesnt","actually","really","never","always","already","still",
        "every","all","only","real","true","false","just","even","now","then",
        "there","here","very","too","so","up","out","not","no","also","back",
        "per","via","from","with","and","but","or","nor","for","yet","both","each",
        "most","more","many","much","some","any","few","can","will","may","might",
        "could","would","should","was","were","been","being","have","has","had",
        "did","does","do","be","am","is","are","get","got","make","made","found",
        "says","said","shows","shown","fact","secret","hidden","truth","about",
        "known","told","left","right","wrong","good","bad","big","small",
        "new","old","long","short","high","low","first","last","ever","once",
    }

    def _words(text: str, n: int = 5) -> list[str]:
        tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
        return [w for w in tokens if len(w) >= 4 and w not in _FRAMING][:n]

    # Titles have structure "CLAIM — PUNCHLINE"; split to get both halves
    parts = re.split(r"\s*[—–]\s*", title)
    left  = _words(parts[0]) if parts else []
    right = _words(parts[1]) if len(parts) > 1 else []

    # q1: up to 5 content words from the core claim (left of em-dash)
    q1 = " ".join(left[:5])
    # q2: bridge last 2 words of claim + first 3 of punchline for the specific angle
    bridge = left[2:4] + right[:3]
    q2 = " ".join(bridge[:5]) if bridge else " ".join(_words(title, n=8)[3:8])

    seen: set[str] = set()
    out: list[str] = []
    for q in [q1, q2]:
        q = q.strip()
        if q and len(q) > 4 and q not in seen:
            seen.add(q)
            out.append(q)
    return out[:2] or [" ".join(_words(title, n=5))]


def bulk_saturation_filter(ideas: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Quick saturation pre-screen for a batch of ideas.
    Runs all searches in parallel (no LLM query generation).
    Returns (fresh_ideas, all_scored).
    """
    if not ideas:
        return [], []

    labels = {1: "🟢 LOW", 2: "🟡 MEDIUM", 3: "🔴 HIGH"}

    # Build a flat list of (idea_idx, query) pairs — no LLM needed
    idea_queries: list[tuple[int, str]] = []
    for i, idea in enumerate(ideas):
        for q in _fast_queries(idea.get("title", "")):
            idea_queries.append((i, q))

    # Run ALL searches across ALL ideas in parallel
    idea_search_data: dict[int, list[dict]] = {i: [] for i in range(len(ideas))}

    def _search(idx_q: tuple[int, str]) -> tuple[int, dict]:
        i, q = idx_q
        try:
            raw = search_youtube_shorts.invoke({"query": q})
            count, videos = _parse_results(raw)
            return i, {"query": q, "results_found": count, "videos": videos}
        except Exception as e:
            return i, {"query": q, "results_found": 0, "videos": [], "error": str(e)}

    workers = min(8, len(idea_queries)) if idea_queries else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for idea_idx, result in ex.map(_search, idea_queries):
            idea_search_data[idea_idx].append(result)

    # Score each idea using thresholds only (no LLM)
    all_scored: list[dict] = []
    filtered:   list[dict] = []

    for i, idea in enumerate(ideas):
        title       = idea.get("title", "")
        search_data = idea_search_data[i]
        total       = sum(s["results_found"] for s in search_data)

        print(f"\n[Bulk Filter] '{title}'")
        local_s = _score_local(search_data, total)
        if local_s:
            score_n, score_l = local_s["score"], local_s["label"]
        elif total <= 4:
            score_n, score_l = 1, "LOW"
        elif total <= 8:
            score_n, score_l = 2, "MEDIUM"
        else:
            score_n, score_l = 3, "HIGH"

        detail = ", ".join(f'"{s["query"]}"={s["results_found"]}' for s in search_data)
        reason = f"{total} results across {len(search_data)} searches ({detail})"
        print(f"[Bulk Filter] {labels[score_n]} — {reason}")

        scored = dict(
            idea,
            saturation_score=score_n,
            saturation_reason=reason,
            saturation_report=_build_report(title, title, search_data, {
                "score": score_n, "label": score_l,
                "reasoning": reason,
                "directly_competing_videos": [],
                "content_gap": "",
            }),
        )
        all_scored.append(scored)
        if score_n <= 2:
            filtered.append(scored)
        else:
            print(f"[Bulk Filter] ✗ REMOVED (HIGH saturation) — saving to history")
            from agents.idea_generator_agent import save_saturated_idea
            save_saturated_idea(idea, reason=reason)

    print(f"\n[Bulk Filter] {len(filtered)}/{len(ideas)} ideas passed\n")
    return filtered, all_scored
