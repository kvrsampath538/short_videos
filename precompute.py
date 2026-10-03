#!/usr/bin/env python3
"""
Offline pre-computation using local Ollama (qwen3.5:9b).

Generates the search plan that idea_generator_agent uses when the user
clicks "Generate Ideas" — so Anthropic only needs to do the final synthesis,
not search planning.

Run automatically on startup (background thread) or via:
    python precompute.py

Cron equivalent (add to crontab for on-boot runs):
    @reboot cd /Users/sampath/Downloads/multiagent && python precompute.py >> logs/precompute.log 2>&1
"""
import json
import os
import sys
import time
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap env & path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

for _line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, v = _line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# ── Paths ─────────────────────────────────────────────────────────────────────
PRECOMPUTED_DIR   = _ROOT / "precomputed"
SEARCH_PLAN_FILE  = PRECOMPUTED_DIR / "search_plan.json"
CONTEXT_PACK_FILE = PRECOMPUTED_DIR / "context_pack.json"
STALENESS_HOURS   = 6   # regenerate if files are older than this


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_stale(path: Path, hours: float = STALENESS_HOURS) -> bool:
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        generated_at = datetime.fromisoformat(data["generated_at"])
        age = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
        return age > hours
    except Exception:
        return True


def _extract_text(response) -> str:
    """Extract text content from a ChatOllama / ChatAnthropic response."""
    raw = response.content if hasattr(response, "content") else response
    if isinstance(raw, list):
        # Content blocks — pick text blocks, skip thinking blocks
        parts = []
        for b in raw:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif "text" in b:
                    parts.append(b["text"])
        raw = " ".join(parts)
    return str(raw).strip()


def _parse_json_response(raw: str) -> dict:
    if not raw:
        raise ValueError("Empty response from model")
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {raw[:200]!r}")
    return json.loads(raw[start:end])


# ── Step 1: Build context pack ───────────────────────────────────────────────

def build_context_pack() -> dict:
    """
    Process history + blacklist → compact context pack.
    Uses Ollama to identify patterns and gaps in existing ideas.
    """
    from langchain_core.messages import HumanMessage
    from ollama_client import get_ollama_llm
    from agents.idea_generator_agent import (
        _load_history, _load_blacklist, _load_favorites,
        _overused_words, _pick_target_areas, AREAS,
    )

    history   = _load_history()
    blacklist = _load_blacklist()
    favorites = _load_favorites()

    all_used  = history + blacklist
    threshold = max(3, len(all_used) // 15)
    banned    = _overused_words(all_used, min_count=threshold)
    targets   = _pick_target_areas(all_used, n=5)

    recent_titles = [i.get("title", "") for i in all_used[-50:] if i.get("title")]

    # Ask Ollama to identify patterns and gaps
    llm = get_ollama_llm(temperature=0.2)
    prompt = (
        "Analyze these recent YouTube Shorts idea titles for a Telugu educational channel. "
        "Identify patterns and gaps.\n\n"
        "RECENT TITLES:\n"
        + "\n".join(f"• {t}" for t in recent_titles)
        + "\n\nReturn ONLY valid JSON:\n"
        '{"saturated_patterns": ["pattern1"], '
        '"gap_areas": ["area1"], '
        '"strong_title_pattern": "description of what makes titles strong here"}'
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        analysis = _parse_json_response(_extract_text(resp))
    except Exception as e:
        print(f"[Precompute] History analysis skipped: {e}")
        analysis = {}

    context_pack = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "target_areas":      targets,
        "banned_words":      sorted(banned),
        "recent_titles":     recent_titles[-25:],
        "history_count":     len(history),
        "favorites_count":   len(favorites),
        "favorites_sample":  [
            {"title": f.get("title", ""), "area": f.get("area", ""),
             "viral_hook": f.get("viral_hook", "")}
            for f in favorites[-5:]
        ],
        "history_analysis":  analysis,
    }

    PRECOMPUTED_DIR.mkdir(exist_ok=True)
    CONTEXT_PACK_FILE.write_text(
        json.dumps(context_pack, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Precompute] Context pack: {len(history)} history entries, "
          f"{len(banned)} banned words, target areas: {targets}")
    return context_pack


# ── Step 2: Generate search plan ─────────────────────────────────────────────

def generate_search_plan(context_pack: dict) -> dict:
    """
    Use Ollama to generate specific search queries for each batch angle × target area.
    Saves precomputed/search_plan.json consumed by idea_generator_agent.
    """
    from langchain_core.messages import HumanMessage
    from ollama_client import get_ollama_llm
    from agents.idea_generator_agent import _BATCH_ANGLES, _pick_variety_queries, _SEARCH_VARIETY

    llm          = get_ollama_llm(temperature=0.5)
    target_areas = context_pack["target_areas"]
    banned_words = context_pack["banned_words"]
    recent_titles = context_pack["recent_titles"]

    batches = []

    def _gen_batch(batch_num: int, angle_label: str, angle_hints: str) -> dict:
        picks = _pick_variety_queries(
            target_areas=target_areas,
            used_queries=[],
            n=6,
            seed=batch_num,
        ) if _SEARCH_VARIETY else []

        area_list    = "\n".join(f"{i+1}. {a}" for i, a in enumerate(target_areas))
        variety_str  = "\n".join(f"• {p}" for p in picks)
        banned_str   = ", ".join(banned_words[:20])
        recent_str   = "\n".join(f"• {t}" for t in recent_titles[-10:])

        prompt = f"""You are a search query planner for a Telugu YouTube Shorts educational channel.

BATCH FOCUS: {angle_label}
GUIDANCE: {angle_hints}

TARGET AREAS (generate 4 queries per area):
{area_list}

SPECIFIC SEARCH SUBJECTS (use these as starting points — don't copy verbatim):
{variety_str}

ALREADY COVERED (avoid repeating these topics):
{recent_str}

AVOID THESE WORDS IN QUERIES: {banned_str}

RULES:
• Each query must name a SPECIFIC entity, mechanism, study, event, or case (not generic)
• Good: "Helios Airways 522 crew incapacitation pressurization NTSB 2005"
• Bad: "aviation shocking facts"
• Use source operators where useful: site:ntsb.gov, site:nasa.gov, site:sciencedaily.com, site:arxiv.org
• Queries for different areas must be completely different topics
• AVOID these ALREADY SATURATED topics on YouTube — do NOT generate queries about them:
  Dunning-Kruger effect, anchoring effect / priming, Milgram experiment, Stanford Prison experiment,
  Marshmallow test, Bystander effect, Tacoma Narrows Bridge, Placebo effect, Baader-Meinhof phenomenon,
  Dunbar's number, Grigori Perelman prize, Desmond Doss Hacksaw Ridge, George Dantzig homework,
  Tsutomu Yamaguchi, Napoleon rabbits, Emu War, CIA cat experiment, Petrov nuclear false alarm,
  PAPI lights, ship anchor chain, U-2 chase car, GPS Einstein correction, Dead Hand system,
  Göbekli Tepe, Antikythera mechanism, Toraja ritual, Famadihana, Naghol land diving,
  Krakatoa pressure wave, Voyager interstellar, Bronze Age collapse
  → If your query is about any of these, replace it with something from a less-known domain.

Return ONLY valid JSON (no markdown, no explanation):
{{
  "queries_by_area": {{
    "area name exactly as listed": ["query1", "query2", "query3", "query4"]
  }},
  "youtube_queries": []
}}"""

        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            parsed = _parse_json_response(_extract_text(resp))
            # Validate structure
            qba = parsed.get("queries_by_area", {})
            ytq = parsed.get("youtube_queries", [])
            print(f"[Precompute] Batch {batch_num} ({angle_label[:40]}): "
                  f"{sum(len(v) for v in qba.values())} queries across {len(qba)} areas")
            return {
                "batch_num":       batch_num,
                "angle":           angle_label,
                "queries_by_area": qba,
                "youtube_queries": ytq,
            }
        except Exception as e:
            print(f"[Precompute] Batch {batch_num} failed: {e}")
            return {
                "batch_num":       batch_num,
                "angle":           angle_label,
                "queries_by_area": {},
                "youtube_queries": [],
            }

    # Generate all batch plans (sequentially — Ollama is single-threaded)
    for batch_num, (angle_label, angle_hints) in enumerate(_BATCH_ANGLES):
        batch = _gen_batch(batch_num, angle_label, angle_hints)
        batches.append(batch)

    search_plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_areas": target_areas,
        "batches":      batches,
    }

    PRECOMPUTED_DIR.mkdir(exist_ok=True)
    SEARCH_PLAN_FILE.write_text(
        json.dumps(search_plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    total_queries = sum(
        len(q) for b in batches for q in b["queries_by_area"].values()
    )
    print(f"[Precompute] Search plan saved — {len(batches)} batches, {total_queries} total queries")
    return search_plan


# ── Step 3: Audience analysis (optional, runs less frequently) ───────────────

def run_audience_analysis() -> None:
    """Refresh audience insights using Ollama (saves to VectorDB)."""
    try:
        from agents.audience_analysis_agent import analyze_audience_interest_node
        analyze_audience_interest_node({})
        print("[Precompute] Audience analysis complete")
    except Exception as e:
        print(f"[Precompute] Audience analysis failed: {e}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all(force: bool = False) -> None:
    """Run all precomputation steps if files are stale."""
    PRECOMPUTED_DIR.mkdir(exist_ok=True)

    needs_plan    = force or _is_stale(SEARCH_PLAN_FILE)
    needs_context = force or _is_stale(CONTEXT_PACK_FILE)

    if not needs_plan and not needs_context:
        print("[Precompute] All files are fresh — nothing to do")
        return

    t0 = time.time()
    print(f"[Precompute] Starting at {datetime.now().strftime('%H:%M:%S')}...")

    context_pack = build_context_pack()
    generate_search_plan(context_pack)

    elapsed = time.time() - t0
    print(f"[Precompute] Done in {elapsed:.0f}s — files ready in {PRECOMPUTED_DIR}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_all(force=force)
