#!/usr/bin/env python3
"""
Multi-Agent YouTube Shorts Idea Generator
Agents: Research → Freshness Check → Your Approval → Image Prompt Generator
         → Image Generation → Telugu Script
"""
import sys
import os
import json
import uuid
from pathlib import Path

def _load_env(path: str):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())
    except FileNotFoundError:
        pass

_load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

missing = []
if not os.getenv("ANTHROPIC_API_KEY"):
    missing.append("ANTHROPIC_API_KEY")
if not os.getenv("TAVILY_API_KEY"):
    missing.append("TAVILY_API_KEY")

if missing:
    print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
    print("Copy .env.example to .env and fill in your API keys.")
    sys.exit(1)

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from workflow import build_graph
from state import AgentState

OUTPUT_BASE = Path(__file__).parent / "generated_images"


# ── Output helpers ─────────────────────────────────────────────────────────────

def _topic_slug(best_idea: dict) -> str:
    title = best_idea.get("title", "short").lower()
    return "".join(c if c.isalnum() else "_" for c in title)[:30]


def _save_storyboard(run_dir: Path, best_idea: dict, scenes: list, generated: list):
    """Save storyboard as both JSON and a human-readable TXT."""
    data = {
        "idea": best_idea,
        "scenes": [
            {
                **scene,
                "image_path": next(
                    (g.get("path") for g in generated
                     if g.get("scene_number") == scene.get("scene_number")), None
                ),
                "image_source": next(
                    (g.get("source") for g in generated
                     if g.get("scene_number") == scene.get("scene_number")), None
                ),
            }
            for scene in scenes
        ],
    }

    # JSON
    json_path = run_dir / "storyboard.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Human-readable TXT
    txt_path = run_dir / "storyboard.txt"
    lines = [
        "=" * 60,
        "STORYBOARD",
        "=" * 60,
        f"Title   : {best_idea.get('title', '')}",
        f"Concept : {best_idea.get('concept', '')}",
        f"Why best: {best_idea.get('why_best', '')}",
        "",
        f"Total duration : {sum(s.get('duration_seconds', 5) for s in scenes)}s",
        f"Scenes         : {len(scenes)}",
        "-" * 60,
    ]
    for s in data["scenes"]:
        lines += [
            "",
            f"Scene {s.get('scene_number'):02d}  [{s.get('duration_seconds')}s]",
            f"  Narration : {s.get('narration', '')}",
            f"  Prompt    : {s.get('image_prompt', '')}",
            f"  Image     : {s.get('image_path') or 'NOT GENERATED'}  [{s.get('image_source', '')}]",
        ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"  Storyboard → {run_dir.name}/storyboard.json + storyboard.txt")
    return json_path, txt_path


def _save_telugu_script(run_dir: Path, script: dict):
    """Save Telugu script as both JSON and a human-readable TXT."""
    if not script:
        print("  Script     → SKIPPED (script agent returned nothing)")
        return

    # Always save the raw JSON regardless of structure
    json_path = run_dir / "telugu_script.json"
    json_path.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")

    txt_path = run_dir / "telugu_script.txt"

    # If JSON parsing failed in the agent, save the raw text as-is
    if script.get("raw"):
        txt_path.write_text(script["raw"], encoding="utf-8")
        print(f"  Script     → {run_dir.name}/telugu_script.json + telugu_script.txt  [raw fallback]")
        return

    lines = [
        script.get("title", ""),
        "",
        f"🎣 HOOK",
        script.get("hook", ""),
        "",
    ]
    for sc in script.get("scenes", []):
        n   = sc.get("scene_number", "")
        dur = sc.get("duration_seconds", "")
        lines += [
            f"── Scene {n}  ({dur} seconds) ──",
            sc.get("telugu_script", ""),
            f"[ {sc.get('transliteration', '')} ]",
            f"({sc.get('english_note', '')})",
            "",
        ]
    lines += [
        f"📣 CALL TO ACTION",
        script.get("call_to_action", ""),
        "",
        f"Total duration: {script.get('total_duration', '')} seconds",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Script     → {run_dir.name}/telugu_script.json + telugu_script.txt")


# ── Workflow runner ────────────────────────────────────────────────────────────

def run_workflow(topic: str) -> dict:
    checkpointer = MemorySaver()
    app = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state: AgentState = {
        "input_topic": topic,
        "research_ideas": [],
        "freshness_approved": None,
        "best_idea": None,
        "rejection_reason": "",
        "iteration": 0,
        "image_prompts": [],
        "human_approved": None,
        "generated_images": [],
        "telugu_script": {},
        "messages": [],
    }

    print(f"\n{'='*60}")
    print(f"Multi-Agent YouTube Shorts Workflow")
    print(f"Topic: {topic}")
    print(f"{'='*60}")

    app.invoke(initial_state, config)

    # ── Human approval interrupt loop ──────────────────────────
    while True:
        snapshot = app.get_state(config)
        if not snapshot.next:
            break

        interrupt_values = [
            intr.value
            for task in snapshot.tasks
            for intr in getattr(task, "interrupts", [])
        ]
        if not interrupt_values:
            break

        answer = input("\nYour decision (yes / no): ").strip() or "no"
        app.invoke(Command(resume=answer), config)

    # ── Collect final state ────────────────────────────────────
    final_state = app.get_state(config).values
    best_idea   = final_state.get("best_idea", {})
    scenes      = final_state.get("image_prompts", [])
    generated   = final_state.get("generated_images", [])
    script      = final_state.get("telugu_script", {})

    # ── Console summary ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL OUTPUT")
    print(f"{'='*60}")

    if best_idea:
        print(f"\nIDEA    : {best_idea.get('title', '')}")
        print(f"Concept : {best_idea.get('concept', '')}")

    if scenes:
        total = sum(s.get("duration_seconds", 5) for s in scenes)
        print(f"\nSTORYBOARD — {len(scenes)} scenes, {total}s")
        for s in scenes:
            print(f"  Scene {s.get('scene_number'):02d} [{s.get('duration_seconds')}s]  {s.get('narration', '')}")

    if generated:
        ok = sum(1 for g in generated if g.get("path"))
        print(f"\nIMAGES — {ok}/{len(generated)} saved")
        for g in generated:
            if g.get("path"):
                print(f"  Scene {g.get('scene_number'):02d}: {Path(g['path']).name}  [{g.get('source')}]")
            else:
                print(f"  Scene {g.get('scene_number'):02d}: FAILED — {g.get('error', '')}")

    if script and not script.get("raw"):
        print(f"\nTELUGU SCRIPT")
        print(f"  Title : {script.get('title', '')}")
        print(f"  Hook  : {script.get('hook', '')}")
        for sc in script.get("scenes", []):
            print(f"  Scene {sc.get('scene_number')} [{sc.get('duration_seconds')}s]: {sc.get('telugu_script', '')}")
        print(f"  CTA   : {script.get('call_to_action', '')}")

    # ── Save outputs to topic folder ───────────────────────────
    if best_idea and scenes:
        slug    = _topic_slug(best_idea)
        run_dir = OUTPUT_BASE / slug
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"SAVED FILES → generated_images/{slug}/")
        print(f"{'='*60}")
        _save_storyboard(run_dir, best_idea, scenes, generated)
        print(f"  [debug] telugu_script keys: {list(script.keys()) if script else 'EMPTY'}")
        _save_telugu_script(run_dir, script)

    print(f"\nDone in {final_state.get('iteration', 0)} iteration(s).")
    return final_state


def main():
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = input("Enter a topic for YouTube Shorts ideas: ").strip()
        if not topic:
            topic = "starter pistol in racing"
            print(f"Using default topic: {topic}")

    run_workflow(topic)


if __name__ == "__main__":
    main()
