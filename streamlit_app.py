import streamlit as st
import sys
import os
import json
import shutil
import traceback
from pathlib import Path
# ── Env & path setup ──────────────────────────────────────────────────────────

def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()
sys.path.insert(0, str(Path(__file__).parent))


def _ensure_openserp():
    """Start the OpenSERP Docker container if it isn't already running."""
    import subprocess, time
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", "openserp"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip() == "true":
            return  # already running
        # Container exists but stopped — remove it so we can restart cleanly
        subprocess.run(["docker", "rm", "-f", "openserp"], capture_output=True, timeout=5)
    except FileNotFoundError:
        print("[OpenSERP] Docker not found — skipping auto-start")
        return
    except Exception:
        pass  # container doesn't exist yet, proceed to start it

    try:
        subprocess.run([
            "docker", "run", "--rm", "-d", "--name", "openserp",
            "-p", "127.0.0.1:7000:7000",
            "karust/openserp:latest",
            "serve", "-a", "0.0.0.0", "-p", "7000",
        ], check=True, capture_output=True, timeout=30)
        print("[OpenSERP] Container started — waiting for it to be ready...")
        time.sleep(3)
        print("[OpenSERP] Ready at http://localhost:7000")
    except subprocess.CalledProcessError as e:
        print(f"[OpenSERP] Failed to start container: {e.stderr.decode().strip()}")
    except Exception as e:
        print(f"[OpenSERP] Auto-start error: {e}")

_ensure_openserp()


_OLLAMA_STOPPED_FLAG = Path(__file__).parent / ".ollama_stopped"


def _ensure_ollama():
    """Ensure Ollama is running and qwen3.5:9b is pre-loaded into memory.
    Skipped if the user explicitly stopped Ollama via the UI (flag file present)."""
    import subprocess, time, threading

    # User explicitly stopped Ollama — don't auto-restart on rerun
    if _OLLAMA_STOPPED_FLAG.exists():
        return

    MODEL = "qwen3.5:9b"
    OLLAMA_URL = "http://localhost:11434"

    def _running() -> bool:
        try:
            import requests as _r
            return _r.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok
        except Exception:
            return False

    def _preload():
        try:
            import requests as _r
            resp = _r.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL, "prompt": "", "keep_alive": -1},
                timeout=120,
                stream=True,
            )
            for _ in resp.iter_content(chunk_size=1024):
                pass
            print(f"[Ollama] Model {MODEL} loaded and ready")
        except Exception as e:
            print(f"[Ollama] Could not pre-load {MODEL}: {e}")

    if _running():
        print(f"[Ollama] Already running — pre-loading {MODEL} in background")
        threading.Thread(target=_preload, daemon=True).start()
        return

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[Ollama] Server starting...")
        for _ in range(15):
            time.sleep(1)
            if _running():
                print(f"[Ollama] Server ready — pre-loading {MODEL} in background")
                threading.Thread(target=_preload, daemon=True).start()
                return
        print("[Ollama] Server did not become ready in 15s — skipping model pre-load")
    except FileNotFoundError:
        print("[Ollama] Not installed — skipping")
    except Exception as e:
        print(f"[Ollama] Error: {e}")

_ensure_ollama()


def _ensure_precomputed():
    """Trigger offline precomputation in a background thread if files are stale."""
    import threading
    from pathlib import Path as _Path

    search_plan = _Path(__file__).parent / "precomputed" / "search_plan.json"

    def _needs_refresh(p: _Path, hours: float = 6.0) -> bool:
        if not p.exists():
            return True
        try:
            import json as _j
            from datetime import datetime as _dt, timezone as _tz
            d = _j.loads(p.read_text())
            age = (_dt.now(_tz.utc) - _dt.fromisoformat(d["generated_at"])).total_seconds() / 3600
            return age > hours
        except Exception:
            return True

    if not _needs_refresh(search_plan):
        return

    def _run():
        try:
            import precompute
            precompute.run_all()
        except Exception as e:
            print(f"[Precompute] Background run failed: {e}")

    t = threading.Thread(target=_run, daemon=True, name="precompute")
    t.start()
    print("[Precompute] Background precomputation started (search plan is stale or missing)")

_ensure_precomputed()

import token_tracker
token_tracker.patch_openai()      # wrap ChatOpenAI.invoke once for the whole process
token_tracker.patch_ollama()      # wrap ChatOllama.invoke once for the whole process

# Import agent functions directly — no LangGraph stream/interrupt needed in UI
from agents.idea_generator_agent import (
    idea_generator_agent_node, save_to_blacklist, save_to_favorites,
    _load_history, _compact_history,
    save_to_bookmarks, load_bookmarks, remove_from_bookmarks, update_history_saturation,
    load_audience_insights,
)
from agents.research_agent import research_agent_node
from agents.freshness_agent import freshness_check_agent_node, bulk_saturation_filter, check_idea_saturation
from agents.image_prompt_agent import image_prompt_agent_node
from agents.script_agent import script_agent_node
from agents.audience_analysis_agent import analyze_audience_interest_node
from agents.channel_analysis_agent import analyze_channel_node, load_channel_analysis as load_ca
from agents.channel_performance_agent import run_performance_analysis, load_performance
from agents.chat_orchestrator_agent import detect_intent, chat_response
from settings import get_channel_handle, get_channel_url, SECTIONS, SETTINGS_SCHEMA, load_all as _load_settings_all, save_settings as _save_settings

OUTPUT_BASE = Path(__file__).parent / "generated_images"
MAX_ITERATIONS = 3


# ══════════════════════════════════════════════════════════════════════════════
# File save helper
# ══════════════════════════════════════════════════════════════════════════════

def _save_outputs(run_dir, best_idea, scenes, script):
    data = {"idea": best_idea, "scenes": scenes}
    (run_dir / "storyboard.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["=" * 60, "STORYBOARD", "=" * 60,
             f"Title   : {best_idea.get('title','')}",
             f"Concept : {best_idea.get('concept','')}",
             f"Why best: {best_idea.get('why_best','')}",
             "", f"Total: {sum(s.get('duration_seconds', 5) for s in scenes)}s", "-" * 60]
    for s in scenes:
        lines += ["",
                  f"Scene {s.get('scene_number') or 0:02d}  [{s.get('duration_seconds')}s]",
                  f"  Narration : {s.get('narration','')}",
                  f"  Prompt    : {s.get('image_prompt','')}"]
    (run_dir / "storyboard.txt").write_text("\n".join(lines), encoding="utf-8")

    if script:
        (run_dir / "telugu_script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")

    if script and not script.get("raw"):
        txt = [script.get("title", ""), "", "HOOK", script.get("hook", ""), ""]
        for sc in script.get("scenes", []):
            txt += [f"── Scene {sc.get('scene_number')}  ({sc.get('duration_seconds')} seconds) ──",
                    sc.get("telugu_script", ""),
                    f"[ {sc.get('transliteration', '')} ]",
                    f"({sc.get('english_note', '')})", ""]
        txt += ["CALL TO ACTION", script.get("call_to_action", ""), "",
                f"Total duration: {script.get('total_duration', '')} seconds"]
        (run_dir / "telugu_script.txt").write_text("\n".join(txt), encoding="utf-8")
    elif script and script.get("raw"):
        (run_dir / "telugu_script.txt").write_text(script["raw"], encoding="utf-8")


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="YouTube Shorts AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Styles are injected dynamically by _inject_phase_theme() after session init


# ── Session state init ────────────────────────────────────────────────────────

def _init():
    defaults = {
        "phase": "input",
        "topic": "",
        "use_idea_generator": False,
        "generated_ideas": [],
        "selected_idea": None,
        "idea_gen_iteration": 0,
        "research_ideas": [],
        "best_idea": None,
        "freshness_approved": None,
        "rejection_reason": "",
        "iteration": 0,
        "image_prompts": [],
        "telugu_script": {},
        "approval_answer": None,
        "all_displayed_ideas": [],    # accumulates across "Generate more" clicks
        "_loved_this_session": [],    # ideas loved in this browser session
        "_sat_check_done": False,     # saturation check phase: has the check run?
        "_sat_check_result": {},      # saturation check phase: result from freshness agent
        "_history_page": 0,            # current page in the history browser
        "_history_blacklisted": set(), # titles blacklisted during this history session
        "_bookmarks_page": 0,              # current page in the bookmarks browser
        "_bookmarked_this_session": set(), # titles bookmarked during this session
        "_bm_removed_this_session": set(), # titles removed from bookmarks this session
        "_audience_analysis_done": False,  # has the audience analysis run this session?
        "_chat_history": [],               # conversation history for chat phase
        "_chat_context_idea": None,        # active idea being discussed in chat
        "_ca_done": False,                 # channel analysis completed this session
        "_ca_result": None,                # channel analysis result dict
        "_force_ca_refresh": False,        # force re-fetch even when cache exists
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Per-phase theme system ─────────────────────────────────────────────────────

_PHASE_THEMES = {
    # gradient uses rgba so the bg image bleeds through at ~15-20% opacity
    "input": {
        "a": "#FF0000",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(28,0,0,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1574717024653-61fd2cf4d44d?auto=format&fit=crop&w=1920&q=25",
        # YouTube creator filming a video
    },
    "history": {
        "a": "#F59E0B",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(28,17,0,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=1920&q=25",
        # Egyptian pyramids — timeless archive
    },
    "bookmarks": {
        "a": "#14B8A6",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(0,28,24,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?auto=format&fit=crop&w=1920&q=25",
        # Grand old library — saved collection
    },
    "audience_analysis": {
        "a": "#22C55E",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(0,28,7,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1511578314322-e5f29b2e78ef?auto=format&fit=crop&w=1920&q=25",
        # Concert crowd — Telugu audience
    },
    "chat": {
        "a": "#7C3AED",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(17,10,36,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=1920&q=25",
        # Modern creative studio workspace
    },
    "idea_selection": {
        "a": "#8B5CF6",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(15,9,34,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1532619187608-e5375cab36aa?auto=format&fit=crop&w=1920&q=25",
        # Array of glowing lightbulbs — ideas
    },
    "saturation_check": {
        "a": "#F97316",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(28,15,0,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=1920&q=25",
        # Analytics charts on MacBook — market data
    },
    "research": {
        "a": "#3B82F6",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(0,15,36,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=1920&q=25",
        # Test tubes in a laboratory — research
    },
    "approval": {
        "a": "#10B981",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(0,28,14,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?auto=format&fit=crop&w=1920&q=25",
        # Person reviewing creative work at desk
    },
    "generating": {
        "a": "#A855F7",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(20,0,41,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?auto=format&fit=crop&w=1920&q=25",
        # Video production camera rig
    },
    "done": {
        "a": "#EAB308",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(28,26,0,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1533174072545-7a4b6ad7a6c3?auto=format&fit=crop&w=1920&q=25",
        # Fireworks celebration — Short is ready!
    },
    "channel_analysis": {
        "a": "#FF0000",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.90) 0%,rgba(22,0,0,.84) 55%,rgba(10,10,10,.90) 100%)",
        "img": "https://images.unsplash.com/photo-1611162617474-5b21e879e113?auto=format&fit=crop&w=1920&q=25",
        # YouTube phone/app screen
    },
    "channel_performance": {
        "a": "#10B981",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(0,22,14,.82) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1920&q=25",
        # Analytics dashboard
    },
    "ready_boards": {
        "a": "#06B6D4",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(0,22,28,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1920&q=25",
        # Cinema seats — your library of ready Shorts
    },
    "board_view": {
        "a": "#06B6D4",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(0,22,28,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1920&q=25",
    },
    "subtitles_upload": {
        "a": "#EC4899",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(28,0,18,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1478737270239-2f02b77fc618?auto=format&fit=crop&w=1920&q=25",
    },
    "subtitles_review": {
        "a": "#EC4899",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(28,0,18,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1478737270239-2f02b77fc618?auto=format&fit=crop&w=1920&q=25",
    },
    "subtitles_video": {
        "a": "#F59E0B",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(28,20,0,.80) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1492691527719-9d1e07e534b4?auto=format&fit=crop&w=1920&q=25",
    },
    "settings": {
        "a": "#64748B",
        "bg": "linear-gradient(160deg,rgba(10,10,10,.88) 0%,rgba(10,14,20,.82) 55%,rgba(10,10,10,.88) 100%)",
        "img": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1920&q=25",
    },
}


def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _inject_phase_theme() -> None:
    t   = _PHASE_THEMES.get(st.session_state.get("phase", "input"), _PHASE_THEMES["input"])
    a, bg, img = t["a"], t["bg"], t["img"]
    ar  = _hex_to_rgb(a)

    st.html(f"""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>
/* ── RESET & TOKENS ─────────────────────────────────────────────── */
:root {{
  --a:{a}; --ar:{ar};
  --s0:rgba(255,255,255,0.025); --s1:rgba(255,255,255,0.045);
  --s2:rgba(255,255,255,0.075); --s3:rgba(255,255,255,0.11);
  --b0:rgba(255,255,255,0.07);  --b1:rgba(255,255,255,0.13);
  --ba:rgba({ar},0.18);          --ba2:rgba({ar},0.32);
  --txt:#ECEDF5; --txt2:#7A7A94; --txt3:#48485E;
  --r1:10px; --r2:16px; --r3:22px;
  --card-shadow:0 1px 3px rgba(0,0,0,0.45),0 6px 24px rgba(0,0,0,0.28);
  --btn-shadow:0 2px 10px rgba({ar},0.30),0 1px 3px rgba(0,0,0,0.45);
}}

/* ── GLOBAL TYPOGRAPHY ──────────────────────────────────────────── */
*,*::before,*::after {{
  font-family:'Plus Jakarta Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
  box-sizing:border-box;
}}

/* ── APP SHELL ──────────────────────────────────────────────────── */
.stApp {{
  background-image:{bg},url('{img}') !important;
  background-size:auto,cover !important;
  background-position:0 0,center center !important;
  background-attachment:fixed,fixed !important;
  color:var(--txt) !important;
}}
.main .block-container {{
  background:transparent !important;
  padding-top:1.5rem !important;
  padding-bottom:4rem !important;
  max-width:940px !important;
}}

/* ── TOP CHROME ─────────────────────────────────────────────────── */
[data-testid="stHeader"] {{
  background:rgba(6,6,9,0.90) !important;
  backdrop-filter:blur(28px) saturate(160%) !important;
  -webkit-backdrop-filter:blur(28px) saturate(160%) !important;
  border-bottom:1px solid rgba({ar},0.11) !important;
}}
[data-testid="stDecoration"] {{ display:none !important; }}
[data-testid="stToolbar"]    {{ display:none !important; }}

/* ── BUTTONS ────────────────────────────────────────────────────── */
[data-testid="stButton"]>button {{
  font-weight:600 !important; font-size:0.875rem !important;
  letter-spacing:0.01em !important; border-radius:var(--r1) !important;
  transition:all 0.17s cubic-bezier(0.4,0,0.2,1) !important;
  padding:8px 16px !important;
}}
[data-testid="stButton"]>button[kind="primary"] {{
  background:linear-gradient(135deg,{a} 0%,rgba({ar},0.78) 100%) !important;
  color:#fff !important; border:1px solid rgba({ar},0.45) !important;
  box-shadow:var(--btn-shadow) !important; text-shadow:0 1px 2px rgba(0,0,0,0.28) !important;
}}
[data-testid="stButton"]>button[kind="primary"]:hover {{
  box-shadow:0 4px 22px rgba({ar},0.50),0 1px 4px rgba(0,0,0,0.5) !important;
  transform:translateY(-1px) !important; filter:brightness(1.10) !important;
}}
[data-testid="stButton"]>button[kind="secondary"],
[data-testid="stButton"]>button[kind="tertiary"] {{
  background:var(--s1) !important; color:var(--txt) !important;
  border:1px solid var(--b0) !important;
}}
[data-testid="stButton"]>button[kind="secondary"]:hover,
[data-testid="stButton"]>button[kind="tertiary"]:hover {{
  background:var(--s2) !important; border-color:var(--ba2) !important; color:#fff !important;
}}
[data-testid="stButton"]>button:disabled {{
  opacity:0.33 !important; cursor:not-allowed !important;
  transform:none !important; filter:none !important;
}}

/* ── INPUTS ─────────────────────────────────────────────────────── */
[data-testid="stTextInput"]>div>div>input,
[data-testid="stTextArea"]>div>div>textarea {{
  background:rgba(255,255,255,0.05) !important; border:1px solid var(--b0) !important;
  color:var(--txt) !important; border-radius:var(--r1) !important;
  font-size:0.925rem !important; padding:9px 13px !important;
  transition:border-color 0.14s,box-shadow 0.14s !important;
}}
[data-testid="stTextInput"]>div>div>input:focus,
[data-testid="stTextArea"]>div>div>textarea:focus {{
  border-color:{a} !important; box-shadow:0 0 0 3px rgba({ar},0.14) !important;
  background:rgba(255,255,255,0.07) !important; outline:none !important;
}}
[data-testid="stTextInput"]>div>div>input::placeholder,
[data-testid="stTextArea"]>div>div>textarea::placeholder {{ color:var(--txt3) !important; }}
[data-testid="stTextInput"] label,[data-testid="stTextArea"] label {{
  color:var(--txt2) !important; font-size:0.79rem !important;
  font-weight:700 !important; letter-spacing:0.06em !important; text-transform:uppercase !important;
}}

/* ── SELECTBOX ──────────────────────────────────────────────────── */
[data-testid="stSelectbox"]>div>div {{
  background:rgba(255,255,255,0.05) !important; border:1px solid var(--b0) !important;
  color:var(--txt) !important; border-radius:var(--r1) !important;
}}
[data-testid="stSelectbox"]>div>div:focus-within {{
  border-color:{a} !important; box-shadow:0 0 0 3px rgba({ar},0.14) !important;
}}
[data-testid="stSelectbox"] label {{
  color:var(--txt2) !important; font-size:0.79rem !important;
  font-weight:700 !important; letter-spacing:0.06em !important; text-transform:uppercase !important;
}}

/* ── BORDERED CONTAINERS ────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {{
  background:linear-gradient(145deg,rgba(255,255,255,0.045) 0%,rgba(255,255,255,0.022) 100%) !important;
  border:1px solid var(--b0) !important; border-radius:var(--r2) !important;
  box-shadow:0 1px 3px rgba(0,0,0,0.35) !important;
  transition:border-color 0.18s,box-shadow 0.18s,background 0.18s !important;
  overflow:hidden !important; position:relative !important;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
  border-color:rgba({ar},0.28) !important; background:var(--s2) !important;
  box-shadow:0 0 0 1px rgba({ar},0.07),0 10px 36px rgba(0,0,0,0.28) !important;
}}

/* ── EXPANDERS ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
  border:1px solid var(--b0) !important; border-radius:var(--r2) !important;
  background:var(--s0) !important; overflow:hidden !important; margin-bottom:10px !important;
}}
[data-testid="stExpander"]>div:first-child {{
  background:rgba(255,255,255,0.032) !important; border-bottom:1px solid var(--b0) !important;
  border-radius:0 !important; padding:13px 18px !important;
}}
[data-testid="stExpander"] summary {{
  font-weight:600 !important; color:var(--txt) !important; font-size:0.94rem !important;
}}
[data-testid="stExpander"]>div:last-child {{ padding:18px 18px 20px !important; }}

/* ── STATUS ─────────────────────────────────────────────────────── */
[data-testid="stStatusContainer"] {{
  background:rgba(255,255,255,0.03) !important; border:1px solid var(--b0) !important;
  border-radius:var(--r2) !important;
}}

/* ── ALERTS ─────────────────────────────────────────────────────── */
[data-testid="stAlert"] {{
  border-radius:var(--r1) !important; font-size:0.885rem !important;
}}

/* ── PROGRESS ───────────────────────────────────────────────────── */
[data-testid="stProgress"]>div {{ border-radius:4px !important; overflow:hidden !important; }}
[data-testid="stProgress"]>div>div {{ background:rgba(255,255,255,0.08) !important; height:3px !important; }}
[data-testid="stProgress"]>div>div>div {{
  background:linear-gradient(90deg,{a},rgba({ar},0.65)) !important;
  transition:width 0.5s cubic-bezier(0.4,0,0.2,1) !important;
}}

/* ── CHAT ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {{
  background:rgba(255,255,255,0.032) !important; border:1px solid var(--b0) !important;
  border-radius:var(--r2) !important; padding:14px 18px !important;
}}
[data-testid="stChatInput"]>div>div {{
  background:rgba(255,255,255,0.05) !important; border:1px solid var(--b0) !important;
  border-radius:var(--r2) !important;
}}
[data-testid="stChatInput"] textarea {{
  background:transparent !important; color:var(--txt) !important; border:none !important;
}}
[data-testid="stChatInput"]:focus-within>div>div {{
  border-color:{a} !important; box-shadow:0 0 0 3px rgba({ar},0.12) !important;
}}

/* ── TABLES / DATAFRAMES ────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
  border-radius:var(--r2) !important; overflow:hidden !important;
  border:1px solid var(--b0) !important;
}}

/* ── DIVIDERS ───────────────────────────────────────────────────── */
hr {{ border:none !important; border-top:1px solid rgba(255,255,255,0.07) !important; margin:18px 0 !important; }}

/* ── CAPTION ────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] {{ color:var(--txt2) !important; font-size:0.82rem !important; }}

/* ── SCROLLBAR ──────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width:4px; height:4px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:rgba({ar},0.32); border-radius:2px; }}
::-webkit-scrollbar-thumb:hover {{ background:rgba({ar},0.58); }}

/* ══ CUSTOM COMPONENTS ══════════════════════════════════════════════ */

/* ── App title ──────────────────────────────────────────────────── */
.app-title {{
  text-align:center; padding:6px 0 2px;
}}
.app-title-main {{
  font-family:'Syne',sans-serif !important; font-size:clamp(1.65rem,4vw,2.3rem);
  font-weight:800; color:var(--txt); letter-spacing:-0.025em; line-height:1.1;
}}
.app-title-accent {{ color:{a}; }}
.app-title-sub {{
  font-size:0.81rem; color:var(--txt2); margin-top:5px; letter-spacing:0.01em;
}}

/* ── Screen header ──────────────────────────────────────────────── */
.screen-hdr {{
  padding:2px 0 20px; border-bottom:1px solid rgba({ar},0.13); margin-bottom:24px;
}}
.screen-hdr-title {{
  font-family:'Syne',sans-serif !important; font-size:clamp(1.4rem,2.8vw,1.9rem) !important;
  font-weight:800 !important; color:var(--txt) !important; margin:0 !important;
  line-height:1.2 !important; text-wrap:balance;
}}
.screen-hdr-sub {{
  color:var(--txt2); font-size:0.875rem; margin-top:5px; line-height:1.55;
}}

/* ── Ollama status ──────────────────────────────────────────────── */
.status-pill {{
  display:inline-flex; align-items:center; gap:8px;
  background:var(--s1); border:1px solid var(--b0); border-radius:40px;
  padding:5px 13px; font-size:0.81rem; font-weight:600; color:var(--txt2);
}}
.dot-green::before {{ content:'●'; color:#22C55E; font-size:0.65em; }}
.dot-red::before   {{ content:'●'; color:#EF4444; font-size:0.65em; }}

/* ── Section label ──────────────────────────────────────────────── */
.slabel {{
  font-size:0.72rem; font-weight:700; letter-spacing:0.09em; text-transform:uppercase;
  color:var(--txt3); margin:0 0 8px;
}}

/* ── Input hero card ────────────────────────────────────────────── */
.input-hero {{
  background:linear-gradient(155deg,rgba(255,255,255,0.05) 0%,rgba(255,255,255,0.02) 100%);
  border:1px solid rgba({ar},0.15); border-radius:var(--r3);
  padding:28px 28px 20px; margin-bottom:8px;
  box-shadow:0 2px 8px rgba(0,0,0,0.35),0 0 40px rgba({ar},0.04);
}}
.input-hero-eyebrow {{
  font-size:0.76rem; font-weight:700; letter-spacing:0.09em; text-transform:uppercase;
  color:{a}; margin-bottom:8px;
}}
.input-hero-title {{
  font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:800; color:var(--txt);
  margin:0 0 4px; line-height:1.3; text-wrap:balance;
}}
.input-hero-sub {{
  font-size:0.83rem; color:var(--txt2); margin:0 0 18px; line-height:1.5;
}}

/* ── Topic area chips ───────────────────────────────────────────── */
.area-section {{ margin-top:10px; }}
.area-section-label {{
  font-size:0.72rem; font-weight:700; letter-spacing:0.09em; text-transform:uppercase;
  color:var(--txt3); margin-bottom:10px;
}}

/* ── Idea card ──────────────────────────────────────────────────── */
.ic {{
  position:relative; padding:20px 22px 18px; overflow:hidden;
  background:linear-gradient(145deg,rgba(255,255,255,0.048) 0%,rgba(255,255,255,0.022) 100%);
}}
.ic-stripe {{
  position:absolute; top:0; left:0; width:3px; height:100%;
  background:linear-gradient(180deg,{a} 0%,rgba({ar},0.25) 100%);
}}
.ic-tag {{
  display:inline-flex; align-items:center; gap:5px; margin-bottom:11px;
  background:rgba({ar},0.11); color:{a}; border:1px solid rgba({ar},0.22);
  border-radius:30px; padding:3px 11px; font-size:0.71rem; font-weight:700;
  letter-spacing:0.07em; text-transform:uppercase;
}}
.ic-sat-low  {{ color:#4ADE80; font-size:0.75rem; font-weight:600; margin-left:8px; }}
.ic-sat-med  {{ color:#FACC15; font-size:0.75rem; font-weight:600; margin-left:8px; }}
.ic-title {{
  font-family:'Syne',sans-serif !important; font-size:1.04rem; font-weight:800;
  color:var(--txt); margin:0 0 9px; line-height:1.35; text-wrap:balance;
}}
.ic-concept {{
  font-size:0.875rem; color:var(--txt2); line-height:1.68; margin:0 0 12px;
}}
.ic-hook {{
  font-size:0.845rem; color:{a}; font-style:italic; line-height:1.55;
  padding:9px 13px; background:rgba({ar},0.07); border-left:2px solid rgba({ar},0.38);
  border-radius:0 8px 8px 0;
}}
/* keep legacy classes working on screens not yet migrated */
.idea-area-tag {{
  display:inline-flex; align-items:center; gap:5px;
  background:rgba({ar},0.11); color:{a}; border:1px solid rgba({ar},0.22);
  border-radius:30px; padding:3px 11px; font-size:0.71rem; font-weight:700;
  letter-spacing:0.07em; text-transform:uppercase; margin-bottom:10px;
}}
.hook-text {{
  font-size:0.845rem; color:{a}; font-style:italic; line-height:1.55;
  padding:9px 13px; background:rgba({ar},0.07); border-left:2px solid rgba({ar},0.38);
  border-radius:0 8px 8px 0;
}}
.telugu-text {{ font-size:1.12em; line-height:1.85; color:var(--txt); }}

/* ── Info note ──────────────────────────────────────────────────── */
.info-note {{
  background:rgba({ar},0.07); border:1px solid rgba({ar},0.18); border-radius:var(--r1);
  padding:11px 16px; font-size:0.86rem; color:var(--txt2); line-height:1.6; margin-bottom:16px;
}}
.info-note code {{
  background:rgba({ar},0.14); color:{a}; padding:1px 5px; border-radius:4px; font-size:0.9em;
}}

/* ── Metric tile (channel perf) ─────────────────────────────────── */
.m-tile {{
  background:var(--s1); border:1px solid var(--b0); border-radius:var(--r2);
  padding:18px 20px; text-align:center;
}}
.m-tile-num {{
  font-family:'Syne',sans-serif; font-size:1.9em; font-weight:800; color:{a}; line-height:1.1;
}}
.m-tile-lbl {{ font-size:0.79rem; color:var(--txt2); margin-top:4px; font-weight:500; }}

/* ── Channel stat card ──────────────────────────────────────────── */
.ch-stat {{
  background:var(--s1); border:1px solid var(--b0); border-radius:var(--r2);
  padding:18px 20px; text-align:center;
}}
.ch-stat-num {{
  font-family:'Syne',sans-serif; font-size:1.85em; font-weight:800; color:var(--txt);
}}
.ch-stat-lbl {{ color:var(--txt2); font-size:0.79rem; margin-top:4px; }}
</style>""")


def _screen_header(icon: str, title: str, subtitle: str = "") -> None:
    sub = f'<p class="screen-hdr-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="screen-hdr">'
        f'<h2 class="screen-hdr-title">{icon}&thinsp; {title}</h2>'
        f'{sub}'
        f'</div>',
        unsafe_allow_html=True,
    )


_inject_phase_theme()


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="app-title">'
    '<div class="app-title-main"><span class="app-title-accent">🎬</span> Shorts<span class="app-title-accent"> AI</span></div>'
    '</div>',
    unsafe_allow_html=True,
)
if st.session_state.get("phase", "input") == "input":
    st.markdown(
        '<p class="app-title-sub" style="text-align:center">'
        'Idea&thinsp;·&thinsp;Research&thinsp;·&thinsp;Freshness&thinsp;·&thinsp;Storyboard&thinsp;·&thinsp;Telugu Script'
        '</p>',
        unsafe_allow_html=True,
    )

# ── Ollama status + controls (main area, always visible) ──────────────────────
import subprocess as _sp

def _ollama_running() -> bool:
    try:
        import requests as _rq
        return _rq.get("http://localhost:11434/api/tags", timeout=2).ok
    except Exception:
        return False

def _stop_ollama():
    import os, time as _t
    _OLLAMA_STOPPED_FLAG.touch()       # prevents _ensure_ollama() from restarting on rerun
    uid = os.getuid()
    _sp.run(["launchctl", "bootout", f"gui/{uid}/com.ollama.ollama"], capture_output=True)
    _t.sleep(0.5)
    _sp.run(["pkill", "-9", "-f", "ollama"], capture_output=True)  # kills serve daemon
    _sp.run(["pkill", "-9", "-f", "Ollama"], capture_output=True)  # kills macOS app + llama-server
    _t.sleep(1)

def _start_ollama():
    import os, time as _t
    _OLLAMA_STOPPED_FLAG.unlink(missing_ok=True)
    if os.path.isdir("/Applications/Ollama.app"):
        _sp.run(["open", "-a", "Ollama"], capture_output=True)
    else:
        _sp.Popen(["ollama", "serve"], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    _t.sleep(3)

_ol_on = _ollama_running()
_sl = "Ollama running — local AI active" if _ol_on else "Ollama stopped — cloud AI active"
_dot_cls = "dot-green" if _ol_on else "dot-red"
_oc1, _oc2, _oc3 = st.columns([4, 1, 1])
with _oc1:
    st.markdown(
        f'<div style="padding-top:5px"><span class="status-pill {_dot_cls}">{_sl}</span></div>',
        unsafe_allow_html=True,
    )
with _oc2:
    if st.button("🛑 Stop", disabled=not _ol_on, use_container_width=True,
                 help="Stop Ollama — all LLM calls switch to Anthropic", key="ol_stop"):
        with st.spinner("Stopping…"):
            _stop_ollama()
        st.toast("Ollama stopped — Anthropic is now handling all AI calls", icon="🛑")
        st.rerun()
with _oc3:
    if st.button("▶️ Start", disabled=_ol_on, use_container_width=True,
                 help="Start local Ollama with qwen3.5:9b", key="ol_start"):
        with st.spinner("Starting…"):
            _start_ollama()
        st.toast("Ollama starting — ready in a few seconds", icon="▶️")
        st.rerun()

# ── Live token-usage display (updated after every LLM call) ──────────────────
_token_ph = st.empty()
token_tracker.set_placeholder(_token_ph)

PHASES = ["input", "idea_selection", "research", "approval", "generating", "done"]
phase_idx = PHASES.index(st.session_state.phase) if st.session_state.phase in PHASES else 0
st.progress(phase_idx / (len(PHASES) - 1))

if st.session_state.phase != "input":
    _, home_col = st.columns([9, 1])
    with home_col:
        if st.button("🏠 Home", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

st.divider()


# ── Reset ─────────────────────────────────────────────────────────────────────

def reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: input
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.phase == "input":

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(
            '<div class="input-hero">'
            '<div class="input-hero-eyebrow">What do you want to make?</div>'
            '<h3 class="input-hero-title">Your next viral Short</h3>'
            '<p class="input-hero-sub">Type a topic below — or let the AI hunt for something nobody\'s covered yet.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        topic = st.text_input(
            "Your own idea",
            placeholder="e.g. ultraedge technology in cricket",
            label_visibility="collapsed",
        )
        b1, b3 = st.columns(2)
        with b1:
            if st.button("🚀 Use My Idea", type="primary", use_container_width=True):
                if topic.strip():
                    st.session_state.topic              = topic.strip()
                    st.session_state.use_idea_generator = False
                    st.session_state.phase              = "research"
                    st.rerun()
                else:
                    st.error("Please type your idea first.")
        with b3:
            if st.button("🎬 Create Storyboard", use_container_width=True):
                if topic.strip():
                    t = topic.strip()
                    st.session_state.topic          = t
                    st.session_state.selected_idea  = {"title": t, "concept": t}
                    st.session_state._skip_research     = True
                    st.session_state.iteration          = 0
                    st.session_state.research_ideas     = []
                    st.session_state.best_idea          = None
                    st.session_state.freshness_approved = None
                    st.session_state.phase              = "research"
                    st.rerun()
                else:
                    st.error("Please type your idea first.")

        # ── Per-topic idea generation buttons ────────────────────────────────────
        st.markdown('<div class="slabel" style="margin-top:20px">Generate ideas by topic</div>', unsafe_allow_html=True)
        _TOPIC_ICONS = {
            "history":            "🏛️",
            "science":            "🔬",
            "latest science news":"📰",
            "geography":          "🌍",
            "nature":             "🌿",
            "space":              "🚀",
            "technology":         "💻",
            "interesting events":  "⚡",
            "inspiring people":   "🌟",
            "archaeology news":   "⛏️",
            "aviation reports":   "✈️",
            "research papers":    "📄",
            "world cultures":     "🎭",
            "hidden mechanisms":  "⚙️",
            "surprising facts":   "🤯",
            "unknown facts":      "💡",
        }
        from agents.idea_generator_agent import AREAS as _ALL_AREAS

        def _go_ideas(forced=None):
            st.session_state.use_idea_generator  = True
            st.session_state.generated_ideas     = []
            st.session_state.all_displayed_ideas = []
            st.session_state.idea_gen_iteration  = 0
            st.session_state._ideas_ready        = False
            st.session_state.forced_areas        = forced
            st.session_state.phase               = "idea_selection"
            st.rerun()

        # "All Topics" button spanning full width
        if st.button("🎲  All Topics — Surprise Me", use_container_width=True):
            _go_ideas(forced=None)

        # Per-area buttons: 3 per row
        _topic_cols = 3
        for _row_start in range(0, len(_ALL_AREAS), _topic_cols):
            _row_areas = _ALL_AREAS[_row_start: _row_start + _topic_cols]
            _cols = st.columns(_topic_cols)
            for _ci, _area in enumerate(_row_areas):
                _icon = _TOPIC_ICONS.get(_area, "📌")
                with _cols[_ci]:
                    if st.button(
                        f"{_icon} {_area.title()}",
                        use_container_width=True,
                        key=f"topic_btn_{_area.replace(' ', '_')}",
                    ):
                        _go_ideas(forced=[_area])

        st.markdown('<div class="slabel" style="margin-top:20px">Tools</div>', unsafe_allow_html=True)
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            if st.button("📊  Check Saturation", use_container_width=True,
                         help="Check if this idea is already saturated on YouTube Shorts"):
                if topic.strip():
                    st.session_state.topic             = topic.strip()
                    st.session_state._sat_check_done   = False
                    st.session_state._sat_check_result = {}
                    st.session_state.phase             = "saturation_check"
                    st.rerun()
                else:
                    st.error("Please type your idea first.")
        with hc2:
            if st.button("📚  Browse History", use_container_width=True,
                         help="Browse all previously generated ideas and turn any into a storyboard"):
                st.session_state._history_page        = 0
                st.session_state._history_blacklisted = set()
                st.session_state.phase                = "history"
                st.rerun()
        with hc3:
            if st.button("🔖  Bookmarks", use_container_width=True,
                         help="View your bookmarked ideas"):
                st.session_state._bookmarks_page = 0
                st.session_state.phase           = "bookmarks"
                st.rerun()

        # ── Audience analysis row ─────────────────────────────────────────────
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        _saved_insights = load_audience_insights()
        _insights_label = "📈  Analyze Public Interest"
        _insights_help  = "Research trending topics for Telugu YouTube Shorts audience and prioritise idea generation"
        if _saved_insights:
            _age = _saved_insights.get("analyzed_at", "")[:10]
            _insights_label = f"📈  Re-Analyze Public Interest"
            _insights_help  = f"Last analyzed: {_age}. Re-run to update trending area scores."
        ai_col, chat_col = st.columns([1, 1])
        with ai_col:
            if st.button(_insights_label, use_container_width=True, help=_insights_help):
                st.session_state._audience_analysis_done = False
                st.session_state.phase                   = "audience_analysis"
                st.rerun()
        with chat_col:
            if st.button("💬  Let's Build Together", use_container_width=True,
                         help="Chat with AI — check saturation, research topics, build storyboards"):
                st.session_state.phase = "chat"
                st.rerun()
        if _saved_insights:
            top5 = _saved_insights.get("top_areas", [])[:5]
            if top5:
                st.caption(f"Top areas for your audience: {' · '.join(top5)}")

        # ── Channel analysis row ──────────────────────────────────────────────
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        _ca_saved  = load_ca()
        _ca_label  = "📺  Re-Analyse My Channel" if _ca_saved else "📺  My Channel Analysis"
        _ca_help   = (
            f"Last analysed: {_ca_saved['analyzed_at'][:10]}. Re-run to refresh stats."
            if _ca_saved else
            f"Analyse all Shorts from {get_channel_handle()} — views, likes & content breakdown by area"
        )
        ca_col, rb_col = st.columns([1, 1])
        with ca_col:
            if st.button(_ca_label, use_container_width=True, help=_ca_help):
                st.session_state._ca_done          = False
                st.session_state._ca_result        = None
                st.session_state._force_ca_refresh = bool(_ca_saved)
                st.session_state.phase             = "channel_analysis"
                st.rerun()
        with rb_col:
            _rb_count = sum(
                1 for d in OUTPUT_BASE.iterdir()
                if d.is_dir() and (d / "storyboard.json").exists()
            ) if OUTPUT_BASE.exists() else 0
            _rb_label = f"📁  View Ready Boards ({_rb_count})" if _rb_count else "📁  View Ready Boards"
            if st.button(_rb_label, use_container_width=True,
                         help="Browse all previously generated storyboards"):
                st.session_state.phase = "ready_boards"
                st.rerun()
        if _ca_saved:
            _ca_total = _ca_saved.get("total_shorts", 0)
            _ca_subs  = _ca_saved.get("channel_stats", {}).get("subscriber_count", 0)
            st.caption(
                f"{get_channel_handle()} · {_ca_total} Shorts analysed"
                + (f" · {_ca_subs:,} subscribers" if _ca_subs else "")
            )

        # ── Channel Performance row ───────────────────────────────────────────
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        _cp_saved = load_performance()
        _cp_label = (
            f"📈  Channel Performance  ·  last run {_cp_saved['generated_at'][:10]}"
            if _cp_saved else "📈  Channel Performance"
        )
        if st.button(_cp_label, use_container_width=True,
                     help="Compare recent Shorts vs all-time high performers — numbers, gaps, and recommendations"):
            st.session_state.phase = "channel_performance"
            st.rerun()

        # ── Subtitles row ─────────────────────────────────────────────────────
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        sub_col, settings_col = st.columns([1, 1])
        with sub_col:
            if st.button("🎙️  Generate Subtitles", use_container_width=True,
                         help="Upload a Telugu MP3 and generate an English SRT subtitle file"):
                st.session_state._sub_segments  = []
                st.session_state._sub_audio_name = ""
                st.session_state.phase           = "subtitles_upload"
                st.rerun()
        with settings_col:
            if st.button("⚙️  Settings", use_container_width=True,
                         help="Configure API keys, LLM models, channel URL and integrations"):
                st.session_state.phase = "settings"
                st.rerun()

        st.caption("Or let the AI browse history, science, space, nature & geography for a viral idea.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: history  — browse ideas_history.json, storyboard or blacklist any idea
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "history":

    PAGE_SIZE = 15

    AREA_ICONS = {
        "history": "🏛️", "science": "🔬", "latest news": "📰",
        "latest science news": "📰", "geography": "🌍",
        "nature": "🌿", "space": "🚀",
        "psychology": "🧠", "psychology studies": "🧠",
        "technology": "💻", "interesting events": "⚡",
        "inspiring people": "🌟", "archaeology news": "🏺",
        "aviation reports": "✈️", "patent filings": "📜",
        "government reports": "🏛", "research papers": "📄",
        "world cultures": "🌐", "india government reports": "🇮🇳",
            "hidden mechanisms": "⚙️",
    }

    all_history = list(reversed(_load_history()))   # newest first

    col_title, col_back = st.columns([5, 1])
    with col_title:
        _screen_header("📚", "Ideas Archive",
                       f"{len(all_history)} ideas · browse, reuse or storyboard any past idea")
    with col_back:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True):
            st.session_state.phase = "input"
            st.rerun()

    # ── Filter controls ───────────────────────────────────────────────────────
    fc1, fc2 = st.columns([3, 1])
    with fc1:
        search = st.text_input("🔍 Search", placeholder="Filter by title or concept...",
                               label_visibility="collapsed", key="_hist_search")
    with fc2:
        known_areas = sorted({i.get("area", "").lower().strip() for i in all_history if i.get("area")})
        area_opts   = ["All areas"] + known_areas
        area_filter = st.selectbox("Area", area_opts, label_visibility="collapsed", key="_hist_area")

    # Apply filters
    blacklisted_this_session = st.session_state.get("_history_blacklisted", set())
    ideas = [i for i in all_history if i.get("title", "") not in blacklisted_this_session]

    if search.strip():
        q = search.strip().lower()
        ideas = [i for i in ideas if q in i.get("title", "").lower() or q in i.get("concept", "").lower()]
    if area_filter != "All areas":
        ideas = [i for i in ideas if i.get("area", "").lower().strip() == area_filter]

    total  = len(ideas)
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page    = min(st.session_state.get("_history_page", 0), n_pages - 1)
    st.session_state._history_page = page

    st.caption(f"{total} idea{'s' if total != 1 else ''} · page {page + 1} of {n_pages}")
    st.divider()

    # ── Idea cards ────────────────────────────────────────────────────────────
    page_ideas = ideas[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

    bookmarked_this_session = st.session_state.get("_bookmarked_this_session", set())

    if not page_ideas:
        st.info("No ideas match your filter." if (search or area_filter != "All areas") else "History is empty.")
    else:
        for idx, idea in enumerate(page_ideas):
            area  = idea.get("area", "science").lower().strip()
            icon  = AREA_ICONS.get(area, "💡")
            title = idea.get("title", "")
            date  = idea.get("generated_at", "")[:10]   # YYYY-MM-DD

            sat_score  = idea.get("saturation_score")
            sat_badge  = {1: "🟢 Fresh", 2: "🟡 Medium", 3: "🔴 Saturated"}.get(sat_score, "")

            col_card, col_sb, col_bk, col_bl = st.columns([6, 1, 1, 1])

            with col_card:
                with st.container(border=True):
                    sat_cls   = {1: "ic-sat-low", 2: "ic-sat-med", 3: "ic-sat-med"}.get(sat_score, "")
                    sat_label = {1: "Fresh", 2: "Medium", 3: "Saturated"}.get(sat_score, "")
                    sat_html  = f'<span class="{sat_cls}">{sat_label}</span>' if sat_cls else ""
                    date_html = f'<span style="font-size:0.73rem;color:var(--txt3);margin-left:8px">{date}</span>' if date else ""
                    t_esc = title.replace("<","&lt;").replace(">","&gt;")
                    c_esc = idea.get("concept","").replace("<","&lt;").replace(">","&gt;")
                    h_esc = idea.get("viral_hook","").replace("<","&lt;").replace(">","&gt;")
                    hook_html = f'<div class="ic-hook">🎣 {h_esc}</div>' if h_esc else ""
                    st.markdown(
                        f'<div class="ic">'
                        f'<div class="ic-stripe"></div>'
                        f'<div><span class="ic-tag">{icon} {area.upper()}</span>{sat_html}{date_html}</div>'
                        f'<div class="ic-title">{t_esc}</div>'
                        f'<div class="ic-concept">{c_esc}</div>'
                        f'{hook_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with col_sb:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("🎬", key=f"hist_sb_{page}_{idx}",
                             use_container_width=True,
                             help="Generate storyboard directly from this idea"):
                    st.session_state.best_idea = {
                        "title":    idea.get("title", ""),
                        "concept":  idea.get("concept", ""),
                        "why_best": "Selected from history",
                    }
                    st.session_state.topic         = idea.get("title", "")
                    st.session_state.image_prompts = []
                    st.session_state.telugu_script = {}
                    st.session_state.phase          = "generating"
                    st.rerun()

            with col_bk:
                st.markdown("<br><br>", unsafe_allow_html=True)
                already_bookmarked = title in bookmarked_this_session
                if st.button("🔖" if not already_bookmarked else "📌",
                             key=f"hist_bk_{page}_{idx}",
                             use_container_width=True,
                             help="Bookmark for later" if not already_bookmarked else "Already bookmarked",
                             disabled=already_bookmarked):
                    saved = save_to_bookmarks(idea, source="history")
                    if saved:
                        st.session_state._bookmarked_this_session = bookmarked_this_session | {title}
                        bookmarked_this_session = st.session_state._bookmarked_this_session
                        st.toast(f"Bookmarked: {title[:50]}", icon="🔖")
                    else:
                        st.toast("Already bookmarked!", icon="📌")
                    st.rerun()

            with col_bl:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("🚫", key=f"hist_bl_{page}_{idx}",
                             use_container_width=True,
                             help="Blacklist — permanently exclude this idea"):
                    save_to_blacklist(idea)
                    _compact_history()   # removes blacklisted entry from history file immediately
                    st.session_state._history_blacklisted = blacklisted_this_session | {title}
                    st.toast(f"Blacklisted: {title[:50]}", icon="🚫")
                    st.rerun()

    # ── Pagination controls ───────────────────────────────────────────────────
    if n_pages > 1:
        st.divider()
        pc1, pc2, pc3 = st.columns([1, 2, 1])
        with pc1:
            if st.button("← Prev", use_container_width=True, disabled=(page == 0)):
                st.session_state._history_page = page - 1
                st.rerun()
        with pc2:
            st.markdown(
                f"<div style='text-align:center;padding-top:8px;color:gray'>Page {page+1} of {n_pages}</div>",
                unsafe_allow_html=True,
            )
        with pc3:
            if st.button("Next →", use_container_width=True, disabled=(page >= n_pages - 1)):
                st.session_state._history_page = page + 1
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: bookmarks  — browse bookmarks.json
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "bookmarks":

    BM_PAGE_SIZE = 15

    BM_AREA_ICONS = {
        "history": "🏛️", "science": "🔬", "latest news": "📰",
        "latest science news": "📰", "geography": "🌍",
        "nature": "🌿", "space": "🚀",
        "psychology": "🧠", "psychology studies": "🧠",
        "technology": "💻", "interesting events": "⚡",
        "inspiring people": "🌟", "archaeology news": "🏺",
        "aviation reports": "✈️", "patent filings": "📜",
        "government reports": "🏛", "research papers": "📄",
        "world cultures": "🌐", "india government reports": "🇮🇳",
            "hidden mechanisms": "⚙️",
    }

    all_bookmarks = list(reversed(load_bookmarks()))  # newest first

    bm_col_title, bm_col_back = st.columns([5, 1])
    with bm_col_title:
        _screen_header("🔖", "Saved Ideas",
                       f"{len(all_bookmarks)} bookmarked · your curated collection for later")
    with bm_col_back:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True, key="bm_back"):
            st.session_state.phase = "input"
            st.rerun()

    removed_this_session = st.session_state.get("_bookmarked_this_session", set())
    # _bookmarked_this_session tracks added; for bookmarks screen we track removed separately
    bm_removed = st.session_state.get("_bm_removed_this_session", set())

    visible = [b for b in all_bookmarks if b.get("title", "") not in bm_removed]

    if not visible:
        st.info("No bookmarks yet. Click 🔖 on any idea to bookmark it for later.")
    else:
        total_bm  = len(visible)
        n_bm_pages = max(1, (total_bm + BM_PAGE_SIZE - 1) // BM_PAGE_SIZE)
        bm_page    = min(st.session_state.get("_bookmarks_page", 0), n_bm_pages - 1)
        st.session_state._bookmarks_page = bm_page

        st.caption(f"{total_bm} bookmark{'s' if total_bm != 1 else ''} · page {bm_page + 1} of {n_bm_pages}")
        st.divider()

        page_bms = visible[bm_page * BM_PAGE_SIZE : (bm_page + 1) * BM_PAGE_SIZE]

        for idx, bm in enumerate(page_bms):
            area   = bm.get("area", "").lower().strip()
            icon   = BM_AREA_ICONS.get(area, "💡")
            title  = bm.get("title", "")
            bm_at  = bm.get("bookmarked_at", "")[:10]
            source = bm.get("bookmark_source", "")
            source_label = {"history": "history", "idea_selection": "generator", "research": "research"}.get(source, source)

            bm_col_card, bm_col_sb, bm_col_rm = st.columns([6, 1, 1])

            with bm_col_card:
                with st.container(border=True):
                    t_esc  = title.replace("<","&lt;").replace(">","&gt;")
                    c_esc  = bm.get("concept","").replace("<","&lt;").replace(">","&gt;")
                    h_esc  = bm.get("viral_hook","").replace("<","&lt;").replace(">","&gt;")
                    meta_parts = []
                    if bm_at:    meta_parts.append(f'<span style="color:var(--txt3);font-size:0.73rem">{bm_at}</span>')
                    if source_label: meta_parts.append(f'<span style="color:#14B8A6;font-size:0.73rem">from {source_label}</span>')
                    meta_html = "&ensp;".join(meta_parts)
                    tag_html = f'<span class="ic-tag">{icon} {area.upper()}</span>' if area else ""
                    hook_html = f'<div class="ic-hook">🎣 {h_esc}</div>' if h_esc else ""
                    st.markdown(
                        f'<div class="ic">'
                        f'<div class="ic-stripe"></div>'
                        f'<div style="margin-bottom:10px">{tag_html}&ensp;{meta_html}</div>'
                        f'<div class="ic-title">{t_esc}</div>'
                        f'<div class="ic-concept">{c_esc}</div>'
                        f'{hook_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with bm_col_sb:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("🎬", key=f"bm_sb_{bm_page}_{idx}",
                             use_container_width=True,
                             help="Generate storyboard from this bookmarked idea"):
                    st.session_state.best_idea = {
                        "title":    bm.get("title", ""),
                        "concept":  bm.get("concept", ""),
                        "why_best": "Selected from bookmarks",
                    }
                    st.session_state.topic         = bm.get("title", "")
                    st.session_state.image_prompts = []
                    st.session_state.telugu_script = {}
                    st.session_state.phase          = "generating"
                    st.rerun()

            with bm_col_rm:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"bm_rm_{bm_page}_{idx}",
                             use_container_width=True,
                             help="Remove from bookmarks"):
                    remove_from_bookmarks(title)
                    bm_removed = bm_removed | {title}
                    st.session_state._bm_removed_this_session = bm_removed
                    st.toast(f"Removed: {title[:50]}", icon="🗑️")
                    st.rerun()

        if n_bm_pages > 1:
            st.divider()
            bpc1, bpc2, bpc3 = st.columns([1, 2, 1])
            with bpc1:
                if st.button("← Prev", key="bm_prev", use_container_width=True, disabled=(bm_page == 0)):
                    st.session_state._bookmarks_page = bm_page - 1
                    st.rerun()
            with bpc2:
                st.markdown(
                    f"<div style='text-align:center;padding-top:8px;color:gray'>Page {bm_page+1} of {n_bm_pages}</div>",
                    unsafe_allow_html=True,
                )
            with bpc3:
                if st.button("Next →", key="bm_next", use_container_width=True, disabled=(bm_page >= n_bm_pages - 1)):
                    st.session_state._bookmarks_page = bm_page + 1
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: audience_analysis  — research trending topics for Telugu audience
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "audience_analysis":

    aa_title_col, aa_back_col = st.columns([5, 1])
    with aa_title_col:
        _screen_header("📈", "Audience Intelligence",
                       "Trending topics scored 1–10 for Telugu YouTube Shorts audience")
    with aa_back_col:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True, key="aa_back"):
            st.session_state.phase = "input"
            st.rerun()

    if not st.session_state.get("_audience_analysis_done"):
        token_tracker.reset()
        with st.status("🔍 Analysing YouTube trends for Telugu audience…", expanded=True) as aa_status:
            st.write("Searching YouTube Shorts for trending Telugu content…")
            st.write("Checking channel content pattern…")
            st.write("Scoring 19 content areas by current audience interest…")
            try:
                aa_result = analyze_audience_interest_node({})
                st.session_state._aa_result = aa_result.get("audience_insights", {})
                st.session_state._audience_analysis_done = True
                aa_status.update(label="✅ Analysis complete!", state="complete")
            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.code(traceback.format_exc(), language="python")
                aa_status.update(label="❌ Analysis failed", state="error")

    insights = st.session_state.get("_aa_result") or load_audience_insights()

    if insights:
        analyzed_at = insights.get("analyzed_at", "")[:10]
        st.success(f"Analysis from {analyzed_at} · Top areas updated · Future idea generation will prioritise these areas.")

        # ── Top areas ranked bar chart ────────────────────────────────────────
        area_scores = insights.get("area_scores", {})
        if area_scores:
            st.markdown("#### 🏆 Area Interest Scores (1–10 for Telugu audience)")
            sorted_areas = sorted(area_scores.items(), key=lambda x: x[1], reverse=True)
            for area, score in sorted_areas:
                bar_fill = int(score * 10)
                color    = "#22c55e" if score >= 8 else "#F59E0B" if score >= 6 else "#6b7280"
                st.markdown(
                    f"<div style='display:flex;align-items:center;margin-bottom:5px'>"
                    f"<div style='width:195px;font-size:0.84em;color:#d0d0d0'>{area}</div>"
                    f"<div style='flex:1;background:rgba(255,255,255,0.1);border-radius:5px;height:13px;margin:0 10px'>"
                    f"<div style='width:{bar_fill}%;background:{color};height:13px;border-radius:5px'></div></div>"
                    f"<div style='width:24px;font-size:0.85em;color:{color}'><b>{score}</b></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Top areas + trending angles ───────────────────────────────────────
        col_left, col_right = st.columns(2)
        with col_left:
            top_areas = insights.get("top_areas", [])
            if top_areas:
                st.markdown("#### 🎯 Top 5 Priority Areas")
                for i, area in enumerate(top_areas[:5], 1):
                    score = area_scores.get(area, "?")
                    st.markdown(f"**{i}.** {area} &nbsp; `{score}/10`", unsafe_allow_html=True)

            channel_pattern = insights.get("channel_pattern", "")
            if channel_pattern:
                st.markdown("#### 📺 Channel Pattern")
                st.info(channel_pattern)

        with col_right:
            trending_angles = insights.get("trending_angles", [])
            if trending_angles:
                st.markdown("#### 🔥 Trending Angles Right Now")
                for angle in trending_angles[:8]:
                    st.markdown(f"• {angle}")

            audience_insights_text = insights.get("audience_insights", "")
            if audience_insights_text:
                st.markdown("#### 👥 Audience Insights")
                st.info(audience_insights_text)

        st.divider()
        search_evidence = insights.get("search_evidence", "")
        if search_evidence:
            with st.expander("🔎 Search evidence", expanded=False):
                st.markdown(search_evidence)

        st.success(
            "✅ Saved to VectorDB — idea generator and research agent will now "
            "prioritise the top areas in every future run."
        )

    re_col, back_col, _ = st.columns([1, 1, 2])
    with re_col:
        if st.button("🔄 Re-run Analysis", use_container_width=True):
            st.session_state._audience_analysis_done = False
            st.rerun()
    with back_col:
        if st.button("🏠 Back to Home", use_container_width=True, type="primary"):
            st.session_state.phase = "input"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: chat  — conversational interface routing to all agents
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "chat":

    # ── Header ────────────────────────────────────────────────────────────────
    ch_title, ch_clear, ch_back = st.columns([4, 1, 1])
    with ch_title:
        _screen_header("💬", "Creative Studio",
                       "Chat · research · check saturation · build storyboards — all in one place")
    with ch_clear:
        st.markdown("<div style='padding-top:8px'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear", use_container_width=True, help="Clear chat history"):
            st.session_state._chat_history     = []
            st.session_state._chat_context_idea = None
            st.rerun()
    with ch_back:
        st.markdown("<div style='padding-top:8px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True):
            st.session_state.phase = "input"
            st.rerun()

    chat_history    = st.session_state.setdefault("_chat_history", [])
    context_idea    = st.session_state.get("_chat_context_idea")

    _CHAT_AREA_ICONS = {
        "history": "🏛️", "science": "🔬", "latest science news": "📰",
        "geography": "🌍", "nature": "🌿", "space": "🚀",
        "psychology": "🧠", "psychology studies": "🧠", "technology": "💻",
        "interesting events": "⚡", "inspiring people": "🌟",
        "archaeology news": "🏺", "aviation reports": "✈️",
        "patent filings": "📜", "government reports": "🏛",
        "research papers": "📄", "world cultures": "🌐",
        "india government reports": "🇮🇳", "hidden mechanisms": "⚙️",
    }

    # ── Helper: render a single assistant message by type ─────────────────────
    def _render_chat_msg(msg: dict, msg_idx: int):
        msg_type = msg.get("type", "text")

        if msg_type == "text":
            st.markdown(msg.get("content", ""))

        elif msg_type == "error":
            st.error(msg.get("content", ""))

        elif msg_type == "saturation":
            approved = msg.get("approved")
            topic    = msg.get("topic", "")
            reason   = msg.get("reason", "")
            if approved:
                st.success(f"✅ **{topic}** — LOW saturation, not heavily covered on YouTube!")
            elif approved is False:
                st.warning(f"⚠️ **{topic}** — heavily saturated on YouTube Shorts.")
            else:
                st.info(f"Saturation check for **{topic}** inconclusive.")
            if reason:
                st.caption(reason)
            if msg.get("idea"):
                idea = msg["idea"]
                with st.container(border=True):
                    st.markdown(f"**{idea.get('title','')}**")
                    st.markdown(idea.get("concept", ""))
                if st.button("Set as active idea", key=f"ch_sat_set_{msg_idx}"):
                    st.session_state._chat_context_idea = idea
                    st.rerun()

        elif msg_type == "ideas":
            st.markdown(msg.get("content", "Here are some ideas:"))
            ideas = msg.get("ideas", [])
            for j, idea in enumerate(ideas):
                area = idea.get("area", "").lower()
                icon = _CHAT_AREA_ICONS.get(area, "💡")
                sat  = idea.get("saturation_score")
                sat_badge = {1: "🟢", 2: "🟡"}.get(sat, "")
                with st.container(border=True):
                    tag = f"<span class='idea-area-tag'>{icon} {area.upper()}</span>"
                    if sat_badge:
                        tag += f"&nbsp;&nbsp;{sat_badge}"
                    st.markdown(tag, unsafe_allow_html=True)
                    st.markdown(f"**{idea.get('title','')}**")
                    st.markdown(idea.get("concept", ""))
                    if idea.get("viral_hook"):
                        st.markdown(f"<div class='hook-text'>🎣 {idea['viral_hook']}</div>",
                                    unsafe_allow_html=True)
                    ic1, ic2, ic3 = st.columns(3)
                    with ic1:
                        if st.button("⚡ Activate", key=f"ch_sel_{msg_idx}_{j}",
                                     use_container_width=True, help="Set as active idea"):
                            st.session_state._chat_context_idea = idea
                            st.rerun()
                    with ic2:
                        if st.button("🔖", key=f"ch_bk_{msg_idx}_{j}",
                                     use_container_width=True, help="Bookmark"):
                            save_to_bookmarks(idea, source="chat")
                            st.toast(f"Bookmarked: {idea.get('title','')[:40]}", icon="🔖")
                    with ic3:
                        if st.button("🎬", key=f"ch_sb_{msg_idx}_{j}",
                                     use_container_width=True, help="Go to storyboard"):
                            st.session_state.best_idea = {
                                "title": idea.get("title",""),
                                "concept": idea.get("concept",""),
                                "why_best": "Selected from chat",
                            }
                            st.session_state.topic         = idea.get("title","")
                            st.session_state.image_prompts = []
                            st.session_state.telugu_script = {}
                            st.session_state.phase         = "generating"
                            st.rerun()

        elif msg_type == "research":
            st.markdown(msg.get("content", "Here are research angles:"))
            ideas = msg.get("ideas", [])
            for j, idea in enumerate(ideas):
                with st.container(border=True):
                    st.markdown(f"**{idea.get('title','')}**")
                    st.markdown(idea.get("concept", ""))
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        if st.button("⚡ Activate", key=f"ch_rsel_{msg_idx}_{j}",
                                     use_container_width=True):
                            st.session_state._chat_context_idea = idea
                            st.rerun()
                    with rc2:
                        if st.button("🔖", key=f"ch_rbk_{msg_idx}_{j}",
                                     use_container_width=True, help="Bookmark"):
                            save_to_bookmarks(idea, source="chat")
                            st.toast(f"Bookmarked: {idea.get('title','')[:40]}", icon="🔖")

        elif msg_type == "storyboard":
            idea   = msg.get("idea", {})
            scenes = msg.get("scenes", [])
            script = msg.get("script", {})
            st.markdown(msg.get("content", "Storyboard ready!"))
            if idea:
                st.markdown(f"**💡 {idea.get('title','')}**")
            if scenes:
                st.markdown(f"*{len(scenes)} scenes · {sum(s.get('duration_seconds',5) for s in scenes)}s total*")
                for s in scenes[:4]:
                    st.markdown(f"**Scene {s.get('scene_number')} [{s.get('duration_seconds')}s]** — {s.get('narration','')}")
                if len(scenes) > 4:
                    st.caption(f"…+{len(scenes)-4} more scenes")
            if script and not script.get("raw"):
                st.markdown(f"**Hook:** {script.get('hook','')}")
            sbc1, sbc2 = st.columns(2)
            with sbc1:
                if st.button("📺 View Full Storyboard", key=f"ch_view_{msg_idx}",
                             use_container_width=True, type="primary"):
                    st.session_state.best_idea    = idea
                    st.session_state.image_prompts = scenes
                    st.session_state.telugu_script = script
                    st.session_state.phase         = "done"
                    st.rerun()
            with sbc2:
                if st.button("🔖 Bookmark idea", key=f"ch_sbk_{msg_idx}",
                             use_container_width=True):
                    save_to_bookmarks(idea, source="chat")
                    st.toast("Bookmarked!", icon="🔖")

        elif msg_type == "audience":
            insights = msg.get("insights", {})
            st.markdown(msg.get("content", "Audience analysis complete!"))
            top5 = insights.get("top_areas", [])[:5]
            if top5:
                st.markdown("**Top areas:** " + " · ".join(f"`{a}`" for a in top5))
            summary = insights.get("audience_insights", "")
            if summary:
                st.info(summary)

    # ── Render existing history ───────────────────────────────────────────────
    if not chat_history:
        st.info(
            "👋 Ask me anything!\n\n"
            "Try: **\"check saturation of [your idea]\"** · **\"research quantum tunneling\"** · "
            "**\"generate ideas about psychology\"** · **\"create storyboard for PAPI lights\"** · "
            "or just ask me a question about content strategy."
        )

    for i, msg in enumerate(chat_history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_chat_msg(msg, i)

    # ── Active idea context bar ───────────────────────────────────────────────
    if context_idea:
        st.divider()
        with st.container(border=True):
            st.markdown(
                f"<span style='color:#7C3AED;font-size:0.85em'>⚡ Active idea</span>&nbsp;&nbsp;"
                f"**{context_idea.get('title','')}**",
                unsafe_allow_html=True,
            )
            st.caption(context_idea.get("concept","")[:120] + "…" if len(context_idea.get("concept","")) > 120 else context_idea.get("concept",""))
            ab1, ab2, ab3, ab4, ab5 = st.columns(5)
            with ab1:
                if st.button("🎬 Storyboard", key="ctx_sb", use_container_width=True, type="primary"):
                    st.session_state.best_idea = {
                        "title":   context_idea.get("title",""),
                        "concept": context_idea.get("concept",""),
                        "why_best": "Activated from chat",
                    }
                    st.session_state.topic         = context_idea.get("title","")
                    st.session_state.image_prompts = []
                    st.session_state.telugu_script = {}
                    st.session_state.phase         = "generating"
                    st.rerun()
            with ab2:
                if st.button("🔖 Bookmark", key="ctx_bk", use_container_width=True):
                    saved = save_to_bookmarks(context_idea, source="chat")
                    st.toast("Bookmarked!" if saved else "Already bookmarked", icon="🔖")
            with ab3:
                if st.button("📊 Saturation", key="ctx_sat", use_container_width=True,
                             help="Check YouTube saturation for this idea"):
                    st.session_state._chat_history.append({
                        "role": "user",
                        "content": f"Check saturation of: {context_idea.get('title','')}",
                    })
                    st.rerun()
            with ab4:
                if st.button("🔬 Research", key="ctx_res", use_container_width=True,
                             help="Research this topic further"):
                    st.session_state._chat_history.append({
                        "role": "user",
                        "content": f"Research this topic further: {context_idea.get('title','')}",
                    })
                    st.rerun()
            with ab5:
                if st.button("✖ Clear", key="ctx_clr", use_container_width=True,
                             help="Remove active idea"):
                    st.session_state._chat_context_idea = None
                    st.rerun()

    # ── Chat input + processing ───────────────────────────────────────────────
    if prompt := st.chat_input(
        "Check saturation, research a topic, generate ideas, create storyboard, or just chat…"
    ):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        chat_history.append({"role": "user", "content": prompt})

        # Detect intent
        with st.chat_message("assistant"):
            intent = detect_intent(prompt, chat_history, context_idea)
            action = intent.get("action", "chat")
            topic  = intent.get("topic") or (context_idea.get("title") if context_idea else None) or prompt

            response_msg: dict = {}

            # ── Route to agent ────────────────────────────────────────────────
            if action == "check_saturation":
                with st.status(f"📊 Checking YouTube saturation for: *{topic}*…", expanded=True) as cs:
                    st.write(f"Searching YouTube for '{topic}'…")
                    try:
                        sat_state = {
                            "input_topic":    topic,
                            "research_ideas": [{"title": topic, "concept": topic}],
                            "messages":       [],
                        }
                        sat_res  = freshness_check_agent_node(sat_state)
                        approved = sat_res.get("freshness_approved")
                        reason   = sat_res.get("rejection_reason", "")
                        best     = sat_res.get("best_idea") or {}
                        cs.update(label="✅ Done", state="complete")
                        response_msg = {
                            "role": "assistant", "type": "saturation",
                            "content": f"Saturation check for **{topic}** complete.",
                            "approved": approved, "reason": reason,
                            "topic": topic, "idea": best,
                        }
                        if best:
                            st.session_state._chat_context_idea = best
                    except Exception as e:
                        cs.update(label="❌ Error", state="error")
                        response_msg = {"role": "assistant", "type": "error", "content": str(e)}

            elif action == "research_topic":
                with st.status(f"🔬 Researching: *{topic}*…", expanded=True) as rs:
                    st.write(f"Finding viral angles on '{topic}'…")
                    try:
                        r_state = {
                            "input_topic": topic, "iteration": 0,
                            "rejection_reason": "", "messages": [],
                        }
                        r_res  = research_agent_node(r_state)
                        ideas  = r_res.get("research_ideas", [])
                        rs.update(label=f"✅ Found {len(ideas)} angles", state="complete")
                        content = f"Found **{len(ideas)} viral angles** on *{topic}*:"
                        response_msg = {
                            "role": "assistant", "type": "research",
                            "content": content, "ideas": ideas,
                        }
                        if ideas:
                            st.session_state._chat_context_idea = {**ideas[0], "why_best": "From research"}
                    except Exception as e:
                        rs.update(label="❌ Error", state="error")
                        response_msg = {"role": "assistant", "type": "error", "content": str(e)}

            elif action == "generate_ideas":
                with st.status("✨ Generating fresh ideas…", expanded=True) as gi:
                    st.write("Scanning areas and checking saturation…")
                    try:
                        gen_state = {"idea_gen_iteration": 0, "generated_ideas": [], "messages": []}
                        gen_res   = idea_generator_agent_node(gen_state)
                        new_ideas = gen_res.get("generated_ideas", [])
                        filtered, all_scored = bulk_saturation_filter(new_ideas)
                        update_history_saturation(all_scored)
                        show_ideas = filtered[:5] if filtered else new_ideas[:5]
                        gi.update(label=f"✅ {len(show_ideas)} fresh ideas ready", state="complete")
                        response_msg = {
                            "role": "assistant", "type": "ideas",
                            "content": f"Here are **{len(show_ideas)} fresh ideas**:",
                            "ideas": show_ideas,
                        }
                        if show_ideas:
                            st.session_state._chat_context_idea = show_ideas[0]
                    except Exception as e:
                        gi.update(label="❌ Error", state="error")
                        response_msg = {"role": "assistant", "type": "error", "content": str(e)}

            elif action == "generate_storyboard":
                idea_for_sb = context_idea or {"title": topic, "concept": topic}
                with st.status(f"🎬 Building storyboard for: *{idea_for_sb.get('title', topic)}*…",
                               expanded=True) as sb:
                    try:
                        sb_state = {
                            "input_topic": topic,
                            "best_idea":   idea_for_sb,
                            "messages":    [],
                        }
                        st.write("Planning scenes…")
                        ip_res = image_prompt_agent_node(sb_state)
                        scenes = ip_res.get("image_prompts", [])
                        sb_state["image_prompts"] = scenes

                        st.write("Writing Telugu script…")
                        sc_res = script_agent_node(sb_state)
                        script = sc_res.get("telugu_script", {})

                        # Store in session for "View Full" navigation
                        st.session_state.best_idea    = idea_for_sb
                        st.session_state.image_prompts = scenes
                        st.session_state.telugu_script = script

                        sb.update(label=f"✅ {len(scenes)} scenes ready", state="complete")
                        response_msg = {
                            "role": "assistant", "type": "storyboard",
                            "content": "Storyboard and Telugu script ready!",
                            "idea": idea_for_sb, "scenes": scenes, "script": script,
                        }
                    except Exception as e:
                        sb.update(label="❌ Error", state="error")
                        response_msg = {"role": "assistant", "type": "error", "content": str(e)}

            elif action == "analyze_audience":
                with st.status("📈 Analysing audience interest…", expanded=True) as aa:
                    try:
                        aa_res   = analyze_audience_interest_node({})
                        insights = aa_res.get("audience_insights", {})
                        aa.update(label="✅ Analysis saved", state="complete")
                        top5 = insights.get("top_areas", [])[:5]
                        response_msg = {
                            "role": "assistant", "type": "audience",
                            "content": f"Audience analysis complete! Top areas: {', '.join(top5)}",
                            "insights": insights,
                        }
                    except Exception as e:
                        aa.update(label="❌ Error", state="error")
                        response_msg = {"role": "assistant", "type": "error", "content": str(e)}

            else:  # "chat" — conversational fallback
                reply = chat_response(prompt, chat_history, context_idea)
                st.markdown(reply)
                response_msg = {"role": "assistant", "type": "text", "content": reply}

            # Render the response inline (only for non-text types — text already rendered above)
            if response_msg and response_msg.get("type") != "text":
                _render_chat_msg(response_msg, len(chat_history))

        chat_history.append(response_msg)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: idea_selection  — calls idea_generator_agent_node directly
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "idea_selection":

    # ── Auto-replace banner ───────────────────────────────────────────────────
    if st.session_state.get("_auto_replacing"):
        st.info("♻️ The selected idea was saturated on YouTube — generating a fresh replacement...")
        st.session_state._auto_replacing = False

    # ── Generate ideas if not already done (or "more" requested) ─────────────
    if not st.session_state.get("_ideas_ready"):

        TARGET_DISPLAY = 5   # always present exactly this many fresh ideas
        MAX_BATCHES    = 4   # max generation rounds to reach TARGET_DISPLAY

        token_tracker.reset()
        gen_state = {
            "idea_gen_iteration": st.session_state.get("idea_gen_iteration", 0),
            "generated_ideas":    st.session_state.get("all_displayed_ideas", []),
            "messages":           [],
            "forced_areas":       st.session_state.get("forced_areas"),
        }

        _forced = st.session_state.get("forced_areas")
        _status_label = (
            f"✨ Searching for **{_forced[0]}** ideas..."
            if _forced else "✨ Idea Generator searching the web..."
        )

        _LIVE_AREA_ICONS = {
            "history": "🏛️", "science": "🔬", "latest science news": "📰",
            "geography": "🌍", "nature": "🌿", "space": "🚀",
            "psychology": "🧠", "psychology studies": "🧠",
            "technology": "💻", "interesting events": "⚡",
            "inspiring people": "🌟", "archaeology news": "🏺",
            "aviation reports": "✈️", "research papers": "📄",
            "world cultures": "🌐", "hidden mechanisms": "⚙️",
            "surprising facts": "🤯", "unknown facts": "💡",
        }

        def _render_live_cards(ideas: list) -> None:
            """Render compact non-interactive previews as ideas arrive."""
            for live_idea in ideas:
                area = live_idea.get("area", "").lower()
                icon = _LIVE_AREA_ICONS.get(area, "💡")
                sat  = live_idea.get("saturation_score")
                sat_cls   = {1: "ic-sat-low", 2: "ic-sat-med"}.get(sat, "")
                sat_label = {1: "Low saturation", 2: "Medium saturation"}.get(sat, "")
                sat_html  = f'<span class="{sat_cls}">{sat_label}</span>' if sat_cls else ""
                concept   = live_idea.get("concept","").replace("<","&lt;").replace(">","&gt;")
                hook      = live_idea.get("viral_hook","").replace("<","&lt;").replace(">","&gt;")
                title_esc = live_idea.get("title","").replace("<","&lt;").replace(">","&gt;")
                with st.container(border=True):
                    st.markdown(
                        f'<div class="ic">'
                        f'<div class="ic-stripe"></div>'
                        f'<div><span class="ic-tag">{icon} {live_idea.get("area","").upper()}</span>{sat_html}</div>'
                        f'<div class="ic-title">{title_esc}</div>'
                        f'<div class="ic-concept">{concept}</div>'
                        f'<div class="ic-hook">🎣 {hook}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Placeholder created BEFORE the status block so cards appear above the progress log
        _live_placeholder = st.empty()

        with st.status(_status_label, expanded=True) as status:
            st.write(
                f"Focused on **{_forced[0]}** · filtering saturated topics..."
                if _forced else
                "Scanning history, science, psychology, technology, inspiring stories & more..."
            )

            fresh_ideas     = []   # ideas that passed saturation filter this cycle
            all_generated   = []   # everything generated (for dedup in subsequent batches)
            existing_titles = {i.get("title", "") for i in st.session_state.all_displayed_ideas}

            try:
                for batch in range(MAX_BATCHES):
                    if len(fresh_ideas) >= TARGET_DISPLAY:
                        break

                    still_need = TARGET_DISPLAY - len(fresh_ideas)
                    if batch > 0:
                        st.write(
                            f"↩️ {still_need} more fresh idea{'s' if still_need != 1 else ''} needed"
                            f" — generating another batch..."
                        )

                    result    = idea_generator_agent_node(gen_state)
                    new_ideas = result.get("generated_ideas", [])
                    st.session_state.idea_gen_iteration = result.get("idea_gen_iteration", batch + 1)
                    all_generated.extend(new_ideas)

                    if not new_ideas:
                        st.warning("No ideas returned — stopping.")
                        break

                    st.write(f"✓ {len(new_ideas)} candidates — checking saturation...")

                    # Check each idea individually and show it as soon as it passes
                    removed = 0
                    for idea in new_ideas:
                        if len(fresh_ideas) >= TARGET_DISPLAY:
                            break
                        scored = check_idea_saturation(idea)
                        update_history_saturation([scored])
                        t = scored.get("title", "")
                        if scored.get("saturation_score", 3) <= 2 and t not in existing_titles:
                            fresh_ideas.append(scored)
                            existing_titles.add(t)
                            st.session_state.all_displayed_ideas.append(scored)
                            # Show card immediately
                            with _live_placeholder.container():
                                _render_live_cards(fresh_ideas)
                            st.write(f"✅ Idea {len(fresh_ideas)}: {t[:60]}")
                        else:
                            removed += 1
                            st.write(f"🚫 Filtered: {t[:55]}...")

                    if removed:
                        st.write(f"↳ {removed} removed (saturated) · {len(fresh_ideas)}/{TARGET_DISPLAY} so far")

                    # Update gen_state so the next batch avoids everything generated so far
                    gen_state = {
                        "idea_gen_iteration": result.get("idea_gen_iteration", batch + 1),
                        "generated_ideas":    st.session_state.all_displayed_ideas + all_generated,
                        "messages":           [],
                        "forced_areas":       st.session_state.get("forced_areas"),
                    }

                # Fallback: if every batch was fully saturated, show best available
                if not fresh_ideas and all_generated:
                    st.warning("Could not find fresh ideas — showing best available.")
                    for idea in all_generated:
                        t = idea.get("title", "")
                        if t not in existing_titles and len(fresh_ideas) < TARGET_DISPLAY:
                            fresh_ideas.append(idea)
                            existing_titles.add(t)
                            st.session_state.all_displayed_ideas.append(idea)
                    with _live_placeholder.container():
                        _render_live_cards(fresh_ideas)

                n = len(fresh_ideas)
                status.update(
                    label=f"✅ {n} fresh idea{'s' if n != 1 else ''} ready!" if fresh_ideas else "⚠️ Failed",
                    state="complete",
                )

            except Exception as e:
                st.error(f"Error generating ideas: {e}")
                st.code(traceback.format_exc(), language="python")
                status.update(label="⚠️ Failed", state="error")

        # Clear the live preview — full interactive cards rendered below take over
        _live_placeholder.empty()
        st.session_state._ideas_ready = True

    # ── Show idea cards ───────────────────────────────────────────────────────
    display_ideas = st.session_state.get("all_displayed_ideas", [])

    if not display_ideas:
        st.error("Ideas could not be generated. Please try again.")
        if st.button("🔄 Retry"):
            st.session_state._ideas_ready = False
            st.rerun()
    else:
        total = len(display_ideas)
        _screen_header("✨", "Choose Your Idea",
                       "Fresh ideas — saturation-checked against YouTube · pick one to turn into a viral Short")
        st.caption(f"{total} idea{'s' if total != 1 else ''} generated — pick one, blacklist unwanted ones, or generate 5 more.")

        AREA_ICONS = {
            "history": "🏛️", "science": "🔬", "latest news": "📰",
            "latest science news": "📰", "geography": "🌍",
            "nature": "🌿", "space": "🚀",
            "psychology": "🧠", "psychology studies": "🧠",
            "technology": "💻", "interesting events": "⚡",
            "inspiring people": "🌟", "archaeology news": "🏺",
            "aviation reports": "✈️", "patent filings": "📜",
            "government reports": "🏛", "research papers": "📄",
            "world cultures": "🌐", "india government reports": "🇮🇳",
            "hidden mechanisms": "⚙️",
        }

        loved_titles      = {i.get("title", "") for i in st.session_state.get("_loved_this_session", [])}
        bookmarked_titles = st.session_state.get("_bookmarked_this_session", set())

        for i, idea in enumerate(display_ideas):
            area  = idea.get("area", "science").lower()
            icon  = AREA_ICONS.get(area, "💡")
            title = idea.get("title", "")
            col_card, col_sel, col_love, col_bk, col_ban = st.columns([5, 1, 1, 1, 1])

            with col_card:
                with st.container(border=True):
                    sat_score  = idea.get("saturation_score")
                    sat_reason = idea.get("saturation_reason", "")
                    sat_cls    = {1: "ic-sat-low", 2: "ic-sat-med"}.get(sat_score, "")
                    sat_lbl    = {1: "Low saturation", 2: "Med saturation"}.get(sat_score, "")
                    sat_html   = f'<span class="{sat_cls}">{sat_lbl}</span>' if sat_cls else ""
                    t_esc = title.replace("<","&lt;").replace(">","&gt;")
                    c_esc = idea.get("concept","").replace("<","&lt;").replace(">","&gt;")
                    h_esc = idea.get("viral_hook","").replace("<","&lt;").replace(">","&gt;")
                    r_esc = sat_reason.replace("<","&lt;").replace(">","&gt;") if sat_reason else ""
                    reason_html = f'<div style="font-size:0.78rem;color:var(--txt3);margin-top:8px">📊 {r_esc}</div>' if r_esc else ""
                    st.markdown(
                        f'<div class="ic">'
                        f'<div class="ic-stripe"></div>'
                        f'<div><span class="ic-tag">{icon} {idea.get("area","").upper()}</span>{sat_html}</div>'
                        f'<div class="ic-title">{t_esc}</div>'
                        f'<div class="ic-concept">{c_esc}</div>'
                        f'<div class="ic-hook">🎣 {h_esc}</div>'
                        f'{reason_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with col_sel:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("Select", key=f"pick_{i}", type="primary", use_container_width=True):
                    selected = display_ideas[i]
                    # Ideas from the generator already passed bulk_saturation_filter —
                    # skip research, freshness check, and approval; go straight to generating.
                    st.session_state.selected_idea = selected
                    st.session_state.best_idea     = {
                        "title":   selected.get("title", ""),
                        "concept": selected.get("concept", ""),
                        "why_best": (
                            f"User selected · saturation: "
                            f"{'Low' if selected.get('saturation_score') == 1 else 'Medium'}"
                        ),
                    }
                    st.session_state.topic        = f"{selected['title']} — {selected['concept']}"
                    st.session_state._ideas_ready = True
                    st.session_state.phase        = "generating"
                    st.rerun()
            with col_love:
                st.markdown("<br><br>", unsafe_allow_html=True)
                already_loved = title in loved_titles
                love_label    = "❤️" if already_loved else "🤍"
                love_help     = "Loved! Future ideas will follow this style." if already_loved else "Love It — use this style for future ideas"
                if st.button(love_label, key=f"love_{i}", use_container_width=True, help=love_help,
                             disabled=already_loved):
                    save_to_favorites(display_ideas[i])
                    if "_loved_this_session" not in st.session_state:
                        st.session_state._loved_this_session = []
                    st.session_state._loved_this_session.append(display_ideas[i])
                    st.session_state._ideas_ready = True  # prevent re-generation on rerun
                    st.rerun()
            with col_bk:
                st.markdown("<br><br>", unsafe_allow_html=True)
                already_bookmarked = title in bookmarked_titles
                if st.button("📌" if already_bookmarked else "🔖",
                             key=f"bk_{i}", use_container_width=True,
                             help="Already bookmarked" if already_bookmarked else "Bookmark for later",
                             disabled=already_bookmarked):
                    saved = save_to_bookmarks(display_ideas[i], source="idea_selection")
                    if saved:
                        st.session_state._bookmarked_this_session = bookmarked_titles | {title}
                        bookmarked_titles = st.session_state._bookmarked_this_session
                        st.toast(f"Bookmarked: {title[:50]}", icon="🔖")
                    else:
                        st.toast("Already bookmarked!", icon="📌")
                    st.session_state._ideas_ready = True
                    st.rerun()
            with col_ban:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("🚫", key=f"ban_{i}", use_container_width=True,
                             help="Blacklist — never suggest this idea again"):
                    save_to_blacklist(display_ideas[i])
                    st.session_state.all_displayed_ideas = [
                        x for x in st.session_state.all_displayed_ideas
                        if x.get("title") != title
                    ]
                    st.session_state._ideas_ready = True  # prevent re-generation on rerun
                    st.rerun()

        st.markdown("---")
        col_more, _ = st.columns([1, 3])
        with col_more:
            if st.button("🔄  Generate 5 More Ideas", use_container_width=True):
                st.session_state._ideas_ready = False
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: saturation_check  — standalone freshness check, no pipeline continuation
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "saturation_check":

    topic_display = st.session_state.topic
    _screen_header("📊", "Market Intelligence",
                   f"Checking YouTube saturation for: {topic_display}")

    if not st.session_state.get("_sat_check_done"):
        token_tracker.reset()
        agent_state = {
            "input_topic":    topic_display,
            "research_ideas": [{"title": topic_display, "concept": topic_display}],
            "messages":       [],
        }
        with st.status("🔍 Analysing YouTube saturation (5 searches)…", expanded=True) as status:
            st.write(f"Extracting core concept from **{topic_display}**…")
            st.write("Generating 5 diverse search angles and querying YouTube…")
            try:
                result   = freshness_check_agent_node(agent_state)
                st.session_state._sat_check_result = result
                st.session_state._sat_check_done   = True
                _rpt     = result.get("saturation_report", {}) or {}
                approved = result.get("freshness_approved", False)
                _lbl     = _rpt.get("score", "HIGH" if not approved else "LOW")
                status.update(
                    label=f"✅ Done — {_lbl} saturation ({_rpt.get('total_results', '?')} results across {_rpt.get('total_searches', '?')} searches)",
                    state="complete",
                )
            except Exception as e:
                st.error(f"Saturation check error: {e}")
                st.code(traceback.format_exc(), language="python")
                status.update(label="❌ Error during check", state="error")

    result   = st.session_state.get("_sat_check_result", {})
    approved = result.get("freshness_approved")
    reason   = result.get("rejection_reason", "")
    best     = result.get("best_idea") or {}
    _rpt     = result.get("saturation_report") or {}

    # ── Verdict banner ────────────────────────────────────────────────────────
    _score_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
    _score_label  = _rpt.get("score", "")
    _score_icon   = _score_colors.get(_score_label, "⚪")
    if approved is True:
        st.success(f"{_score_icon} **{_score_label} saturation** — {_rpt.get('reasoning', best.get('why_best', ''))}")
    elif approved is False:
        st.error(f"{_score_icon} **{_score_label} saturation** — {_rpt.get('reasoning', reason)}")

    # ── Detailed evidence report ──────────────────────────────────────────────
    if _rpt:
        st.markdown("---")
        st.markdown("#### 🔍 Search Evidence")

        _core = _rpt.get("core_concept", "")
        if _core:
            st.markdown(f"**Concept understood as:** {_core}")

        _searches = _rpt.get("searches", [])
        _total    = _rpt.get("total_results", 0)
        _n_s      = _rpt.get("total_searches", len(_searches))

        # Summary metric row
        _ms1, _ms2, _ms3 = st.columns(3)
        _ms1.metric("Searches Run", str(_n_s))
        _ms2.metric("Total Results Found", str(_total))
        _ms3.metric("Saturation Score", _score_label or "—")

        # Per-search breakdown
        st.markdown("**Results per search query:**")
        for _s in _searches:
            _q     = _s.get("query", "")
            _cnt   = _s.get("results_found", 0)
            _vids  = _s.get("videos", [])
            _bar   = "█" * _cnt + "░" * max(0, 5 - _cnt)
            with st.expander(f"`{_q}` — **{_cnt}** result(s)  {_bar}", expanded=False):
                if _vids:
                    for _v in _vids:
                        st.markdown(f"- {_v}")
                else:
                    st.caption("No results found for this query.")

        # Competitors and gap
        _comps = _rpt.get("competitors", [])
        _gap   = _rpt.get("content_gap", "")
        if _comps:
            st.markdown("**Most directly competing videos found:**")
            for _c in _comps:
                st.markdown(f"- {_c}")
        if _gap:
            st.markdown(f"**Content gap / your angle:** {_gap}")

    st.divider()
    col_back, col_use, _ = st.columns([1, 1, 2])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.phase             = "input"
            st.session_state._sat_check_done   = False
            st.session_state._sat_check_result = {}
            st.rerun()
    with col_use:
        if st.button("🚀 Use This Idea", type="primary", use_container_width=True,
                     help="Proceed to research and storyboard with this idea"):
            st.session_state.use_idea_generator = False
            st.session_state._sat_check_done    = False
            st.session_state._sat_check_result  = {}
            st.session_state.phase              = "research"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: research
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "research":

    selected_idea = st.session_state.get("selected_idea") or {}
    topic_display = selected_idea.get("title") or st.session_state.topic

    # ── Fast path: idea came from generator → only freshness check ────────────
    if st.session_state.get("_skip_research"):

        _screen_header("🔍", "Saturation Check",
                       f"Checking how crowded YouTube is for: {topic_display}")

        # Run freshness check only if not already done
        if st.session_state.get("freshness_approved") is None:
            token_tracker.reset()
            agent_state = {
                "input_topic":    st.session_state.topic,
                "research_ideas": [selected_idea],
                "messages":       [],
            }
            with st.status("📊 Saturation Check — 5 searches…", expanded=True) as f_status:
                st.write(f"Extracting concept from **{topic_display}**, generating search angles…")
                try:
                    f_result = freshness_check_agent_node(agent_state)
                    best     = f_result.get("best_idea")
                    approved = f_result.get("freshness_approved", False)
                    reason   = f_result.get("rejection_reason", "")
                    _rpt_i   = f_result.get("saturation_report", {}) or {}
                    st.session_state.best_idea            = best
                    st.session_state.freshness_approved   = approved
                    st.session_state.rejection_reason     = reason
                    st.session_state._inline_sat_report   = _rpt_i
                    _il = _rpt_i.get("score", "HIGH" if not approved else "LOW")
                    f_status.update(
                        label=f"{'✅' if approved else '⚠️'} {_il} saturation · "
                              f"{_rpt_i.get('total_results','?')} results across "
                              f"{_rpt_i.get('total_searches','?')} searches",
                        state="complete",
                    )
                except Exception as e:
                    st.error(f"Freshness check error: {e}")
                    f_status.update(label="❌ Error", state="error")

        # Show saturation report if available
        _ir = st.session_state.get("_inline_sat_report") or {}
        if _ir:
            _sc = _ir.get("score", "")
            _sc_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(_sc, "⚪")
            _mc1, _mc2, _mc3 = st.columns(3)
            _mc1.metric("Searches Run", str(_ir.get("total_searches", "?")))
            _mc2.metric("Results Found", str(_ir.get("total_results", "?")))
            _mc3.metric("Saturation", f"{_sc_icon} {_sc}")
            if _ir.get("reasoning"):
                st.caption(_ir["reasoning"])
            with st.expander("🔍 See all search queries & results", expanded=False):
                _core_i = _ir.get("core_concept", "")
                if _core_i:
                    st.markdown(f"**Concept:** {_core_i}")
                for _si in _ir.get("searches", []):
                    _cnt_i = _si.get("results_found", 0)
                    _bar_i = "█" * _cnt_i + "░" * max(0, 5 - _cnt_i)
                    st.markdown(f"**`{_si.get('query','')}`** — {_cnt_i} result(s) {_bar_i}")
                    for _vi in _si.get("videos", []):
                        st.markdown(f"  - {_vi}")
                _gap_i = _ir.get("content_gap", "")
                if _gap_i:
                    st.markdown(f"**Your angle:** {_gap_i}")

        # Show result and let user decide
        if st.session_state.get("freshness_approved"):
            st.session_state._skip_research = False
            st.session_state.phase = "approval"
            st.rerun()

        elif st.session_state.get("freshness_approved") is None:
            # Error occurred — give user a way out
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Retry Check", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("← Back to Ideas", use_container_width=True):
                    st.session_state._skip_research     = False
                    st.session_state.freshness_approved = None
                    st.session_state.best_idea          = None
                    st.session_state._ideas_ready       = True
                    st.session_state.phase              = "idea_selection"
                    st.rerun()

        elif st.session_state.get("freshness_approved") is False:
            reason = st.session_state.get("rejection_reason", "")
            st.warning(
                f"⚠️ **{topic_display}** is already heavily covered on YouTube.\n\n"
                f"{reason}\n\n"
                f"Finding a fresh replacement idea automatically..."
            )
            # Blacklist the saturated idea so it's never suggested again
            save_to_blacklist(selected_idea)
            # Remove it from displayed ideas
            st.session_state.all_displayed_ideas = [
                x for x in st.session_state.get("all_displayed_ideas", [])
                if x.get("title") != selected_idea.get("title", "")
            ]
            # Reset freshness state and trigger new idea generation
            st.session_state._skip_research     = False
            st.session_state.freshness_approved = None
            st.session_state.best_idea          = None
            st.session_state._ideas_ready       = False   # triggers auto-generation
            st.session_state._auto_replacing    = True    # show banner in idea_selection
            st.session_state.phase              = "idea_selection"
            col1, _ = st.columns([1, 3])
            with col1:
                if st.button("Continue Anyway →", type="primary", use_container_width=True):
                    st.session_state.best_idea = {
                        "title":    selected_idea.get("title", ""),
                        "concept":  selected_idea.get("concept", ""),
                        "why_best": "User chose to proceed despite saturation",
                    }
                    st.session_state.freshness_approved = True
                    st.session_state._skip_research     = False
                    st.session_state._auto_replacing    = False
                    st.session_state.phase              = "approval"
                    st.rerun()
            st.rerun()

    # ── Full path: "Use My Idea" → research + freshness with retry ────────────
    else:
        _screen_header("🔬", "Research Lab",
                       f"Finding the most viral angles for: {topic_display}")

        token_tracker.reset()
        agent_state = {
            "input_topic":      st.session_state.topic,
            "research_ideas":   [],
            "freshness_approved": None,
            "best_idea":        None,
            "rejection_reason": "",
            "iteration":        st.session_state.get("iteration", 0),
            "messages":         [],
        }

        col1, col2 = st.columns(2)
        found_best = False

        for attempt in range(MAX_ITERATIONS):
            with col1:
                label = f"🔬 Research Agent (attempt {attempt + 1})" if attempt else "🔬 Research Agent"
                with st.status(label, expanded=(attempt == 0)) as r_status:
                    st.write(f"Searching for concepts about: {topic_display[:60]}")
                    try:
                        r_result = research_agent_node(agent_state)
                        agent_state.update(r_result)
                        ideas = r_result.get("research_ideas", [])
                        st.session_state.research_ideas = ideas
                        st.session_state.iteration      = agent_state.get("iteration", attempt + 1)
                        st.write(f"Found {len(ideas)} ideas")
                        for idx, idea in enumerate(ideas, 1):
                            st.markdown(f"**{idx}. {idea.get('title','')}**  \n{idea.get('concept','')}")
                        r_status.update(label=f"✅ {label}", state="complete")
                    except Exception as e:
                        st.error(f"Research error: {e}")
                        r_status.update(label=f"❌ {label}", state="error")
                        break

            with col2:
                label2 = f"📊 Freshness Check (attempt {attempt + 1})" if attempt else "📊 Freshness Check Agent"
                with st.status(label2, expanded=(attempt == 0)) as f_status:
                    st.write("Checking YouTube saturation...")
                    try:
                        f_result = freshness_check_agent_node(agent_state)
                        agent_state.update(f_result)
                        best   = f_result.get("best_idea")
                        reason = f_result.get("rejection_reason", "")
                        st.session_state.best_idea          = best
                        st.session_state.freshness_approved = f_result.get("freshness_approved")
                        st.session_state.rejection_reason   = reason
                        if best:
                            st.write(f"Selected: **{best.get('title','')}**")
                            st.caption(best.get("why_best", ""))
                            f_status.update(label=f"✅ {label2}", state="complete")
                            found_best = True
                        else:
                            st.warning(f"All saturated — {reason}")
                            f_status.update(label=f"🔄 {label2} — retrying", state="complete")
                    except Exception as e:
                        st.error(f"Freshness check error: {e}")
                        f_status.update(label=f"❌ {label2}", state="error")
                        break

            if found_best:
                break
            agent_state["iteration"] = attempt + 1

        # Fallback: if every attempt was saturated, use the first idea from the last batch
        if not found_best:
            last_ideas = st.session_state.get("research_ideas", [])
            if last_ideas:
                st.session_state.best_idea = {
                    **last_ideas[0],
                    "why_best": "All angles were saturated on YouTube — showing best available.",
                }
                st.session_state.freshness_approved = True
            else:
                st.error("Could not find any ideas. Please try a different topic.")
                st.stop()

        st.session_state.phase = "approval"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: approval
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "approval":

    best = st.session_state.best_idea or {}
    _screen_header("✋", "Creative Review",
                   "The AI found the freshest angle — approve to generate your storyboard & script, or reject to try again")
    st.info("Review the selected idea below. Approve to proceed or reject to find something fresher.")

    with st.container(border=True):
        st.markdown(f"## 💡 {best.get('title', 'No title')}")
        st.markdown(best.get("concept", ""))
        if best.get("why_best"):
            st.caption(f"Why chosen: {best.get('why_best')}")

    col_a, col_r, col_bk, _ = st.columns([1, 1, 1, 2])
    with col_a:
        if st.button("✅ Approve & Generate", type="primary", use_container_width=True):
            st.session_state.approval_answer = "yes"
            st.session_state.phase           = "generating"
            st.rerun()
    with col_r:
        if st.button("❌ Reject — try again", use_container_width=True):
            # Reset research state and go back to research phase
            st.session_state.approval_answer   = "no"
            st.session_state.best_idea         = None
            st.session_state.freshness_approved = None
            st.session_state.research_ideas    = []
            st.session_state.rejection_reason  = "User rejected the idea — please find something fresher."
            st.session_state.iteration         = 0
            st.session_state.phase             = "research"
            st.rerun()
    with col_bk:
        _appr_title    = best.get("title", "")
        _appr_bkd      = _appr_title in st.session_state.get("_bookmarked_this_session", set())
        if st.button("📌 Bookmarked" if _appr_bkd else "🔖 Bookmark",
                     use_container_width=True,
                     help="Already bookmarked" if _appr_bkd else "Bookmark this idea for later",
                     disabled=_appr_bkd):
            saved = save_to_bookmarks(best, source="research")
            if saved:
                st.session_state._bookmarked_this_session = (
                    st.session_state.get("_bookmarked_this_session", set()) | {_appr_title}
                )
                st.toast(f"Bookmarked: {_appr_title[:50]}", icon="🔖")
            else:
                st.toast("Already bookmarked!", icon="📌")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: generating  — image prompts → images → script
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "generating":

    best_idea = st.session_state.best_idea or {}
    _screen_header("⚙️", "Production Pipeline",
                   f"Generating storyboard & Telugu script for: {best_idea.get('title', '')}")

    token_tracker.reset()
    agent_state = {
        "input_topic": st.session_state.topic,
        "best_idea":   best_idea,
        "messages":    [],
    }

    # ── Image Prompt Agent ────────────────────────────────────────────────────
    with st.status("🎨 Image Prompt Agent", expanded=True) as status:
        st.write("Planning scenes and image prompts...")
        try:
            result = image_prompt_agent_node(agent_state)
            scenes = result.get("image_prompts", [])
            st.session_state.image_prompts = scenes
            agent_state["image_prompts"] = scenes
            st.write(f"{len(scenes)} scenes planned")
            for s in scenes:
                st.markdown(
                    f"**Scene {s.get('scene_number')} [{s.get('duration_seconds')}s]** — "
                    f"{s.get('narration','')}"
                )
            status.update(label="✅ Image Prompt Agent", state="complete")
        except Exception as e:
            st.error(f"Image prompt error: {e}")
            status.update(label="❌ Image Prompt Agent", state="error")

    # ── Telugu Script Agent ───────────────────────────────────────────────────
    with st.status("📝 Telugu Script Agent", expanded=True) as status:
        st.write("Writing Telugu narration script...")
        try:
            result = script_agent_node(agent_state)
            script = result.get("telugu_script", {})
            st.session_state.telugu_script = script
            st.write("Telugu script ready")
            status.update(label="✅ Telugu Script Agent", state="complete")
        except Exception as e:
            st.error(f"Script error: {e}")
            status.update(label="❌ Telugu Script Agent", state="error")
            script = {}

    # ── Save files ────────────────────────────────────────────────────────────
    scenes = st.session_state.image_prompts
    script = st.session_state.telugu_script

    if best_idea and scenes:
        slug    = "".join(c if c.isalnum() else "_" for c in best_idea.get("title","short").lower())[:30]
        run_dir = OUTPUT_BASE / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        _save_outputs(run_dir, best_idea, scenes, script)

    st.session_state.phase = "done"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: done
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "done":

    best   = st.session_state.best_idea or {}
    scenes = st.session_state.image_prompts
    script = st.session_state.telugu_script

    _screen_header("🏆", "Your Short is Ready!",
                   "Storyboard + Telugu script generated — download and create your viral Short")

    with st.container(border=True):
        st.markdown(f"## 🏆 {best.get('title','')}")
        st.markdown(best.get("concept", ""))
        st.caption(f"Completed in {st.session_state.iteration} iteration(s)")

    st.divider()

    left, right = st.columns([1.2, 1.3])

    with left:
        st.markdown("#### 🎨 Storyboard")
        for s in scenes:
            with st.expander(
                f"Scene {s.get('scene_number')} · {s.get('duration_seconds')}s — {s.get('narration','')[:40]}"
            ):
                st.markdown(f"**Narration:** {s.get('narration','')}")
                st.markdown("**Image prompt:**")
                st.caption(s.get("image_prompt",""))

    with right:
        st.markdown("#### 📝 Telugu Script")
        if script and not script.get("raw"):
            scenes_with_tr = [
                sc for sc in script.get("scenes", [])
                if sc.get("transliteration", "").strip()
            ]
            if scenes_with_tr:
                with st.container(border=True):
                    st.markdown(f"**{script.get('title','')}**")
                    st.divider()
                    for sc in scenes_with_tr:
                        st.markdown(f"**{sc.get('scene_number')}.** {sc.get('transliteration','').strip()}")
            else:
                st.info("No transliteration available.")
        elif script and script.get("raw"):
            st.text(script["raw"])
        else:
            st.info("Script not available.")

    st.divider()

    col_dl, col_reset = st.columns([3, 1])
    if best:
        slug    = "".join(c if c.isalnum() else "_" for c in best.get("title","short").lower())[:30]
        run_dir = OUTPUT_BASE / slug
        _dl_files = [
            ("telugu_script.txt", "⬇️  Telugu Script"),
            ("storyboard.txt",    "⬇️  Storyboard"),
        ]
        for fname, label in _dl_files:
            p = run_dir / fname
            if p.exists():
                with col_dl:
                    st.download_button(label, data=p.read_text(encoding="utf-8"),
                                       file_name=fname, mime="text/plain")
    with col_reset:
        if st.button("🔁 Start Over", type="primary", use_container_width=True):
            reset()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: channel_analysis  — YouTube channel performance breakdown by area
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "channel_analysis":

    _has_yt_key = bool(os.getenv("YOUTUBE_API_KEY"))

    ca_title_col, ca_back_col = st.columns([5, 1])
    with ca_title_col:
        _screen_header("📺", "My Channel Analysis",
                       f"Performance breakdown of {get_channel_handle()} Shorts by content area")
    with ca_back_col:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True, key="ca_back"):
            st.session_state.phase = "input"
            st.rerun()

    if not _has_yt_key:
        st.error(
            "**YouTube API key not configured.**\n\n"
            "**YouTube API key not configured.**\n\n"
            "Go to **⚙️ Settings** to add your YouTube API key, or add `YOUTUBE_API_KEY=AIza...` to your `.env` file.\n\n"
            "**How to get one (free):**\n"
            "1. Go to Google Cloud Console (console.cloud.google.com)\n"
            "2. Create or select a project → Enable **YouTube Data API v3**\n"
            "3. Credentials → **Create credentials** → **API key**"
        )
        st.stop()

    # ── Run analysis (use cache unless forced refresh) ────────────────────────
    if not st.session_state.get("_ca_done"):
        cached = load_ca()
        if cached and not st.session_state.get("_force_ca_refresh"):
            st.session_state._ca_result        = cached
            st.session_state._ca_done          = True
            st.session_state._force_ca_refresh = False
        else:
            with st.status(f"📡 Fetching Shorts from {get_channel_handle()}…", expanded=True) as ca_status:
                st.write("Connecting to YouTube Data API v3…")
                st.write("Fetching all uploads from your channel…")
                st.write("Filtering to Shorts (≤ 3 min or #shorts tag)…")
                st.write("Categorising Shorts by content area with AI…")
                try:
                    r = analyze_channel_node({})
                    err = r.get("error")
                    if err == "NO_API_KEY":
                        ca_status.update(label="❌ No YouTube API key", state="error")
                        st.error("YouTube API key not set. Go to ⚙️ Settings to add it.")
                        st.stop()
                    elif err:
                        ca_status.update(label="❌ Error", state="error")
                        st.error(f"Analysis failed: {err}")
                        st.stop()
                    st.session_state._ca_result        = r.get("channel_analysis")
                    st.session_state._ca_done          = True
                    st.session_state._force_ca_refresh = False
                    ca_status.update(label="✅ Analysis complete!", state="complete")
                except Exception as e:
                    ca_status.update(label="❌ Failed", state="error")
                    st.error(f"Analysis failed: {e}")
                    st.code(traceback.format_exc(), language="python")
                    st.stop()

    ca = st.session_state.get("_ca_result") or {}
    if not ca:
        st.warning("No data available. Click Re-Analyse to run a fresh analysis.")
        st.stop()

    ch_stats = ca.get("channel_stats", {})
    area_st  = ca.get("area_stats", {})
    top_ovr  = ca.get("top_overall", [])
    total_sh = ca.get("total_shorts", 0)
    analyzed = ca.get("analyzed_at", "")[:10]

    # ── Channel overview stat cards ───────────────────────────────────────────
    st.markdown(
        f"<div style='color:#888;font-size:0.8em;margin-bottom:12px'>"
        f"Analysis from {analyzed} · "
        f"<a href='{get_channel_url()}' target='_blank' "
        f"style='color:#FF0000'>{get_channel_handle()}</a></div>",
        unsafe_allow_html=True,
    )

    def _ca_stat_card(col, icon, label, value, color):
        with col:
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.05);border:1px solid {color}30;"
                f"border-radius:12px;padding:16px;text-align:center'>"
                f"<div style='font-size:1.8em'>{icon}</div>"
                f"<div style='font-size:1.5em;font-weight:800;color:{color};margin:4px 0'>{value}</div>"
                f"<div style='font-size:0.78em;color:#9ca3af'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    def _ca_fmt(n):
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}K"
        return str(n)

    ov1, ov2, ov3, ov4 = st.columns(4)
    _ca_stat_card(ov1, "📹", "Total Shorts Analysed", str(total_sh), "#FF0000")
    _ca_stat_card(ov2, "👁️", "Channel Total Views",
                  _ca_fmt(ch_stats.get("total_view_count", 0)), "#3B82F6")
    _ca_stat_card(ov3, "🔔", "Subscribers",
                  _ca_fmt(ch_stats.get("subscriber_count", 0)), "#22C55E")
    _ca_stat_card(ov4, "🎞️", "All Videos on Channel",
                  str(ch_stats.get("total_video_count", 0)), "#F59E0B")

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    # ── Per-area breakdown ────────────────────────────────────────────────────
    _CA_ICONS = {
        "history": "🏛️", "science": "🔬", "latest science news": "📰",
        "geography": "🌍", "nature": "🌿", "space": "🚀",
        "psychology": "🧠", "psychology studies": "🧠", "technology": "💻",
        "interesting events": "⚡", "inspiring people": "🌟",
        "archaeology news": "🏺", "aviation reports": "✈️",
        "research papers": "📄", "world cultures": "🌐",
        "hidden mechanisms": "⚙️",
    }

    if area_st:
        st.markdown("#### 📊 Content Area Breakdown")
        st.caption("Sorted by average views per Short — which areas resonate most with your audience?")

        sorted_areas = sorted(
            area_st.items(), key=lambda x: x[1].get("avg_views", 0), reverse=True
        )
        max_avg = sorted_areas[0][1].get("avg_views", 1) if sorted_areas else 1

        for area, d in sorted_areas:
            icon      = _CA_ICONS.get(area, "📌")
            count     = d.get("count", 0)
            avg_views = d.get("avg_views", 0)
            avg_likes = d.get("avg_likes", 0)
            bar_pct   = int(avg_views / max_avg * 100) if max_avg else 0
            color     = "#22c55e" if bar_pct >= 70 else "#F59E0B" if bar_pct >= 35 else "#6b7280"

            st.markdown(
                f"<div style='display:flex;align-items:center;margin-bottom:7px;gap:10px'>"
                f"<div style='width:26px;font-size:1.1em'>{icon}</div>"
                f"<div style='width:172px;font-size:0.84em;color:#d0d0d0'>{area}</div>"
                f"<div style='width:34px;font-size:0.78em;color:#9ca3af;text-align:right'>"
                f"{count}×</div>"
                f"<div style='flex:1;background:rgba(255,255,255,0.1);border-radius:5px;height:13px'>"
                f"<div style='width:{bar_pct}%;background:{color};height:13px;border-radius:5px'>"
                f"</div></div>"
                f"<div style='width:80px;font-size:0.82em;color:{color};text-align:right'>"
                f"avg {_ca_fmt(avg_views)} 👁️</div>"
                f"<div style='width:65px;font-size:0.82em;color:#9ca3af;text-align:right'>"
                f"{_ca_fmt(avg_likes)} ❤️</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Top 3 videos for this area in an expander
            top_vids = d.get("top_videos", [])
            if top_vids:
                with st.expander(f"Top Shorts in {area}", expanded=False):
                    for v in top_vids[:3]:
                        st.markdown(
                            f"• **{v.get('title','')}** — "
                            f"{_ca_fmt(v.get('view_count',0))} views · "
                            f"{_ca_fmt(v.get('like_count',0))} likes · "
                            f"[▶ Watch](https://www.youtube.com/shorts/{v.get('id','')})"
                        )

        st.divider()

    # ── Top 10 overall ────────────────────────────────────────────────────────
    if top_ovr:
        st.markdown("#### 🏆 Top 10 Performing Shorts Overall")
        for rank, v in enumerate(top_ovr, 1):
            area = v.get("area", "")
            icon = _CA_ICONS.get(area, "📌")
            with st.container(border=True):
                tc1, tc2 = st.columns([5, 1])
                with tc1:
                    t_esc = v.get("title","").replace("<","&lt;").replace(">","&gt;")
                    vid_id = v.get("id","")
                    pub_date = v.get("published_at","")[:10]
                    st.markdown(
                        f'<div style="margin-bottom:6px">'
                        f'<span style="font-family:Syne,sans-serif;font-size:0.78rem;font-weight:700;color:var(--a,#FF0000);letter-spacing:0.06em">#{rank}</span>'
                        f'&ensp;<span style="font-family:Syne,sans-serif;font-weight:700;font-size:1.0rem;color:#ECEDF5">{t_esc}</span>'
                        f'</div>'
                        f'<span class="ic-tag">{icon} {area.upper()}</span>'
                        f'&ensp;<span style="color:var(--txt3,#48485E);font-size:0.73rem">{pub_date}</span>'
                        + (f'&ensp;<a href="https://www.youtube.com/shorts/{vid_id}" target="_blank" style="color:#FF0000;font-size:0.79rem;text-decoration:none">▶ Watch</a>' if vid_id else ""),
                        unsafe_allow_html=True,
                    )
                with tc2:
                    st.markdown(
                        f"<div style='text-align:right;padding-top:4px'>"
                        f"<div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#60A5FA'>{_ca_fmt(v.get('view_count',0))}</div>"
                        f"<div style='font-size:0.73rem;color:var(--txt3,#48485E);margin-top:1px'>views</div>"
                        f"<div style='font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#F43F5E;margin-top:6px'>{_ca_fmt(v.get('like_count',0))}</div>"
                        f"<div style='font-size:0.73rem;color:var(--txt3,#48485E);margin-top:1px'>likes</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    st.divider()
    ca_re_col, ca_back_col2, _ = st.columns([1, 1, 2])
    with ca_re_col:
        if st.button("🔄 Re-Analyse", use_container_width=True):
            st.session_state._ca_done          = False
            st.session_state._force_ca_refresh = True
            st.rerun()
    with ca_back_col2:
        if st.button("🏠 Back to Home", use_container_width=True, type="primary"):
            st.session_state.phase = "input"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: ready_boards  — browse all saved storyboards in generated_images/
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "ready_boards":

    import os as _os
    from datetime import datetime as _dt

    _RB_AREA_ICONS = {
        "history": "🏛️", "science": "🔬", "latest science news": "📰",
        "geography": "🌍", "nature": "🌿", "space": "🚀",
        "psychology": "🧠", "psychology studies": "🧠", "technology": "💻",
        "interesting events": "⚡", "inspiring people": "🌟",
        "archaeology news": "🏺", "aviation reports": "✈️",
        "research papers": "📄", "world cultures": "🌐",
        "hidden mechanisms": "⚙️",
    }

    rb_title_col, rb_back_col = st.columns([5, 1])
    with rb_title_col:
        _screen_header("📁", "Ready Boards",
                       "All your generated storyboards — pick one to view or re-download")
    with rb_back_col:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True, key="rb_back"):
            st.session_state.phase = "input"
            st.rerun()

    # ── Collect all saved boards ──────────────────────────────────────────────
    _boards = []
    if OUTPUT_BASE.exists():
        for _d in OUTPUT_BASE.iterdir():
            _sb_path = _d / "storyboard.json"
            if not (_d.is_dir() and _sb_path.exists()):
                continue
            try:
                _data    = json.loads(_sb_path.read_text(encoding="utf-8"))
                _idea    = _data.get("idea", {})
                _scenes  = _data.get("scenes", [])
                _mtime   = _sb_path.stat().st_mtime
                _n_imgs  = len(list(_d.glob("scene_*.png")))
                _boards.append({
                    "folder":   _d,
                    "idea":     _idea,
                    "scenes":   _scenes,
                    "mtime":    _mtime,
                    "date_str": _dt.fromtimestamp(_mtime).strftime("%Y-%m-%d"),
                    "n_images": _n_imgs,
                })
            except Exception:
                pass
    _boards.sort(key=lambda x: x["mtime"], reverse=True)  # newest first

    if not _boards:
        st.info(
            "No storyboards generated yet.\n\n"
            "Use **Generate Ideas** or **Use My Idea** from the home screen to create your first one!"
        )
    else:
        st.caption(f"{len(_boards)} storyboard{'s' if len(_boards) != 1 else ''} saved")
        st.divider()

        for _i, _b in enumerate(_boards):
            _idea   = _b["idea"]
            _scenes = _b["scenes"]
            _title  = _idea.get("title", "Untitled")
            _concept = _idea.get("concept", "")
            _area   = _idea.get("area", "").lower().strip()
            _icon   = _RB_AREA_ICONS.get(_area, "🎬")
            _dur    = sum(s.get("duration_seconds", 5) for s in _scenes)
            _n_sc   = len(_scenes)
            _n_img  = _b["n_images"]

            _col_card, _col_view, _col_del = st.columns([5, 1, 1])

            with _col_card:
                with st.container(border=True):
                    import html as _html_mod
                    _t_esc  = _html_mod.escape(_title)
                    _c_esc  = _html_mod.escape(_concept[:140] + ("…" if len(_concept) > 140 else "")) if _concept else ""
                    _first_narr = _scenes[0].get("narration", "") if _scenes else ""
                    _nr_esc = _html_mod.escape(_first_narr[:100] + ("…" if len(_first_narr) > 100 else "")) if _first_narr else ""
                    _meta_html = (
                        f"<span style='color:var(--txt3);font-size:0.78em'>{_b['date_str']}</span>"
                        f"&ensp;<span style='color:var(--a);font-size:0.78em'>"
                        f"{_n_sc} scenes · {_dur}s"
                        + (f" · {_n_img} 🖼️" if _n_img else "")
                        + "</span>"
                    )
                    _hook_html = f'<div class="ic-hook">🎬 {_nr_esc}</div>' if _nr_esc else ""
                    st.markdown(
                        f'<div class="ic">'
                        f'<div class="ic-stripe"></div>'
                        f'<div><span class="ic-tag">{_icon} {_area.upper()}</span>&ensp;{_meta_html}</div>'
                        f'<div class="ic-title">{_t_esc}</div>'
                        + (f'<div class="ic-concept">{_c_esc}</div>' if _c_esc else "")
                        + _hook_html
                        + f'</div>',
                        unsafe_allow_html=True,
                    )

            with _col_view:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("▶ View", key=f"rb_view_{_i}",
                             type="primary", use_container_width=True,
                             help="Open full storyboard"):
                    # Load script if available
                    _sc_path = _b["folder"] / "telugu_script.json"
                    _script  = {}
                    if _sc_path.exists():
                        try:
                            _script = json.loads(_sc_path.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    st.session_state.best_idea     = _idea
                    st.session_state.image_prompts = _scenes
                    st.session_state.telugu_script = _script
                    st.session_state._rb_folder    = str(_b["folder"])
                    st.session_state.phase         = "board_view"
                    st.rerun()

            with _col_del:
                st.markdown("<br><br>", unsafe_allow_html=True)
                _del_confirm_key = f"rb_del_confirm_{_i}"
                if st.session_state.get(_del_confirm_key):
                    st.warning("Delete?")
                    _dc1, _dc2 = st.columns(2)
                    with _dc1:
                        if st.button("Yes", key=f"rb_del_yes_{_i}", use_container_width=True):
                            shutil.rmtree(_b["folder"], ignore_errors=True)
                            st.session_state.pop(_del_confirm_key, None)
                            st.rerun()
                    with _dc2:
                        if st.button("No", key=f"rb_del_no_{_i}", use_container_width=True):
                            st.session_state.pop(_del_confirm_key, None)
                            st.rerun()
                else:
                    if st.button("🗑 Delete", key=f"rb_del_{_i}",
                                 use_container_width=True,
                                 help="Permanently delete this board"):
                        st.session_state[_del_confirm_key] = True
                        st.rerun()

        st.divider()
        if st.button("🏠 Back to Home", type="primary"):
            st.session_state.phase = "input"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: board_view  — full read-only storyboard view for a saved board
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "board_view":

    best   = st.session_state.best_idea or {}
    scenes = st.session_state.image_prompts or []
    script = st.session_state.telugu_script or {}

    # Reconstruct run_dir from stored path or title slug
    _bv_folder = st.session_state.get("_rb_folder")
    if _bv_folder:
        _bv_dir = Path(_bv_folder)
    else:
        _slug   = "".join(c if c.isalnum() else "_" for c in best.get("title","short").lower())[:30]
        _bv_dir = OUTPUT_BASE / _slug

    # ── Header ────────────────────────────────────────────────────────────────
    bv_title_col, bv_back_col = st.columns([5, 1])
    with bv_title_col:
        _bv_dur = sum(s.get("duration_seconds", 5) for s in scenes)
        _screen_header("🎬", best.get("title", "Storyboard"),
                       f"{len(scenes)} scenes · {_bv_dur}s · saved storyboard")
    with bv_back_col:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Boards", use_container_width=True, key="bv_back"):
            st.session_state.phase = "ready_boards"
            st.rerun()

    # ── Concept card ──────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(f"### 💡 {best.get('title','')}")
        st.markdown(best.get("concept",""))
        if best.get("why_best"):
            st.caption(f"Why chosen: {best['why_best']}")

    st.divider()

    # ── Scene images grid (if PNG files exist) ────────────────────────────────
    _bv_imgs = sorted(_bv_dir.glob("scene_*.png")) if _bv_dir.exists() else []
    if _bv_imgs:
        st.markdown("#### 🖼️ Scene Images")
        _bv_img_cols = st.columns(4)
        for _bv_i, _bv_img in enumerate(_bv_imgs):
            with _bv_img_cols[_bv_i % 4]:
                st.image(str(_bv_img), caption=f"Scene {_bv_i+1}", use_container_width=True)
        st.divider()

    # ── Two-column: storyboard + script ──────────────────────────────────────
    left_col, right_col = st.columns([1.2, 1.3])

    with left_col:
        st.markdown("#### 🎨 Storyboard")
        for s in scenes:
            with st.expander(
                f"Scene {s.get('scene_number')} · {s.get('duration_seconds')}s"
                f" — {s.get('narration','')[:40]}"
            ):
                st.markdown(f"**Narration:** {s.get('narration','')}")
                st.markdown("**Image prompt:**")
                st.caption(s.get("image_prompt",""))

    with right_col:
        st.markdown("#### 📝 Telugu Script")
        if script and not script.get("raw"):
            _bv_sc_tr = [sc for sc in script.get("scenes",[]) if sc.get("transliteration","").strip()]
            if _bv_sc_tr:
                with st.container(border=True):
                    st.markdown(f"**{script.get('title','')}**")
                    st.divider()
                    for sc in _bv_sc_tr:
                        st.markdown(f"**{sc.get('scene_number')}.** {sc.get('transliteration','').strip()}")
            else:
                st.info("No transliteration available.")
        elif script and script.get("raw"):
            st.text(script["raw"])
        else:
            st.info("Script not available.")

    st.divider()

    # ── Download buttons ──────────────────────────────────────────────────────
    dl_col, use_col, back_col2 = st.columns([2, 1, 1])
    with dl_col:
        _bv_dl_files = [
            ("telugu_script.txt",   "⬇️ Telugu Script"),
            ("storyboard.txt",      "⬇️ Storyboard"),
        ]
        _bv_dl_cols = st.columns(len(_bv_dl_files))
        for _ci, (fname, label) in enumerate(_bv_dl_files):
            _p = _bv_dir / fname
            if _p.exists():
                with _bv_dl_cols[_ci]:
                    st.download_button(label, data=_p.read_text(encoding="utf-8"),
                                       file_name=fname, mime="text/plain")
    with use_col:
        if st.button("🔁 Regenerate", use_container_width=True,
                     help="Use this idea to regenerate the storyboard"):
            st.session_state.topic     = best.get("title","")
            st.session_state.best_idea = {
                "title":   best.get("title",""),
                "concept": best.get("concept",""),
                "why_best": "Regenerated from saved board",
            }
            st.session_state.image_prompts = []
            st.session_state.telugu_script = {}
            st.session_state.phase         = "generating"
            st.rerun()
    with back_col2:
        if st.button("🏠 Home", type="primary", use_container_width=True):
            st.session_state.phase = "input"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: subtitles_upload  — upload Telugu MP3 and transcribe
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "subtitles_upload":
    import tempfile as _tempfile

    _su_title_col, _su_back_col = st.columns([5, 1])
    with _su_title_col:
        _screen_header("🎙️", "Generate Subtitles",
                       "Upload a Telugu MP3 — AI will transcribe and translate to English")
    with _su_back_col:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True, key="sub_back"):
            st.session_state.phase = "input"
            st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    def _run_whisper_timing(audio_path: str):
        """Run Whisper; return (model, eng_segments, tel_segments, audio_duration_secs).

        Uses condition_on_previous_text=False to prevent the early-stop loop
        that causes Whisper to process only the first ~20s of long Telugu audio.
        """
        import whisper as _w
        _m = _w.load_model("base")

        _audio_arr = _w.load_audio(audio_path)
        _duration  = len(_audio_arr) / 16000.0

        _common_opts = dict(
            language="te",
            verbose=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.3,
            compression_ratio_threshold=2.8,
            fp16=False,
        )
        _eng = _m.transcribe(audio_path, task="translate",   **_common_opts)
        _tel = _m.transcribe(audio_path, task="transcribe",  **_common_opts)
        return _m, _eng.get("segments", []), _tel.get("segments", []), _duration

    def _save_audio_tmp(uploaded_file) -> str:
        _ext = Path(uploaded_file.name).suffix or ".mp3"
        with _tempfile.NamedTemporaryFile(suffix=_ext, delete=False) as _tf:
            _tf.write(uploaded_file.read())
            return _tf.name

    _, _su_mid, _ = st.columns([1, 2, 1])
    with _su_mid:
        _tab_auto, _tab_script = st.tabs(["🎙️ Auto Transcribe", "📝 Paste Telugu Script"])

        # ── TAB 1: Auto Transcribe ────────────────────────────────────────────
        with _tab_auto:
            st.caption(
                "Whisper will detect Telugu speech segments, transcribe them and translate to English.  \n"
                "Works best for clear audio. If accuracy is poor, try the **Paste Script** tab instead."
            )
            _su_file = st.file_uploader(
                "Choose audio file", type=["mp3", "wav", "m4a", "ogg", "flac"],
                label_visibility="collapsed", key="su_file_auto",
            )
            if _su_file:
                st.audio(_su_file)
                st.markdown(f"**File:** `{_su_file.name}` · {_su_file.size // 1024:,} KB")
                if st.button("🔍 Transcribe & Translate", type="primary",
                             use_container_width=True, key="sub_go_auto"):
                    _tmp_path = _save_audio_tmp(_su_file)
                    with st.status("Transcribing Telugu audio…", expanded=True) as _su_status:
                        try:
                            st.write("Loading Whisper model (base)…")
                            _, _eng_segs, _tel_segs, _dur = _run_whisper_timing(_tmp_path)
                            st.write(f"Audio: {_dur:.1f}s · {len(_eng_segs)} segments detected")
                            _segments = []
                            for _idx, _eseg in enumerate(_eng_segs):
                                _tseg = _tel_segs[_idx] if _idx < len(_tel_segs) else {}
                                _segments.append({
                                    "id":      _idx + 1,
                                    "start":   _eseg["start"],
                                    "end":     _eseg["end"],
                                    "telugu":  _tseg.get("text", "").strip(),
                                    "english": _eseg["text"].strip(),
                                })
                            st.session_state._sub_segments   = _segments
                            st.session_state._sub_audio_name = Path(_su_file.name).stem
                            _su_status.update(
                                label=f"Done — {len(_segments)} segments across {_dur:.1f}s",
                                state="complete"
                            )
                        except Exception as _sue:
                            _su_status.update(label="Error", state="error")
                            st.error(f"Transcription failed: {_sue}")
                            st.code(__import__("traceback").format_exc())
                            st.stop()
                        finally:
                            Path(_tmp_path).unlink(missing_ok=True)
                    st.session_state.phase = "subtitles_review"
                    st.rerun()

        # ── TAB 2: Paste Script ───────────────────────────────────────────────
        with _tab_script:
            st.caption(
                "Paste the Telugu script written in English letters (e.g. *'mee kosam chala manchi'*).  \n"
                "Upload the audio for timing. The AI will align the script to the audio and translate."
            )
            _ps_file = st.file_uploader(
                "Choose audio file", type=["mp3", "wav", "m4a", "ogg", "flac"],
                label_visibility="collapsed", key="su_file_script",
            )
            if _ps_file:
                st.audio(_ps_file)
                st.markdown(f"**File:** `{_ps_file.name}` · {_ps_file.size // 1024:,} KB")

            _ps_script = st.text_area(
                "Telugu script (written in English / transliteration)",
                placeholder=(
                    "Paste your full script here, e.g:\n"
                    "Mee kosam chala manchi vishayalu share chestha.\n"
                    "Meeru idi chustey surprise avuthaaru..."
                ),
                height=180,
                key="ps_script_text",
            )

            if st.button("🤖 Align & Translate", type="primary",
                         use_container_width=True, key="sub_go_script",
                         disabled=not (_ps_file and _ps_script.strip())):
                _tmp_path = _save_audio_tmp(_ps_file)
                with st.status("Aligning script to audio…", expanded=True) as _ps_status:
                    try:
                        import re as _re

                        # Split user script into natural sentences
                        _sentences = [
                            s.strip()
                            for s in _re.split(r'(?<=[.!?])\s+|\n+', _ps_script.strip())
                            if s.strip()
                        ]
                        _n_sent = len(_sentences)

                        st.write("Loading Whisper model (base) for timing…")
                        _, _eng_segs, _, _audio_dur = _run_whisper_timing(_tmp_path)
                        _n_segs = len(_eng_segs)
                        _whisper_end = _eng_segs[-1]["end"] if _eng_segs else 0
                        _coverage = _whisper_end / _audio_dur if _audio_dur else 0
                        st.write(
                            f"Audio: {_audio_dur:.1f}s · "
                            f"Whisper covered {_whisper_end:.1f}s ({_coverage*100:.0f}%) · "
                            f"{_n_segs} segments · {_n_sent} script lines"
                        )

                        from ollama_client import get_haiku_llm as _get_haiku
                        from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM
                        _llm = _get_haiku(temperature=0)

                        if _coverage >= 0.80 and _n_segs >= _n_sent:
                            # ── Good coverage: LLM assigns Whisper segments to script lines ──
                            st.write("Good coverage — asking AI to assign segments to script lines…")
                            _seg_info = "\n".join(
                                f"  S{_i}: {_s['start']:.2f}s→{_s['end']:.2f}s"
                                f" | heard: \"{_s.get('text','').strip()[:60]}\""
                                for _i, _s in enumerate(_eng_segs)
                            )
                            _sent_info = "\n".join(
                                f"  Line {_j+1}: {_sent}"
                                for _j, _sent in enumerate(_sentences)
                            )
                            _sys = (
                                "You are a subtitle timing expert for Telugu audio.\n\n"
                                "You receive Whisper-detected audio segments (S0, S1, …) with ACCURATE timestamps, "
                                "and a Telugu script split into numbered lines.\n\n"
                                "Your ONLY job: decide which consecutive Whisper segments correspond to each script line.\n"
                                "Rules:\n"
                                "- Every Whisper segment must be assigned to exactly one line.\n"
                                "- Segments must stay in order (no reordering or skipping).\n"
                                "- Each line gets 1 or more consecutive segments.\n"
                                "- Also translate each script line to natural English.\n\n"
                                "Return ONLY valid JSON — no markdown, no explanation:\n"
                                '[\n'
                                '  {"line": 1, "segments": [0, 1], "telugu": "Line 1 text", "english": "English translation"},\n'
                                '  ...\n'
                                ']\n'
                                f"Return exactly {_n_sent} objects."
                            )
                            _usr = (
                                f"Whisper segments (S0–S{_n_segs-1}):\n{_seg_info}\n\n"
                                f"Script lines (Line 1–{_n_sent}):\n{_sent_info}\n\n"
                                f"Assign all {_n_segs} Whisper segments to the {_n_sent} lines and translate."
                            )
                            _resp = _llm.invoke([_SM(content=_sys), _HM(content=_usr)])
                            _raw  = _resp.content.strip()
                            if _raw.startswith("```"):
                                _raw = "\n".join(
                                    l for l in _raw.splitlines()
                                    if not l.strip().startswith("```")
                                ).strip()
                            _llm_assignments = json.loads(_raw)

                            _segments = []
                            for _asgn in _llm_assignments:
                                _grp_idxs = [x for x in _asgn.get("segments", []) if 0 <= x < _n_segs]
                                if not _grp_idxs:
                                    continue
                                _grp_segs = [_eng_segs[x] for x in _grp_idxs]
                                _segments.append({
                                    "id":      len(_segments) + 1,
                                    "start":   _grp_segs[0]["start"],
                                    "end":     _grp_segs[-1]["end"],
                                    "telugu":  _asgn.get("telugu", "").strip(),
                                    "english": _asgn.get("english", "").strip(),
                                })

                        else:
                            # ── Poor coverage: proportional timestamps across full audio ──
                            st.write(
                                f"Low Whisper coverage ({_coverage*100:.0f}%) — "
                                f"distributing {_n_sent} lines proportionally across {_audio_dur:.1f}s. "
                                "Asking AI to translate…"
                            )
                            _char_counts = [max(len(s), 1) for s in _sentences]
                            _total_chars  = sum(_char_counts)
                            _prop_segs = []
                            _cursor = 0.0
                            for _j, _sent in enumerate(_sentences):
                                _frac  = _char_counts[_j] / _total_chars
                                _start = _cursor
                                _end   = min(_cursor + _audio_dur * _frac, _audio_dur)
                                _prop_segs.append({"start": round(_start, 3), "end": round(_end, 3), "text": _sent})
                                _cursor = _end

                            _sys_t = (
                                "You are a Telugu translator. Translate each of the following Telugu lines "
                                "(written in English transliteration) into natural English. "
                                "Return ONLY a JSON array of strings — one English translation per line, "
                                "in the same order, no extra keys."
                            )
                            _usr_t = "\n".join(f"{_j+1}. {_s}" for _j, _s in enumerate(_sentences))
                            _resp_t = _llm.invoke([_SM(content=_sys_t), _HM(content=_usr_t)])
                            _raw_t  = _resp_t.content.strip()
                            if _raw_t.startswith("```"):
                                _raw_t = "\n".join(
                                    l for l in _raw_t.splitlines()
                                    if not l.strip().startswith("```")
                                ).strip()
                            _translations = json.loads(_raw_t)

                            _segments = []
                            for _j, _ps in enumerate(_prop_segs):
                                _eng_text = _translations[_j] if _j < len(_translations) else ""
                                _segments.append({
                                    "id":      _j + 1,
                                    "start":   _ps["start"],
                                    "end":     _ps["end"],
                                    "telugu":  _ps["text"],
                                    "english": str(_eng_text).strip(),
                                })

                        st.session_state._sub_segments   = _segments
                        st.session_state._sub_audio_name = Path(_ps_file.name).stem
                        _ps_status.update(
                            label=f"Done — {len(_segments)} subtitle entries across {_audio_dur:.1f}s",
                            state="complete"
                        )

                    except Exception as _pse:
                        _ps_status.update(label="Error", state="error")
                        st.error(f"Alignment failed: {_pse}")
                        st.code(__import__("traceback").format_exc())
                        st.stop()
                    finally:
                        Path(_tmp_path).unlink(missing_ok=True)

                st.session_state.phase = "subtitles_review"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: subtitles_review  — review & edit subtitles, export SRT
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "subtitles_review":

    def _fmt_srt_time(seconds: float) -> str:
        h  = int(seconds // 3600)
        m  = int((seconds % 3600) // 60)
        s  = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    _sr_segments  = st.session_state.get("_sub_segments", [])
    _sr_audio_name = st.session_state.get("_sub_audio_name", "subtitles")

    _sr_title_col, _sr_back_col = st.columns([5, 1])
    with _sr_title_col:
        _screen_header("📝", "Review Subtitles",
                       f"{len(_sr_segments)} segments — edit English text then save SRT")
    with _sr_back_col:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True, key="sr_back"):
            st.session_state.phase = "subtitles_upload"
            st.rerun()

    if not _sr_segments:
        st.warning("No segments found. Go back and re-upload.")
    else:
        st.caption(
            "The **Telugu** column shows what was heard. "
            "The **English subtitle** column is editable — fix translations before saving."
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        _edited = []
        for _seg in _sr_segments:
            _sid    = _seg["id"]
            _start  = _seg["start"]
            _end    = _seg["end"]
            _t      = _seg["telugu"]
            _e      = _seg["english"]
            _timing = f"{_fmt_srt_time(_start)} → {_fmt_srt_time(_end)}"

            with st.container(border=True):
                _rc1, _rc2, _rc3 = st.columns([1, 3, 4])
                with _rc1:
                    st.markdown(
                        f"<div style='font-size:1.1em;font-weight:700;color:#EC4899'>"
                        f"#{_sid}</div>"
                        f"<div style='font-size:0.72em;color:#9ca3af;font-family:monospace'>"
                        f"{_timing}</div>",
                        unsafe_allow_html=True,
                    )
                with _rc2:
                    st.markdown("**Telugu (original)**")
                    _tel_inner = _t if _t else "<em style='color:#555'>—</em>"
                    st.markdown(
                        f"<div style='background:rgba(255,255,255,0.04);border-radius:6px;"
                        f"padding:8px 10px;font-size:0.85em;color:#d0d0d0;min-height:52px'>"
                        f"{_tel_inner}</div>",
                        unsafe_allow_html=True,
                    )
                with _rc3:
                    st.markdown("**English subtitle** *(editable)*")
                    _eng_edit = st.text_area(
                        "eng", value=_e,
                        key=f"sr_eng_{_sid}",
                        label_visibility="collapsed",
                        height=72,
                    )
                    _edited.append({
                        "id":    _sid,
                        "start": _start,
                        "end":   _end,
                        "text":  _eng_edit,
                    })

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.divider()

        _srt_col, _vid_col, _home_col = st.columns([3, 2, 1])
        with _vid_col:
            if st.button("🎬  Generate Video", use_container_width=True, key="sr_to_video",
                         help="Create a styled MP4 subtitle video (Karaoke / Pop / Highlight / Word-by-word)"):
                st.session_state.phase = "subtitles_video"
                st.rerun()
        with _srt_col:
            if st.button("💾  Save SRT File", type="primary",
                         use_container_width=True, key="sr_save"):
                _srt_lines = []
                for _seg in _edited:
                    if not _seg["text"].strip():
                        continue
                    _srt_lines.append(str(_seg["id"]))
                    _srt_lines.append(
                        f"{_fmt_srt_time(_seg['start'])} --> {_fmt_srt_time(_seg['end'])}"
                    )
                    _srt_lines.append(_seg["text"].strip())
                    _srt_lines.append("")
                _srt_content = "\n".join(_srt_lines)

                _srt_dir = OUTPUT_BASE / "subtitles"
                _srt_dir.mkdir(parents=True, exist_ok=True)
                _srt_path = _srt_dir / f"{_sr_audio_name}.srt"
                _srt_path.write_text(_srt_content, encoding="utf-8")

                st.success(f"SRT saved to: `{_srt_path}`")
                st.download_button(
                    "⬇️  Download SRT",
                    data=_srt_content,
                    file_name=f"{_sr_audio_name}.srt",
                    mime="text/plain",
                    key="sr_dl",
                )

        with _home_col:
            if st.button("🏠 Home", type="primary",
                         use_container_width=True, key="sr_home"):
                st.session_state.phase = "input"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: subtitles_video  — generate styled subtitle MP4
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "subtitles_video":
    import subprocess as _svsp
    import numpy as _svnp
    from PIL import Image as _SvPIL, ImageDraw as _SvDraw, ImageFont as _SvFont

    _SV_W, _SV_H, _SV_FPS = 1080, 384, 30
    _SV_PAD = 48  # left/right margin

    # ── Font helpers ──────────────────────────────────────────────────────────

    def _sv_font(size, bold=False):
        _candidates = (
            ["/System/Library/Fonts/Supplemental/Impact.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
            if bold else
            ["/System/Library/Fonts/Supplemental/Arial.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]
        )
        for _p in _candidates:
            if Path(_p).exists():
                try:
                    return _SvFont.truetype(_p, size)
                except Exception:
                    pass
        return _SvFont.load_default()

    def _sv_word_w(draw, word, font):
        return draw.textbbox((0, 0), word, font=font)[2]

    def _sv_space_w(draw, font):
        return _sv_word_w(draw, " ", font)

    def _sv_font_h(draw, font):
        bb = draw.textbbox((0, 0), "Ag", font=font)
        return bb[3] - bb[1]

    def _sv_wrap(words, draw, font, max_w):
        """Returns list of (word_list, line_pixel_width)."""
        sw = _sv_space_w(draw, font)
        lines, widths = [[]], [0]
        for w in words:
            ww = _sv_word_w(draw, w, font)
            if lines[-1] and widths[-1] + sw + ww > max_w:
                lines.append([w]); widths.append(ww)
            else:
                if lines[-1]: widths[-1] += sw
                lines[-1].append(w); widths[-1] += ww
        return list(zip(lines, widths))

    def _sv_best_font(text, draw, bold=False):
        max_w = _SV_W - 2 * _SV_PAD
        for sz in [76, 62, 50, 40]:
            f = _sv_font(sz, bold)
            if len(_sv_wrap(text.split(), draw, f, max_w)) <= 2:
                return f, sz
        return _sv_font(40, bold), 40

    # ── Template renderers ────────────────────────────────────────────────────

    def _sv_karaoke(draw, text, progress):
        words = text.split()
        if not words: return
        max_w = _SV_W - 2 * _SV_PAD
        font, _ = _sv_best_font(text, draw, bold=True)
        lines = _sv_wrap(words, draw, font, max_w)
        fh  = _sv_font_h(draw, font)
        sw  = _sv_space_w(draw, font)
        lsp = int(fh * 0.38)
        total_h = len(lines) * fh + max(0, len(lines) - 1) * lsp
        y = (_SV_H - total_h) // 2
        cur_idx = int(progress * len(words))
        wi = 0
        for line_words, line_w in lines:
            x = (_SV_W - line_w) // 2
            for word in line_words:
                if wi < cur_idx:
                    col = (130, 130, 130)       # past  — gray
                elif wi == cur_idx:
                    col = (255, 215, 0)         # now   — gold
                else:
                    col = (255, 255, 255)       # next  — white
                draw.text((x, y), word, font=font, fill=col,
                          stroke_width=2, stroke_fill=(0, 0, 0))
                x += _sv_word_w(draw, word, font) + sw
                wi += 1
            y += fh + lsp

    def _sv_pop(draw, text, t_in):
        max_w = _SV_W - 2 * _SV_PAD
        _, base_sz = _sv_best_font(text, draw, bold=True)
        scale = min(1.0, 0.78 + 0.22 * (t_in / 0.18)) if t_in < 0.18 else 1.0
        font = _sv_font(max(18, int(base_sz * scale)), bold=True)
        lines = _sv_wrap(text.split(), draw, font, max_w)
        fh  = _sv_font_h(draw, font)
        lsp = int(fh * 0.38)
        total_h = len(lines) * fh + max(0, len(lines) - 1) * lsp
        y = (_SV_H - total_h) // 2
        for line_words, line_w in lines:
            x = (_SV_W - line_w) // 2
            draw.text((x, y), " ".join(line_words), font=font,
                      fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
            y += fh + lsp

    def _sv_highlight(draw, text, progress):
        words = text.split()
        if not words: return
        max_w = _SV_W - 2 * _SV_PAD
        font, _ = _sv_best_font(text, draw, bold=True)
        lines = _sv_wrap(words, draw, font, max_w)
        fh  = _sv_font_h(draw, font)
        sw  = _sv_space_w(draw, font)
        lsp = int(fh * 0.38)
        total_h = len(lines) * fh + max(0, len(lines) - 1) * lsp
        y = (_SV_H - total_h) // 2
        cur_idx = min(int(progress * len(words)), len(words) - 1)
        PAD_X, PAD_Y = 10, 5
        wi = 0
        for line_words, line_w in lines:
            x = (_SV_W - line_w) // 2
            for word in line_words:
                ww = _sv_word_w(draw, word, font)
                if wi == cur_idx:
                    draw.rounded_rectangle(
                        [x - PAD_X, y - PAD_Y, x + ww + PAD_X, y + fh + PAD_Y],
                        radius=8, fill=(255, 230, 0)
                    )
                    draw.text((x, y), word, font=font, fill=(0, 0, 0))
                else:
                    draw.text((x, y), word, font=font, fill=(255, 255, 255),
                              stroke_width=2, stroke_fill=(0, 0, 0))
                x += ww + sw
                wi += 1
            y += fh + lsp

    def _sv_wordbyword(draw, text, progress):
        words = text.split()
        if not words: return
        max_w = _SV_W - 2 * _SV_PAD
        n_vis = max(1, int(progress * len(words)) + 1)
        font, _ = _sv_best_font(text, draw, bold=True)
        lines = _sv_wrap(words[:n_vis], draw, font, max_w)
        fh  = _sv_font_h(draw, font)
        sw  = _sv_space_w(draw, font)
        lsp = int(fh * 0.38)
        total_h = len(lines) * fh + max(0, len(lines) - 1) * lsp
        y = (_SV_H - total_h) // 2
        wi = 0
        for line_words, line_w in lines:
            x = (_SV_W - line_w) // 2
            for word in line_words:
                col = (74, 222, 128) if wi == n_vis - 1 else (255, 255, 255)
                draw.text((x, y), word, font=font, fill=col,
                          stroke_width=2, stroke_fill=(0, 0, 0))
                x += _sv_word_w(draw, word, font) + sw
                wi += 1
            y += fh + lsp

    def _sv_frame(t, segs, template):
        img  = _SvPIL.new("RGB", (_SV_W, _SV_H), (0, 0, 0))
        draw = _SvDraw.Draw(img)
        seg  = next((s for s in segs if s["start"] <= t < s["end"]), None)
        if not seg:
            return img
        text = seg.get("english", "").strip()
        if not text:
            return img
        t_in     = t - seg["start"]
        dur      = max(seg["end"] - seg["start"], 0.001)
        progress = min(t_in / dur, 1.0)
        if template == "karaoke":
            _sv_karaoke(draw, text, progress)
        elif template == "pop":
            _sv_pop(draw, text, t_in)
        elif template == "highlight":
            _sv_highlight(draw, text, progress)
        elif template == "wordbyword":
            _sv_wordbyword(draw, text, progress)
        return img

    # ── UI ───────────────────────────────────────────────────────────────────

    _sv_hdr, _sv_bk = st.columns([5, 1])
    with _sv_hdr:
        _screen_header("🎬", "Subtitle Video Generator",
                       "Styled MP4 · 1080 × 384 · fits bottom 20% of a 9:16 Short")
    with _sv_bk:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True, key="sv_back"):
            st.session_state.phase = "subtitles_review"
            st.rerun()

    _sv_segs = st.session_state.get("_sub_segments", [])
    _sv_name = st.session_state.get("_sub_audio_name", "subtitles")

    if not _sv_segs:
        st.warning("No subtitle data. Go back and generate subtitles first.")
    else:
        _sv_total = max(s["end"] for s in _sv_segs)
        st.caption(
            f"{len(_sv_segs)} subtitle entries · {_sv_total:.1f}s · "
            f"output: 1080 × 384 px · {_SV_FPS} fps"
        )

        st.markdown("#### Choose Template")
        _tc1, _tc2, _tc3, _tc4 = st.columns(4)
        for _col, _icon, _name, _desc, _preview in [
            (_tc1, "🎤", "Karaoke",
             "Past=gray · Current=gold · Future=white",
             "<span style='color:#888'>This is </span>"
             "<span style='color:#FFD700;font-weight:700'>ka·ra·oke</span>"
             "<span style='color:#fff'> text</span>"),
            (_tc2, "💥", "Pop",
             "Scales in · bold stroke · TikTok style",
             "<span style='font-size:1.3em;font-weight:900;color:#fff;"
             "text-shadow:-2px 0 #000,2px 0 #000,0 -2px #000,0 2px #000'>POP!</span>"),
            (_tc3, "✨", "Highlight",
             "Yellow box · current line · YouTube style",
             "<span style='background:#FFE600;color:#000;padding:2px 8px;"
             "border-radius:4px;font-weight:700'>Highlight</span>"),
            (_tc4, "📝", "Word-by-word",
             "Words build up · latest=green",
             "<span style='color:#fff'>Word by </span>"
             "<span style='color:#4ADE80;font-weight:700'>word →</span>"),
        ]:
            with _col:
                st.markdown(
                    f"<div style='border:1px solid #333;border-radius:8px;padding:12px;"
                    f"background:#111;min-height:130px'>"
                    f"<div style='font-weight:700;margin-bottom:6px'>{_icon} {_name}</div>"
                    f"<div style='background:#000;padding:8px;border-radius:4px;"
                    f"font-family:Arial;text-align:center;min-height:36px'>{_preview}</div>"
                    f"<div style='font-size:0.72em;color:#888;margin-top:6px'>{_desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        _sv_tmpl = st.selectbox(
            "Select template",
            options=["karaoke", "pop", "highlight", "wordbyword"],
            format_func=lambda x: {
                "karaoke": "🎤 Karaoke", "pop": "💥 Pop",
                "highlight": "✨ Highlight", "wordbyword": "📝 Word-by-word",
            }[x],
            key="sv_template",
        )

        if st.button("🎬  Generate Video", type="primary",
                     use_container_width=True, key="sv_gen"):

            _ff_ok = _svsp.run(["ffmpeg", "-version"], capture_output=True)
            if _ff_ok.returncode != 0:
                st.error("ffmpeg not found. Install with: `brew install ffmpeg`")
                st.stop()

            _out_dir  = OUTPUT_BASE / "subtitles"
            _out_dir.mkdir(parents=True, exist_ok=True)
            _out_path = _out_dir / f"{_sv_name}_{_sv_tmpl}.mp4"
            _n_frames = int(_sv_total * _SV_FPS) + 1

            with st.status(
                f"Rendering {_sv_tmpl} video · {_sv_total:.1f}s · {_n_frames} frames…",
                expanded=True,
            ) as _sv_status:
                _sv_prog = st.progress(0.0)
                try:
                    _ff_cmd = [
                        "ffmpeg", "-y",
                        "-f", "rawvideo", "-vcodec", "rawvideo",
                        "-s", f"{_SV_W}x{_SV_H}",
                        "-pix_fmt", "rgb24",
                        "-r", str(_SV_FPS),
                        "-i", "-",
                        "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", "20",
                        str(_out_path),
                    ]
                    _ff_proc = _svsp.Popen(
                        _ff_cmd, stdin=_svsp.PIPE,
                        stdout=_svsp.PIPE, stderr=_svsp.PIPE,
                    )
                    for _fi in range(_n_frames):
                        _t = _fi / _SV_FPS
                        _img = _sv_frame(_t, _sv_segs, _sv_tmpl)
                        _ff_proc.stdin.write(_svnp.array(_img).tobytes())
                        if _fi % (_SV_FPS * 2) == 0:
                            _sv_prog.progress(min(_fi / _n_frames, 1.0))

                    _ff_proc.stdin.close()
                    _stderr = _ff_proc.stderr.read()
                    _ff_proc.wait()
                    if _ff_proc.returncode != 0:
                        raise RuntimeError(_stderr.decode(errors="replace"))

                    _sv_prog.progress(1.0)
                    _size_kb = _out_path.stat().st_size // 1024
                    _sv_status.update(
                        label=f"Done! {_size_kb:,} KB → {_out_path.name}",
                        state="complete",
                    )
                    st.video(str(_out_path))
                    with open(_out_path, "rb") as _vf:
                        st.download_button(
                            f"⬇️ Download {_sv_tmpl.title()} MP4",
                            data=_vf.read(),
                            file_name=_out_path.name,
                            mime="video/mp4",
                            key="sv_dl",
                        )

                except Exception as _sve:
                    _sv_status.update(label="Error generating video", state="error")
                    st.error(str(_sve))
                    st.code(__import__("traceback").format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: channel_performance  — recent vs high-performers analysis
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "channel_performance":

    _cp_hdr, _cp_bk = st.columns([5, 1])
    with _cp_hdr:
        _screen_header(
            "📈", "Channel Performance",
            f"Recent Shorts vs All-Time High Performers · {get_channel_handle()}",
        )
    with _cp_bk:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True, key="cp_back"):
            st.session_state.phase = "input"
            st.rerun()

    # ── Check prerequisites ───────────────────────────────────────────────────
    from vectordb import get_db as _get_vdb
    _cp_has_data = bool(_get_vdb().load_config("channel_analysis"))

    if not _cp_has_data:
        st.warning(
            "Individual video data is not yet available. "
            "Please run **My Channel Analysis** first (or re-run it if you ran it before this update)."
        )
        if st.button("📺  Go to Channel Analysis", type="primary", key="cp_goto_ca"):
            st.session_state._ca_done          = False
            st.session_state._ca_result        = None
            st.session_state._force_ca_refresh = True
            st.session_state.phase             = "channel_analysis"
            st.rerun()
        st.stop()

    # ── Load cached or run fresh ──────────────────────────────────────────────
    _cp_cached = load_performance()

    _cp_run_col, _cp_info_col = st.columns([2, 3])
    with _cp_run_col:
        _cp_btn_label = "🔄  Re-analyse Performance" if _cp_cached else "🚀  Analyse Performance"
        _cp_run = st.button(_cp_btn_label, type="primary" if not _cp_cached else "secondary",
                            use_container_width=True, key="cp_run")
    with _cp_info_col:
        if _cp_cached:
            st.caption(
                f"Last analysed: {_cp_cached['generated_at'][:10]}  ·  "
                f"Recent {_cp_cached.get('recent_metrics', {}).get('count', '?')} vs "
                f"Top {_cp_cached.get('top_metrics', {}).get('count', '?')} Shorts"
            )

    if _cp_run:
        with st.status("Analysing channel performance…", expanded=True) as _cp_status:
            try:
                st.write("Comparing recent vs high-performer metrics…")
                st.write("Running LLM analysis (Gemini 2.5 Flash)…")
                _cp_result = run_performance_analysis()
                st.session_state._cp_result = _cp_result
                _cp_status.update(label="Analysis complete!", state="complete")
                st.rerun()
            except Exception as _cpe:
                _cp_status.update(label="Error", state="error")
                st.error(str(_cpe))
                st.stop()

    _cp_data = st.session_state.get("_cp_result") or _cp_cached
    if not _cp_data:
        st.info("Click **Analyse Performance** to start.")
        st.stop()

    # ── Metrics comparison cards ──────────────────────────────────────────────
    _rm  = _cp_data.get("recent_metrics",  {})
    _tm  = _cp_data.get("top_metrics",     {})
    _gap = _cp_data.get("views_gap_pct",   0)

    def _cpfmt(n):
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}K"
        return str(n)

    st.markdown("### Performance Comparison")

    _mc1, _mc2 = st.columns(2)

    def _metric_card(col, title, icon, color, m: dict):
        top_area = next(iter(m.get("area_breakdown", {}).keys()), "—")
        hex_rgb = _hex_to_rgb(color)
        with col:
            st.markdown(
                f"<div style='border:1px solid rgba({hex_rgb},0.25);border-radius:16px;padding:20px;"
                f"background:linear-gradient(145deg,rgba({hex_rgb},0.07) 0%,rgba({hex_rgb},0.03) 100%);"
                f"box-shadow:0 1px 3px rgba(0,0,0,0.3)'>"
                f"<div style='font-size:0.76rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;"
                f"color:rgba({hex_rgb},0.9);margin-bottom:14px'>{icon}&ensp;{title}</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>"
                f"<div><div style='font-family:Syne,sans-serif;font-size:1.75em;font-weight:800;color:#ECEDF5'>{_cpfmt(m.get('avg_views',0))}</div>"
                f"<div style='color:var(--txt3,#48485E);font-size:0.76em;margin-top:2px'>avg views</div></div>"
                f"<div><div style='font-family:Syne,sans-serif;font-size:1.75em;font-weight:800;color:#ECEDF5'>{_cpfmt(m.get('avg_likes',0))}</div>"
                f"<div style='color:var(--txt3,#48485E);font-size:0.76em;margin-top:2px'>avg likes</div></div>"
                f"<div><div style='font-family:Syne,sans-serif;font-size:1.75em;font-weight:800;color:#ECEDF5'>{m.get('engagement_rate',0)}%</div>"
                f"<div style='color:var(--txt3,#48485E);font-size:0.76em;margin-top:2px'>engagement</div></div>"
                f"<div><div style='font-family:Syne,sans-serif;font-size:1.75em;font-weight:800;color:#ECEDF5'>{_cpfmt(m.get('max_views',0))}</div>"
                f"<div style='color:var(--txt3,#48485E);font-size:0.76em;margin-top:2px'>best video</div></div>"
                f"</div>"
                f"<div style='margin-top:12px;font-size:0.79rem;color:var(--txt2,#7A7A94)'>Top area: <b style='color:#ECEDF5'>{top_area}</b></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    _metric_card(_mc1, f"Recent {_rm.get('count','?')} Shorts", "🕐", "#3B82F6", _rm)
    _metric_card(_mc2, f"Top {_tm.get('count','?')} Performers", "🏆", "#10B981", _tm)

    if _gap:
        _gap_color = "#EF4444" if _gap > 0 else "#10B981"
        st.markdown(
            f"<div style='text-align:center;margin:12px 0;font-size:0.88rem;color:{_gap_color}'>"
            f"High performers average <b>{abs(_gap)}% {'more' if _gap > 0 else 'fewer'}</b> views than recent Shorts</div>",
            unsafe_allow_html=True,
        )

    # ── LLM analysis ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Analysis")
    st.markdown(_cp_data.get("analysis", "No analysis available."))

    # ── Video tables ──────────────────────────────────────────────────────────
    st.markdown("---")

    def _video_table(shorts: list[dict], label: str):
        if not shorts:
            return
        import pandas as _pd
        rows = []
        for s in shorts:
            rows.append({
                "Title":     s.get("title", ""),
                "Area":      s.get("area", ""),
                "Views":     s.get("view_count", 0),
                "Likes":     s.get("like_count", 0),
                "Comments":  s.get("comment_count", 0),
                "Published": s.get("published_at", "")[:10],
                "URL":       s.get("url", ""),
            })
        _df = _pd.DataFrame(rows)
        with st.expander(label, expanded=False):
            st.dataframe(
                _df[["Title", "Area", "Views", "Likes", "Comments", "Published"]],
                use_container_width=True,
                column_config={
                    "Views":    st.column_config.NumberColumn(format="%d"),
                    "Likes":    st.column_config.NumberColumn(format="%d"),
                    "Comments": st.column_config.NumberColumn(format="%d"),
                },
                hide_index=True,
            )

    _video_table(_cp_data.get("recent", []),        f"🕐 Recent {_rm.get('count','?')} Shorts")
    _video_table(_cp_data.get("top_performers", []), f"🏆 Top {_tm.get('count','?')} Performers")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE: settings — configure API keys, LLM models, channel, integrations
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "settings":

    _hdr_col, _bk_col = st.columns([5, 1])
    with _hdr_col:
        _screen_header("⚙️", "Settings",
                       "Configure API keys, LLM models, channel details and integrations")
    with _bk_col:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True, key="settings_back"):
            st.session_state.phase = "input"
            st.rerun()

    # ── Load current values (live env takes priority over file) ──────────────
    _env_file_vals = _load_settings_all()
    _section_icons = {
        "API Keys":          "🔑",
        "LLM Models":        "🤖",
        "Local LLM (Ollama)": "💻",
        "Integrations":      "🔌",
        "Channel":           "📺",
    }

    st.markdown(
        '<div class="info-note">'
        'Changes are saved to your <code>.env</code> file and applied immediately. '
        'Switching LLM provider takes full effect on the next app restart.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Build a form per section ──────────────────────────────────────────────
    _all_updates: dict[str, str] = {}

    for _section_name, _keys in SECTIONS.items():
        _icon = _section_icons.get(_section_name, "⚙️")
        with st.expander(f"{_icon}  {_section_name}", expanded=True):
            for _key in _keys:
                _meta = SETTINGS_SCHEMA[_key]
                # Current value: prefer what's in os.environ (already loaded), else file
                _current = os.environ.get(_key, _env_file_vals.get(_key, _meta.get("default", "")))

                _label    = _meta["label"]
                _help     = _meta.get("help", "")
                _type     = _meta.get("type", "text")

                if _type == "password":
                    # Show a masked input; placeholder shows whether a value exists
                    _placeholder = "●●●●●●●● (set)" if _current else "Not set"
                    _col1, _col2 = st.columns([4, 1])
                    with _col1:
                        _show_key = f"_show_{_key}"
                        _show = st.session_state.get(_show_key, False)
                        if _show:
                            _val = st.text_input(
                                _label, value=_current, key=f"set_{_key}",
                                help=_help, placeholder=_placeholder,
                            )
                        else:
                            _val = st.text_input(
                                _label, value=_current, key=f"set_{_key}",
                                type="password", help=_help, placeholder=_placeholder,
                            )
                    with _col2:
                        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                        _toggle_label = "Hide" if _show else "Show"
                        if st.button(_toggle_label, key=f"toggle_{_key}", use_container_width=True):
                            st.session_state[_show_key] = not _show
                            st.rerun()
                    _all_updates[_key] = _val

                elif _type == "select":
                    _opts = _meta.get("options", [])
                    _idx  = _opts.index(_current) if _current in _opts else 0
                    _val = st.selectbox(_label, options=_opts, index=_idx, help=_help, key=f"set_{_key}")
                    _all_updates[_key] = _val

                else:  # text
                    _val = st.text_input(_label, value=_current, help=_help, key=f"set_{_key}")
                    _all_updates[_key] = _val

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    _save_col, _test_col = st.columns([3, 1])
    with _save_col:
        if st.button("💾  Save Settings", use_container_width=True, type="primary"):
            # Only write non-empty updates (don't erase keys the user left blank accidentally)
            _to_save = {k: v for k, v in _all_updates.items() if v is not None}
            _save_settings(_to_save)
            st.success("✅ Settings saved! Changes are live now.")
            st.rerun()
    with _test_col:
        if st.button("🔄  Test Keys", use_container_width=True,
                     help="Quick connectivity check for OpenAI, Anthropic, and YouTube API"):
            _results = []
            # OpenAI
            _oai = os.environ.get("OPENAI_API_KEY", "")
            if _oai:
                try:
                    import openai as _oai_mod
                    _oai_mod.api_key = _oai
                    _oai_mod.models.list()
                    _results.append("✅ OpenAI — connected")
                except Exception as _e:
                    _results.append(f"❌ OpenAI — {_e}")
            else:
                _results.append("⚠️ OpenAI — key not set")

            # YouTube
            _yt = os.environ.get("YOUTUBE_API_KEY", "")
            if _yt:
                try:
                    import requests as _rq
                    _r = _rq.get(
                        "https://www.googleapis.com/youtube/v3/channels",
                        params={"part": "id", "id": "UC_x5XG1OV2P6uZZ5FSM9Ttw", "key": _yt},
                        timeout=8,
                    )
                    if _r.ok:
                        _results.append("✅ YouTube API — connected")
                    else:
                        _results.append(f"❌ YouTube API — {_r.status_code} {_r.json().get('error',{}).get('message','')}")
                except Exception as _e:
                    _results.append(f"❌ YouTube API — {_e}")
            else:
                _results.append("⚠️ YouTube API — key not set")

            # Ollama
            _ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            try:
                import requests as _rq
                if _rq.get(f"{_ollama_url}/api/tags", timeout=2).ok:
                    _results.append(f"✅ Ollama — reachable at {_ollama_url}")
                else:
                    _results.append(f"❌ Ollama — not reachable at {_ollama_url}")
            except Exception:
                _results.append(f"⚠️ Ollama — not running at {_ollama_url}")

            for _msg in _results:
                if _msg.startswith("✅"):
                    st.success(_msg)
                elif _msg.startswith("❌"):
                    st.error(_msg)
                else:
                    st.warning(_msg)
