import streamlit as st
import sys
import os
import json
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

# Import agent functions directly — no LangGraph stream/interrupt needed in UI
from agents.idea_generator_agent import (
    idea_generator_agent_node, save_to_blacklist, save_to_favorites,
    _load_history, _compact_history,
    save_to_bookmarks, load_bookmarks, remove_from_bookmarks, update_history_saturation,
    load_audience_insights,
)
from agents.research_agent import research_agent_node
from agents.freshness_agent import freshness_check_agent_node, bulk_saturation_filter
from agents.image_prompt_agent import image_prompt_agent_node
from agents.script_agent import script_agent_node
from agents.audience_analysis_agent import analyze_audience_interest_node
from agents.channel_analysis_agent import analyze_channel_node, load_channel_analysis as load_ca
from agents.chat_orchestrator_agent import detect_intent, chat_response

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
                  f"Scene {s.get('scene_number'):02d}  [{s.get('duration_seconds')}s]",
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
}


def _inject_phase_theme() -> None:
    t = _PHASE_THEMES.get(st.session_state.get("phase", "input"), _PHASE_THEMES["input"])
    a, bg, img = t["a"], t["bg"], t["img"]
    st.markdown(f"""<style>
    .stApp{{
        background-image:{bg},url('{img}') !important;
        background-size:auto auto,cover !important;
        background-position:0 0,center center !important;
        background-attachment:fixed,fixed !important;
        background-repeat:no-repeat,no-repeat !important;
    }}
    .main .block-container{{background:transparent !important;}}
    header[data-testid="stHeader"]{{background:rgba(8,8,8,0.9) !important;backdrop-filter:blur(12px) !important;border-bottom:1px solid {a}18 !important;}}
    div[data-testid="stButton"]>button{{border-radius:10px !important;font-weight:600 !important;letter-spacing:0.2px !important;transition:all 0.18s ease !important;}}
    div[data-testid="stButton"]>button[kind="primary"]{{background:linear-gradient(135deg,{a},{a}cc) !important;color:white !important;border:none !important;box-shadow:0 4px 14px {a}38 !important;}}
    div[data-testid="stButton"]>button[kind="primary"]:hover{{box-shadow:0 6px 20px {a}55 !important;transform:translateY(-1px) !important;filter:brightness(1.08) !important;}}
    div[data-testid="stButton"]>button[kind="secondary"]{{background:rgba(255,255,255,0.06) !important;color:#e0e0e0 !important;border:1px solid rgba(255,255,255,0.12) !important;}}
    div[data-testid="stButton"]>button[kind="secondary"]:hover{{background:rgba(255,255,255,0.11) !important;border-color:{a}55 !important;color:white !important;}}
    div[data-testid="stVerticalBlockBorderWrapper"]{{background:rgba(255,255,255,0.04) !important;border:1px solid {a}22 !important;border-radius:14px !important;transition:all 0.2s ease;}}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover{{border-color:{a}45 !important;background:rgba(255,255,255,0.06) !important;}}
    div[data-testid="stTextInput"]>div>div>input{{background:rgba(255,255,255,0.08) !important;border:1px solid {a}35 !important;color:white !important;border-radius:10px !important;}}
    div[data-testid="stTextInput"]>div>div>input:focus{{border-color:{a} !important;box-shadow:0 0 0 2px {a}22 !important;}}
    div[data-testid="stTextInput"]>div>div>input::placeholder{{color:rgba(255,255,255,0.35) !important;}}
    div[data-testid="stSelectbox"]>div>div{{background:rgba(255,255,255,0.08) !important;border:1px solid {a}35 !important;color:white !important;border-radius:10px !important;}}
    div[data-testid="stStatusContainer"]{{background:rgba(255,255,255,0.05) !important;border:1px solid {a}28 !important;border-radius:12px !important;}}
    div[data-testid="stExpander"]>div:first-child{{background:rgba(255,255,255,0.05) !important;border:1px solid {a}25 !important;border-radius:10px !important;}}
    div[data-testid="stChatMessage"]{{background:rgba(255,255,255,0.04) !important;border:1px solid rgba(255,255,255,0.07) !important;border-radius:14px !important;}}
    div[data-testid="stChatInput"] textarea{{background:rgba(255,255,255,0.08) !important;border-color:{a}35 !important;color:white !important;border-radius:12px !important;}}
    hr{{border-color:{a}14 !important;}}
    ::-webkit-scrollbar{{width:5px;height:5px;}}
    ::-webkit-scrollbar-track{{background:transparent;}}
    ::-webkit-scrollbar-thumb{{background:{a}50;border-radius:3px;}}
    ::-webkit-scrollbar-thumb:hover{{background:{a};}}
    .idea-card{{background:rgba(255,255,255,0.04);border-radius:12px;padding:18px 20px;margin-bottom:10px;border-left:4px solid {a};}}
    .idea-area-tag{{display:inline-block;background:{a}20;color:{a};border:1px solid {a}35;border-radius:20px;padding:2px 10px;font-size:0.75em;margin-bottom:8px;font-weight:600;}}
    .hook-text{{color:{a};font-style:italic;font-size:0.92em;}}
    .telugu-text{{font-size:1.1em;line-height:1.8;}}
    .yt-banner{{background:linear-gradient(90deg,{a}1f 0%,{a}0a 65%,transparent 100%);border-left:4px solid {a};border-radius:0 14px 14px 0;padding:14px 22px;margin-bottom:20px;}}
    .yt-banner-title{{font-size:1.5em;font-weight:800;color:white;line-height:1.3;margin:0;}}
    .yt-banner-sub{{color:{a};font-size:0.83em;margin-top:3px;}}
    </style>""", unsafe_allow_html=True)


def _screen_header(icon: str, title: str, subtitle: str = "") -> None:
    sub = f'<div class="yt-banner-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="yt-banner"><div class="yt-banner-title">{icon}&nbsp; {title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


_inject_phase_theme()


# ── Header ────────────────────────────────────────────────────────────────────

_pa = _PHASE_THEMES.get(st.session_state.get("phase", "input"), _PHASE_THEMES["input"])["a"]
st.markdown(
    f"<h1 style='text-align:center;margin-bottom:0;color:white'>"
    f"<span style='color:{_pa}'>🎬</span> YouTube Shorts AI Generator</h1>",
    unsafe_allow_html=True,
)
if st.session_state.get("phase", "input") == "input":
    st.markdown(
        "<p style='text-align:center;color:#888;margin-top:4px;font-size:0.82em'>"
        "Idea Generator → Research → Freshness Check → Your Approval → Images → Telugu Script"
        "</p>",
        unsafe_allow_html=True,
    )

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
            "<div style='text-align:center;margin-bottom:14px'>"
            "<div style='font-size:1.1em;font-weight:700;color:white'>💡 What's your YouTube Short about?</div>"
            "<div style='font-size:0.78em;color:#888;margin-top:4px'>"
            "Type your idea below · or let AI discover something viral for you</div></div>",
            unsafe_allow_html=True,
        )
        topic = st.text_input(
            "Your own idea",
            placeholder="e.g. ultraedge technology in cricket",
            label_visibility="collapsed",
        )
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("🚀 Use My Idea", type="primary", use_container_width=True):
                if topic.strip():
                    st.session_state.topic              = topic.strip()
                    st.session_state.use_idea_generator = False
                    st.session_state.phase              = "research"
                    st.rerun()
                else:
                    st.error("Please type your idea first.")
        with b2:
            if st.button("✨ Generate Ideas for Me", use_container_width=True):
                st.session_state.use_idea_generator  = True
                st.session_state.generated_ideas     = []
                st.session_state.all_displayed_ideas = []
                st.session_state.idea_gen_iteration  = 0
                st.session_state._ideas_ready        = False
                st.session_state.phase               = "idea_selection"
                st.rerun()
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

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
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
            "Analyse all Shorts from @Worldaffairs0138 — views, likes & content breakdown by area"
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
                f"@Worldaffairs0138 · {_ca_total} Shorts analysed"
                + (f" · {_ca_subs:,} subscribers" if _ca_subs else "")
            )

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
                    tag_line = (
                        f"<span class='idea-area-tag'>{icon} {area.upper()}</span>"
                        + (f"&nbsp;&nbsp;<span style='color:#6b7280;font-size:0.75em'>{date}</span>" if date else "")
                    )
                    if sat_badge:
                        tag_line += f"&nbsp;&nbsp;<span style='font-size:0.8em'>{sat_badge}</span>"
                    st.markdown(tag_line, unsafe_allow_html=True)
                    st.markdown(f"**{title}**")
                    st.markdown(idea.get("concept", ""))
                    if idea.get("viral_hook"):
                        st.markdown(
                            f"<div class='hook-text'>🎣 {idea['viral_hook']}</div>",
                            unsafe_allow_html=True,
                        )
                    if idea.get("saturation_reason"):
                        st.caption(f"📊 {idea['saturation_reason']}")

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
                    tag_line = ""
                    if area:
                        tag_line += f"<span class='idea-area-tag'>{icon} {area.upper()}</span>"
                    if bm_at:
                        tag_line += f"&nbsp;&nbsp;<span style='color:#6b7280;font-size:0.75em'>{bm_at}</span>"
                    if source_label:
                        tag_line += f"&nbsp;&nbsp;<span style='color:#14B8A6;font-size:0.75em'>from {source_label}</span>"
                    if tag_line:
                        st.markdown(tag_line, unsafe_allow_html=True)
                    st.markdown(f"**{title}**")
                    st.markdown(bm.get("concept", ""))
                    if bm.get("viral_hook"):
                        st.markdown(
                            f"<div class='hook-text'>🎣 {bm['viral_hook']}</div>",
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
            "✅ Saved to `audience_insights.json` — idea generator and research agent will now "
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

        gen_state = {
            "idea_gen_iteration": st.session_state.get("idea_gen_iteration", 0),
            "generated_ideas":    st.session_state.get("all_displayed_ideas", []),
            "messages":           [],
        }

        with st.status("✨ Idea Generator searching the web...", expanded=True) as status:
            st.write("Scanning history, science, psychology, technology, inspiring stories & more...")

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

                    st.write(f"✓ {len(new_ideas)} candidates — checking YouTube saturation...")
                    filtered, all_scored = bulk_saturation_filter(new_ideas)
                    update_history_saturation(all_scored)

                    added   = 0
                    removed = len(new_ideas) - len(filtered)
                    for idea in filtered:
                        t = idea.get("title", "")
                        if t not in existing_titles and len(fresh_ideas) < TARGET_DISPLAY:
                            fresh_ideas.append(idea)
                            existing_titles.add(t)
                            added += 1

                    if removed > 0:
                        st.write(
                            f"🚫 {removed} removed (saturated on YouTube) · "
                            f"✅ +{added} fresh · {len(fresh_ideas)}/{TARGET_DISPLAY} total"
                        )
                    else:
                        st.write(f"✅ +{added} fresh ideas ({len(fresh_ideas)}/{TARGET_DISPLAY})")

                    # Update gen_state so the next batch avoids everything generated so far
                    gen_state = {
                        "idea_gen_iteration": result.get("idea_gen_iteration", batch + 1),
                        "generated_ideas":    st.session_state.all_displayed_ideas + all_generated,
                        "messages":           [],
                    }

                # Fallback: if every batch was fully saturated, show best available
                if not fresh_ideas and all_generated:
                    st.warning("Could not find fresh ideas after multiple attempts — showing best available.")
                    for idea in all_generated:
                        t = idea.get("title", "")
                        if t not in existing_titles and len(fresh_ideas) < TARGET_DISPLAY:
                            fresh_ideas.append(idea)
                            existing_titles.add(t)

                for idea in fresh_ideas:
                    st.session_state.all_displayed_ideas.append(idea)

                n = len(fresh_ideas)
                status.update(
                    label=f"✅ {n} fresh idea{'s' if n != 1 else ''} ready!" if fresh_ideas else "⚠️ Failed",
                    state="complete",
                )

            except Exception as e:
                st.error(f"Error generating ideas: {e}")
                st.code(traceback.format_exc(), language="python")
                status.update(label="⚠️ Failed", state="error")

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
                    sat_badge  = {1: "🟢 Low saturation", 2: "🟡 Medium saturation"}.get(sat_score, "")
                    sat_reason = idea.get("saturation_reason", "")
                    tag_line   = f"<span class='idea-area-tag'>{icon} {idea.get('area','').upper()}</span>"
                    if sat_badge:
                        tag_line += f"&nbsp;&nbsp;<span style='font-size:0.8em;color:#6ee7b7'>{sat_badge}</span>"
                    st.markdown(tag_line, unsafe_allow_html=True)
                    st.markdown(f"**{title}**")
                    st.markdown(idea.get("concept", ""))
                    st.markdown(
                        f"<div class='hook-text'>🎣 {idea.get('viral_hook','')}</div>",
                        unsafe_allow_html=True,
                    )
                    if sat_reason:
                        st.caption(f"📊 {sat_reason}")
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
        agent_state = {
            "input_topic":    topic_display,
            "research_ideas": [{"title": topic_display, "concept": topic_display}],
            "messages":       [],
        }
        with st.status("🔍 Searching YouTube Shorts...", expanded=True) as status:
            st.write(f"Checking how saturated **{topic_display}** is on YouTube Shorts...")
            try:
                result   = freshness_check_agent_node(agent_state)
                st.session_state._sat_check_result = result
                st.session_state._sat_check_done   = True
                approved = result.get("freshness_approved", False)
                status.update(
                    label="✅ Check complete — low saturation!" if approved else
                          "⚠️ Check complete — heavily saturated",
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

    if approved is True:
        st.success(
            f"✅ **{topic_display}** has LOW saturation — not heavily covered on YouTube yet!\n\n"
            + (best.get("why_best", "") or "")
        )
    elif approved is False:
        st.error(
            f"⚠️ **{topic_display}** is already heavily saturated on YouTube Shorts.\n\n"
            + (reason or "")
        )

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
            agent_state = {
                "input_topic":    st.session_state.topic,
                "research_ideas": [selected_idea],
                "messages":       [],
            }
            with st.status("📊 Freshness Check Agent", expanded=True) as f_status:
                st.write(f"Searching YouTube for: **{topic_display}**")
                try:
                    f_result = freshness_check_agent_node(agent_state)
                    best     = f_result.get("best_idea")
                    approved = f_result.get("freshness_approved", False)
                    reason   = f_result.get("rejection_reason", "")
                    st.session_state.best_idea          = best
                    st.session_state.freshness_approved = approved
                    st.session_state.rejection_reason   = reason
                    if approved:
                        f_status.update(label="✅ Idea is fresh — not saturated!", state="complete")
                    else:
                        f_status.update(label="⚠️ Idea is saturated on YouTube", state="complete")
                except Exception as e:
                    st.error(f"Freshness check error: {e}")
                    f_status.update(label="❌ Error", state="error")

        # Show result and let user decide
        if st.session_state.get("freshness_approved"):
            best = st.session_state.best_idea or {}
            st.success(f"✅ **{topic_display}** is fresh — not heavily covered on YouTube!")
            if best.get("why_best"):
                st.caption(best["why_best"])
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
        for fname, label in [("telugu_script.txt", "⬇️  Telugu Script"), ("storyboard.txt", "⬇️  Storyboard")]:
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
                       "Performance breakdown of @Worldaffairs0138 Shorts by content area")
    with ca_back_col:
        st.markdown("<div style='padding-top:18px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True, key="ca_back"):
            st.session_state.phase = "input"
            st.rerun()

    if not _has_yt_key:
        st.error(
            "**YouTube API key not configured.**\n\n"
            "This feature requires a `YOUTUBE_API_KEY` in your `.env` file.\n\n"
            "**How to get one (free):**\n"
            "1. Go to Google Cloud Console (console.cloud.google.com)\n"
            "2. Create or select a project → Enable **YouTube Data API v3**\n"
            "3. Credentials → **Create credentials** → **API key**\n"
            "4. Add to `.env`: `YOUTUBE_API_KEY=AIza...`\n"
            "5. Restart the Streamlit app"
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
            with st.status("📡 Fetching Shorts from @Worldaffairs0138…", expanded=True) as ca_status:
                st.write("Connecting to YouTube Data API v3…")
                st.write("Fetching all uploads from your channel…")
                st.write("Filtering to Shorts (≤ 3 min or #shorts tag)…")
                st.write("Categorising Shorts by content area with AI…")
                try:
                    r = analyze_channel_node({})
                    err = r.get("error")
                    if err == "NO_API_KEY":
                        ca_status.update(label="❌ No YouTube API key", state="error")
                        st.error("YOUTUBE_API_KEY not set. Add it to `.env` and restart.")
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
        f"<a href='https://www.youtube.com/@Worldaffairs0138' target='_blank' "
        f"style='color:#FF0000'>@Worldaffairs0138</a></div>",
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
                    st.markdown(
                        f"<span style='color:#FF0000;font-weight:800;font-size:1.1em'>#{rank}</span>"
                        f"&nbsp;&nbsp;**{v.get('title','')}**",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<span class='idea-area-tag'>{icon} {area.upper()}</span>"
                        f"&nbsp;&nbsp;<span style='color:#9ca3af;font-size:0.8em'>"
                        f"{v.get('published_at','')[:10]}</span>"
                        f"&nbsp;&nbsp;<a href='https://www.youtube.com/shorts/{v.get('id','')}' "
                        f"target='_blank' style='color:#FF0000;font-size:0.8em'>▶ Watch</a>",
                        unsafe_allow_html=True,
                    )
                with tc2:
                    st.markdown(
                        f"<div style='text-align:right;padding-top:4px'>"
                        f"<div style='color:#3B82F6;font-weight:700'>"
                        f"{_ca_fmt(v.get('view_count',0))} 👁️</div>"
                        f"<div style='color:#F43F5E;font-size:0.85em'>"
                        f"{_ca_fmt(v.get('like_count',0))} ❤️</div>"
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

            _col_card, _col_view = st.columns([5, 1])

            with _col_card:
                with st.container(border=True):
                    # Tag row
                    _tag = ""
                    if _area:
                        _tag += f"<span class='idea-area-tag'>{_icon} {_area.upper()}</span>&nbsp;&nbsp;"
                    _tag += (
                        f"<span style='color:#9ca3af;font-size:0.78em'>{_b['date_str']}</span>"
                        f"&nbsp;&nbsp;<span style='color:#06B6D4;font-size:0.78em'>"
                        f"{_n_sc} scenes · {_dur}s"
                        + (f" · {_n_img} 🖼️" if _n_img else "")
                        + "</span>"
                    )
                    st.markdown(_tag, unsafe_allow_html=True)
                    st.markdown(f"**{_title}**")
                    if _concept:
                        st.caption(_concept[:140] + ("…" if len(_concept) > 140 else ""))
                    # First narration as hook preview
                    _first_narr = _scenes[0].get("narration", "") if _scenes else ""
                    if _first_narr:
                        st.markdown(
                            f"<div class='hook-text'>🎬 {_first_narr[:100]}"
                            + ("…" if len(_first_narr) > 100 else "")
                            + "</div>",
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
        _bv_dl_cols = st.columns(2)
        for _ci, (fname, label) in enumerate([
            ("telugu_script.txt", "⬇️ Telugu Script"),
            ("storyboard.txt",    "⬇️ Storyboard"),
        ]):
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
