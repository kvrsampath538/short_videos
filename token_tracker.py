"""
Live token usage tracker for ChatOpenAI, ChatAnthropic and ChatOllama calls.

Usage:
    import token_tracker
    token_tracker.patch_openai()     # call once at app startup
    token_tracker.patch_ollama()     # call once at app startup

    # In Streamlit, set a placeholder to get live updates:
    ph = st.empty()
    token_tracker.set_placeholder(ph)

    token_tracker.reset()            # clear before a new operation
"""

import threading
from datetime import datetime

_lock        = threading.Lock()
_calls: list[dict] = []
_placeholder = None   # Streamlit st.empty() reference


# ── Patch helpers ─────────────────────────────────────────────────────────────

def patch_openai() -> None:
    """Monkey-patch ChatOpenAI.invoke once so every call is automatically tracked."""
    from langchain_openai import ChatOpenAI
    if getattr(ChatOpenAI, "_token_tracked", False):
        return

    _orig = ChatOpenAI.invoke

    def _wrap(self, input, config=None, **kwargs):
        response = _orig(self, input, config, **kwargs)
        try:
            # LangChain standard interface
            usage = getattr(response, "usage_metadata", None) or {}
            in_t  = usage.get("input_tokens", 0)
            out_t = usage.get("output_tokens", 0)
            # Fallback to response_metadata["token_usage"]
            if not in_t and not out_t:
                meta  = getattr(response, "response_metadata", {}) or {}
                tu    = meta.get("token_usage", {})
                in_t  = tu.get("prompt_tokens", 0)
                out_t = tu.get("completion_tokens", 0)
            _record(
                source        = "openai",
                model         = getattr(self, "model_name", "gpt-4o"),
                input_tokens  = in_t,
                output_tokens = out_t,
            )
        except Exception:
            pass
        return response

    ChatOpenAI.invoke         = _wrap
    ChatOpenAI._token_tracked = True


def patch_anthropic() -> None:
    """Monkey-patch ChatAnthropic.invoke (kept for backward compat — no longer primary)."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return
    if getattr(ChatAnthropic, "_token_tracked", False):
        return

    _orig = ChatAnthropic.invoke

    def _wrap(self, input, config=None, **kwargs):
        response = _orig(self, input, config, **kwargs)
        try:
            meta  = getattr(response, "response_metadata", {}) or {}
            usage = meta.get("usage", {})
            _record(
                source        = "openai",   # display under same bucket
                model         = getattr(self, "model", "claude"),
                input_tokens  = usage.get("input_tokens",  0),
                output_tokens = usage.get("output_tokens", 0),
            )
        except Exception:
            pass
        return response

    ChatAnthropic.invoke         = _wrap
    ChatAnthropic._token_tracked = True


def patch_ollama() -> None:
    """Monkey-patch ChatOllama.invoke once so every call is automatically tracked."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        return  # package not installed — skip silently

    if getattr(ChatOllama, "_token_tracked", False):
        return

    _orig = ChatOllama.invoke

    def _wrap(self, input, config=None, **kwargs):
        response = _orig(self, input, config, **kwargs)
        try:
            meta = getattr(response, "response_metadata", {}) or {}
            _record(
                source        = "ollama",
                model         = getattr(self, "model", "ollama"),
                input_tokens  = meta.get("prompt_eval_count", 0),
                output_tokens = meta.get("eval_count",        0),
            )
        except Exception:
            pass
        return response

    ChatOllama.invoke         = _wrap
    ChatOllama._token_tracked = True


# ── Internal helpers ──────────────────────────────────────────────────────────

def _record(source: str, model: str, input_tokens: int, output_tokens: int) -> None:
    with _lock:
        _calls.append({
            "n":      len(_calls) + 1,
            "source": source,
            "model":  model,
            "in":     input_tokens,
            "out":    output_tokens,
            "time":   datetime.now().strftime("%H:%M:%S"),
        })
    _refresh()


def _refresh() -> None:
    """Re-render the Streamlit placeholder with the current call log."""
    if _placeholder is None:
        return
    try:
        with _lock:
            calls = list(_calls)

        if not calls:
            _placeholder.empty()
            return

        oai = [c for c in calls if c["source"] == "openai"]
        olm = [c for c in calls if c["source"] == "ollama"]

        def _bar(label: str, icon: str, color: str, subset: list) -> str:
            if not subset:
                return ""
            ti = sum(c["in"]  for c in subset)
            to = sum(c["out"] for c in subset)
            return (
                f"<span style='margin-right:18px'>"
                f"{icon} <b style='color:{color}'>{label}</b> &nbsp;"
                f"⬆ <b>{ti:,}</b> in &nbsp;+&nbsp; ⬇ <b>{to:,}</b> out "
                f"= <b>{ti+to:,}</b> &nbsp;"
                f"<span style='color:#888;font-size:0.9em'>({len(subset)} call{'s' if len(subset)!=1 else ''})</span>"
                f"</span>"
            )

        oai_bar = _bar("OpenAI", "🤖", "#60A5FA", oai)
        olm_bar = _bar("Ollama", "🖥️", "#34D399", olm)

        recent      = calls[-3:]
        recent_note = (
            f" &nbsp;·&nbsp; <span style='color:#888;font-size:0.9em'>"
            f"showing last {len(recent)} of {len(calls)}</span>"
            if len(calls) > len(recent) else ""
        )

        rows = "\n".join(
            f"| {c['n']} | {'🤖' if c['source']=='openai' else '🖥️'} `{c['model']}` "
            f"| {c['in']:,} | {c['out']:,} | **{c['in']+c['out']:,}** "
            f"| {c['time']} |"
            for c in recent
        )

        _placeholder.markdown(
            f"<div style='"
            f"background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.10);"
            f"border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:0.82em'>"
            f"🔢 <b>Token Usage</b>{recent_note}<br>"
            f"<div style='margin-top:6px'>{oai_bar}{olm_bar}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        _placeholder.markdown(
            "| # | Model | Input | Output | Total | Time |\n"
            "|--:|-------|------:|------:|------:|------|\n"
            + rows,
        )
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def reset() -> None:
    """Clear the call log and blank the placeholder (call before each new operation)."""
    with _lock:
        _calls.clear()
    if _placeholder is not None:
        try:
            _placeholder.empty()
        except Exception:
            pass


def set_placeholder(ph) -> None:
    """Register a Streamlit st.empty() placeholder for live display."""
    global _placeholder
    _placeholder = ph
    _refresh()


def get_totals() -> dict:
    with _lock:
        calls = list(_calls)
    oai = [c for c in calls if c["source"] == "openai"]
    olm = [c for c in calls if c["source"] == "ollama"]
    return {
        "calls":              len(calls),
        "openai_in":          sum(c["in"]  for c in oai),
        "openai_out":         sum(c["out"] for c in oai),
        "ollama_in":          sum(c["in"]  for c in olm),
        "ollama_out":         sum(c["out"] for c in olm),
        "total_input_tokens": sum(c["in"]  for c in calls),
        "total_output_tokens":sum(c["out"] for c in calls),
        "total_tokens":       sum(c["in"] + c["out"] for c in calls),
    }
