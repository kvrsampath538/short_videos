"""
Application settings manager.
Reads from and writes to the .env file in the project root.
All agents and LLM clients read their configuration through this module
(or via os.environ which this module keeps in sync).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_ROOT    = Path(__file__).parent
_ENV_FILE = _ROOT / ".env"

# ── Schema ─────────────────────────────────────────────────────────────────────
# Each entry: key → {label, section, type, default, help, secret}
SETTINGS_SCHEMA: dict[str, dict] = {
    # ── API Keys ──────────────────────────────────────────────────────────────
    "OPENAI_API_KEY": {
        "label":   "OpenAI API Key",
        "section": "API Keys",
        "type":    "password",
        "default": "",
        "help":    "Used by the research, idea-generation, and analysis agents. Get yours at platform.openai.com",
        "secret":  True,
    },
    "ANTHROPIC_API_KEY": {
        "label":   "Anthropic API Key",
        "section": "API Keys",
        "type":    "password",
        "default": "",
        "help":    "Used if you switch any agent to a Claude model. Get yours at console.anthropic.com",
        "secret":  True,
    },
    "YOUTUBE_API_KEY": {
        "label":   "YouTube Data API Key",
        "section": "API Keys",
        "type":    "password",
        "default": "",
        "help":    "Required for 'My Channel Analysis'. Get yours at console.cloud.google.com (YouTube Data API v3)",
        "secret":  True,
    },
    "TAVILY_API_KEY": {
        "label":   "Tavily Search API Key",
        "section": "API Keys",
        "type":    "password",
        "default": "",
        "help":    "Optional alternative search backend. Get yours at app.tavily.com",
        "secret":  True,
    },
    "GOOGLE_AI_STUDIO_API_KEY": {
        "label":   "Google AI Studio API Key",
        "section": "API Keys",
        "type":    "password",
        "default": "",
        "help":    "Optional — for Google Gemini models. Get yours at aistudio.google.com",
        "secret":  True,
    },

    # ── LLM Models ────────────────────────────────────────────────────────────
    "LLM_FAST_MODEL": {
        "label":   "Fast / Cheap Model",
        "section": "LLM Models",
        "type":    "text",
        "default": "gpt-4o-mini",
        "help":    "Used for intent detection, JSON generation, image prompts, chat routing, audience analysis, channel performance. Examples: gpt-4o-mini, gpt-3.5-turbo, claude-haiku-4-5-20251001",
        "secret":  False,
    },
    "LLM_QUALITY_MODEL": {
        "label":   "Quality / Reasoning Model",
        "section": "LLM Models",
        "type":    "text",
        "default": "gpt-4o",
        "help":    "Used for idea synthesis, research, script generation, channel analysis. Examples: gpt-4o, gpt-4-turbo, claude-sonnet-5",
        "secret":  False,
    },
    "LLM_PROVIDER": {
        "label":   "LLM Provider",
        "section": "LLM Models",
        "type":    "select",
        "options": ["openai", "anthropic"],
        "default": "openai",
        "help":    "Which provider to use for LLM_FAST_MODEL and LLM_QUALITY_MODEL. Ollama always uses the local server regardless of this setting.",
        "secret":  False,
    },

    # ── Ollama (local LLM) ────────────────────────────────────────────────────
    "OLLAMA_MODEL": {
        "label":   "Ollama Model",
        "section": "Local LLM (Ollama)",
        "type":    "text",
        "default": "qwen3.5:9b",
        "help":    "Local model used for precompute search planning. Must be pulled with `ollama pull <model>`. Examples: qwen3.5:9b, llama3.2, mistral",
        "secret":  False,
    },
    "OLLAMA_BASE_URL": {
        "label":   "Ollama Base URL",
        "section": "Local LLM (Ollama)",
        "type":    "text",
        "default": "http://localhost:11434",
        "help":    "URL where the Ollama server is running. Default: http://localhost:11434",
        "secret":  False,
    },

    # ── Search Integrations ───────────────────────────────────────────────────
    "OPENSERP_BASE_URL": {
        "label":   "OpenSERP Base URL",
        "section": "Integrations",
        "type":    "text",
        "default": "http://localhost:7070",
        "help":    "URL for the OpenSERP Docker search service used by research agents. Default: http://localhost:7070",
        "secret":  False,
    },

    # ── Channel ───────────────────────────────────────────────────────────────
    "CHANNEL_URL": {
        "label":   "YouTube Channel URL",
        "section": "Channel",
        "type":    "text",
        "default": "https://www.youtube.com/@Worldaffairs0138",
        "help":    "Your YouTube channel URL. Used by channel analysis and performance features.",
        "secret":  False,
    },
    "CHANNEL_LANGUAGE": {
        "label":   "Channel Language",
        "section": "Channel",
        "type":    "text",
        "default": "Telugu",
        "help":    "Primary language of the channel. Used in prompts to tailor content and research to the right audience.",
        "secret":  False,
    },
}

# Group schema by section (preserving insertion order)
def _sections() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, meta in SETTINGS_SCHEMA.items():
        s = meta["section"]
        out.setdefault(s, []).append(key)
    return out

SECTIONS = _sections()


# ── Read / Write .env ─────────────────────────────────────────────────────────

def _parse_env(path: Path) -> dict[str, str]:
    """Parse .env file into a dict (preserving all lines including comments)."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _write_env(path: Path, values: dict[str, str]) -> None:
    """Write key=value pairs to .env, preserving comment lines and unknown keys."""
    existing_lines: list[str] = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()

    # Track which keys already have a line in the file
    handled: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in values:
                new_lines.append(f"{k}={values[k]}")
                handled.add(k)
            else:
                new_lines.append(line)  # keep existing unknown key unchanged
        else:
            new_lines.append(line)  # keep comments / blank lines

    # Append any new keys not already in the file
    for k, v in values.items():
        if k not in handled:
            new_lines.append(f"{k}={v}")

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ── Public API ────────────────────────────────────────────────────────────────

def load_all() -> dict[str, str]:
    """Return all env vars from the .env file (raw strings)."""
    return _parse_env(_ENV_FILE)


def get_setting(key: str, default: str | None = None) -> str:
    """
    Get a setting value.  Priority: os.environ > .env file > schema default > default arg.
    """
    # os.environ is already populated from .env at app startup via _load_env()
    env_val = os.environ.get(key)
    if env_val is not None:
        return env_val
    # Fall back to schema default, then the caller's default
    schema_default = SETTINGS_SCHEMA.get(key, {}).get("default", "")
    return schema_default if schema_default else (default or "")


def save_settings(updates: dict[str, str]) -> None:
    """
    Persist updated settings to .env and apply them to os.environ immediately.
    Empty values are kept (they clear the key in the file but don't unset env).
    """
    current = _parse_env(_ENV_FILE)
    current.update(updates)
    _write_env(_ENV_FILE, current)

    # Apply to running process so agents pick up changes without a restart
    for k, v in updates.items():
        if v:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]


def get_channel_url() -> str:
    return get_setting("CHANNEL_URL", "https://www.youtube.com/@Worldaffairs0138")


def get_channel_handle() -> str:
    """Return just the @Handle portion from CHANNEL_URL."""
    url = get_channel_url()
    if "/@" in url:
        return "@" + url.split("/@", 1)[1].rstrip("/")
    return url


def get_channel_language() -> str:
    return get_setting("CHANNEL_LANGUAGE", "Telugu")
