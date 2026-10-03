"""
Shared LLM factory.
Primary: local Ollama (free, used for precompute search planning).
All other calls use the configured provider (OpenAI by default).

Model names and provider are read from env vars (set via the Settings screen):
  LLM_FAST_MODEL    — fast/cheap model  (default: gpt-4o-mini)
  LLM_QUALITY_MODEL — quality model     (default: gpt-4o)
  LLM_PROVIDER      — "openai" or "anthropic" (default: openai)
  OLLAMA_MODEL      — local Ollama model (default: qwen3.5:9b)
  OLLAMA_BASE_URL   — Ollama server URL  (default: http://localhost:11434)
"""
import os

try:
    from langchain_ollama import ChatOllama as _ChatOllama
    _OLLAMA_PKG = True
except ImportError:
    _ChatOllama = None  # type: ignore
    _OLLAMA_PKG = False


def _fast_model() -> str:
    return os.environ.get("LLM_FAST_MODEL", "gpt-4o-mini")


def _quality_model() -> str:
    return os.environ.get("LLM_QUALITY_MODEL", "gpt-4o")


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "openai").lower()


def _ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _ollama_reachable() -> bool:
    try:
        import requests
        return requests.get(f"{_ollama_base_url()}/api/tags", timeout=2).ok
    except Exception:
        return False


def _make_llm(model: str, temperature: float, timeout: int = 90):
    """Instantiate the right LLM class based on LLM_PROVIDER env var."""
    provider = _provider()
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature, timeout=timeout)
        except ImportError:
            print("[LLM] langchain_anthropic not installed — falling back to OpenAI")

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=temperature, timeout=timeout)


def get_ollama_llm(temperature: float = 0.3, think: bool = False):
    """
    Return a ChatOllama instance for local inference (precompute search planning).
    Falls back to the configured fast model if the local Ollama server is not reachable.
    """
    if _OLLAMA_PKG and _ollama_reachable():
        kwargs: dict = {
            "model":           _ollama_model(),
            "temperature":     temperature,
            "base_url":        _ollama_base_url(),
            "think":           think,
            "request_timeout": 120,
        }
        return _ChatOllama(**kwargs)

    print(f"[Ollama] Server not reachable — falling back to {_fast_model()}")
    return _make_llm(_fast_model(), temperature)


def get_haiku_llm(temperature: float = 0.3):
    """Fast, cheap model for tool calls, intent detection, JSON generation, simple tasks."""
    return _make_llm(_fast_model(), temperature)


def get_sonnet_llm(temperature: float = 0.5):
    """High-quality model for script generation, complex reasoning, long-form output."""
    return _make_llm(_quality_model(), temperature)
