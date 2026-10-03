from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Shared configuration for the lab: repo paths, compact-memory limits, and model providers."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _normalize_provider(value: str | None) -> str:
    if value is None:
        return "openai"

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "openai": "openai",
        "custom": "custom",
        "custom_openai": "custom",
        "openai_compatible": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google_generative_ai": "gemini",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "open_router": "openrouter",
        "model_provider": "openai",
    }
    return aliases.get(normalized, normalized)


def _default_model_for_provider(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o-mini",
        "custom": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-5-haiku-latest",
        "ollama": "llama3.1",
        "openrouter": "openai/gpt-4o-mini",
    }
    return defaults.get(provider, "gpt-4o-mini")


def _api_key_for_provider(provider: str) -> str | None:
    provider = _normalize_provider(provider)
    lookup = {
        "openai": ("OPENAI_API_KEY",),
        "custom": ("CUSTOM_API_KEY", "OPENAI_API_KEY"),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "ollama": ("OLLAMA_API_KEY",),
        "openrouter": ("OPENROUTER_API_KEY",),
    }
    return _first_env(*lookup.get(provider, ()))


def _base_url_for_provider(provider: str) -> str | None:
    provider = _normalize_provider(provider)
    lookup = {
        "openai": ("OPENAI_BASE_URL",),
        "custom": ("CUSTOM_BASE_URL", "OPENAI_BASE_URL"),
        "gemini": ("GEMINI_BASE_URL", "GOOGLE_BASE_URL"),
        "anthropic": ("ANTHROPIC_BASE_URL",),
        "ollama": ("OLLAMA_BASE_URL",),
        "openrouter": ("OPENROUTER_BASE_URL",),
    }
    return _first_env(*lookup.get(provider, ()))


def _build_provider_config(
    provider_name: str,
    model_name: str | None,
    *,
    default_temperature: float,
) -> ProviderConfig:
    provider = _normalize_provider(provider_name)
    return ProviderConfig(
        provider=provider,
        model_name=model_name or _default_model_for_provider(provider),
        temperature=default_temperature,
        api_key=_api_key_for_provider(provider),
        base_url=_base_url_for_provider(provider),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load project settings and provider configuration from environment variables."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    if load_dotenv is not None:
        load_dotenv(root / ".env")

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold_tokens = int(_first_env("COMPACT_THRESHOLD_TOKENS") or 2000)
    compact_keep_messages = int(_first_env("COMPACT_KEEP_MESSAGES") or 12)

    primary_provider = _normalize_provider(
        _first_env("LLM_PROVIDER", "MODEL_PROVIDER") or "openai"
    )
    primary_model = _first_env("LLM_MODEL", "MODEL_NAME") or _default_model_for_provider(primary_provider)
    judge_provider = _normalize_provider(
        _first_env("JUDGE_PROVIDER", "JUDGE_LLM_PROVIDER") or primary_provider
    )
    judge_model = _first_env("JUDGE_MODEL", "JUDGE_MODEL_NAME") or primary_model

    model_cfg = _build_provider_config(
        primary_provider,
        primary_model,
        default_temperature=float(_first_env("LLM_TEMPERATURE") or 0.1),
    )
    if primary_provider == "custom":
        model_cfg.base_url = _first_env("CUSTOM_BASE_URL", "OPENAI_BASE_URL") or model_cfg.base_url
        model_cfg.api_key = _first_env("CUSTOM_API_KEY", "OPENAI_API_KEY") or model_cfg.api_key
    elif primary_provider == "ollama":
        model_cfg.base_url = _first_env("OLLAMA_BASE_URL") or model_cfg.base_url or "http://localhost:11434"
    elif primary_provider == "openai":
        model_cfg.api_key = _first_env("OPENAI_API_KEY") or model_cfg.api_key

    judge_cfg = _build_provider_config(
        judge_provider,
        judge_model,
        default_temperature=float(_first_env("JUDGE_TEMPERATURE") or 0.0),
    )
    if judge_provider == "custom":
        judge_cfg.base_url = _first_env("CUSTOM_BASE_URL", "OPENAI_BASE_URL") or judge_cfg.base_url
        judge_cfg.api_key = _first_env("CUSTOM_API_KEY", "OPENAI_API_KEY") or judge_cfg.api_key
    elif judge_provider == "ollama":
        judge_cfg.base_url = _first_env("OLLAMA_BASE_URL") or judge_cfg.base_url or "http://localhost:11434"
    elif judge_provider == "openai":
        judge_cfg.api_key = _first_env("OPENAI_API_KEY") or judge_cfg.api_key

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model_cfg,
        judge_model=judge_cfg,
    )
