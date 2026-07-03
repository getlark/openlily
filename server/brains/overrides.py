"""Optional per-brain overrides loaded from ``server/brains.yaml``.

A user can pick which brain runs (``default_brain``) and override each brain's
STT/TTS/LLM *model names* and the TTS *voice* without editing code -- copy
``brains.yaml.example`` to ``brains.yaml`` (git-ignored, like ``.env``) and edit.
The built-in defaults stay inline in each brain's ``build()``; this only layers
the file on top, so a missing file or an omitted key keeps today's behavior.

Validation is strict and fails fast (mirrors ``env.require_env``): a *present*
file that is malformed, has unknown brain/field keys, wrong types, or no
``default_brain`` raises a clear ``RuntimeError`` at load time rather than
silently running with the wrong config.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .base import BrainName, ToolName

# server/brains/overrides.py -> server/brains.yaml (one dir up from this package).
_OVERRIDE_PATH = Path(__file__).resolve().parents[1] / "brains.yaml"


class _Service(BaseModel):
    # ``extra="forbid"`` turns typos (e.g. ``mdoel``) into errors instead of
    # silent no-ops. ``protected_namespaces=()`` lets us name a field ``model``
    # without pydantic's "model_" namespace warning.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str | None = None


class _TTS(_Service):
    voice: str | None = None


class _Cascade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stt: _Service = Field(default_factory=_Service)
    llm: _Service = Field(default_factory=_Service)
    tts: _TTS = Field(default_factory=_TTS)


class _Realtime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The realtime model has no separate STT service; ``stt.model`` is its
    # internal transcription model, named ``stt`` for consistency with cascades.
    stt: _Service = Field(default_factory=_Service)
    llm: _Service = Field(default_factory=_Service)


class BrainOverrides(BaseModel):
    """Parsed contents of ``brains.yaml`` (or empty defaults when absent)."""

    model_config = ConfigDict(extra="forbid")

    # Optional at the model level so the no-file path can construct an empty
    # ``BrainOverrides()``; presence is enforced in the loader when a file exists.
    default_brain: BrainName | None = None

    # Optional generic tools to enable, by name (see ``ToolName``). The always-on
    # session tool is not listed. Omitted/empty -> no optional tools, matching
    # the previous "everything commented out" default. Unknown names fail
    # validation (``list[ToolName]``), like any other invalid setting.
    tools: list[ToolName] = Field(default_factory=list)

    openai_standard: _Cascade = Field(default_factory=_Cascade)
    cartesia_openai: _Cascade = Field(default_factory=_Cascade)
    openai_realtime: _Realtime = Field(default_factory=_Realtime)
    local_whisper_ollama_kokoro: _Cascade = Field(default_factory=_Cascade)


@lru_cache(maxsize=1)
def get_brain_overrides() -> BrainOverrides:
    """Load and validate ``brains.yaml`` once, or return empty defaults.

    Cached: the file is read on first access (which happens after ``load_dotenv``
    and well before any service is built), so edits require a restart -- the same
    contract as ``.env``.
    """
    if not _OVERRIDE_PATH.exists():
        return BrainOverrides()

    try:
        raw = yaml.safe_load(_OVERRIDE_PATH.read_text()) or {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"brains.yaml is not valid YAML: {e}") from e

    if not isinstance(raw, dict):
        raise RuntimeError("brains.yaml must be a mapping of brain name -> settings.")

    try:
        overrides = BrainOverrides.model_validate(raw)
    except ValidationError as e:
        raise RuntimeError(f"brains.yaml has invalid settings:\n{e}") from e

    if overrides.default_brain is None:
        raise RuntimeError(
            "brains.yaml is present but does not set 'default_brain'. "
            f"Set it to one of: {[b.value for b in BrainName]}."
        )

    return overrides


__all__ = ["BrainOverrides", "get_brain_overrides"]
