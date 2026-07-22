"""Shared types for swappable "brains" (LLM harnesses).

A *brain* bundles everything that varies between a cascade pipeline
(STT -> LLM -> TTS) and a realtime speech-to-speech model: how to build the
services, whether the pipeline needs separate STT/TTS, and which tools the LLM
gets. The pipeline shape in ``bot.py`` is chosen from ``BrainSpec.is_realtime``.

Add a brain by creating ``brains/<name>.py`` that exposes a
``build(system_instruction) -> BrainServices`` function and a ``SPEC`` of type
``BrainSpec``, then register its ``SPEC`` in ``brains/__init__.py``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from openlily.tools.contracts import ToolId


class BrainName(StrEnum):
    """Canonical names of the selectable brains -- the single source of truth.

    Used everywhere a brain is named (each ``SPEC.name``, the brain-loader
    registry keys in ``brains/__init__.py``, the ``default_brain`` override, and
    ``DEFAULT_BRAIN``). ``StrEnum``
    members are real strings, so they stay drop-in for logging, dict keys, and
    matching YAML/env values, while giving the type checker and pydantic one
    definition to enforce. Add a brain by adding a member here, registering its
    ``SPEC`` in ``brains/__init__.py``, and (optionally) a section in
    ``brains/overrides.py``.
    """

    OPENAI_STANDARD = "openai_standard"
    CARTESIA_OPENAI = "cartesia_openai"
    CARTESIA_META = "cartesia_meta"
    OPENAI_REALTIME = "openai_realtime"
    LOCAL_WHISPER_OLLAMA_KOKORO = "local_whisper_ollama_kokoro"


@dataclass
class BrainServices:
    """The service objects a brain contributes to the pipeline.

    For a realtime brain only ``llm`` is set (the speech-to-speech service does
    STT + LLM + TTS itself); ``stt`` and ``tts`` stay ``None``.
    """

    llm: LLMService[Any]
    stt: STTService | None = None
    tts: TTSService | None = None


@dataclass(frozen=True)
class BrainSpec:
    """Describes a selectable brain for the agent."""

    name: BrainName
    is_realtime: bool
    """When True, the pipeline omits STT/TTS (the LLM is a speech-to-speech model)."""
    build: Callable[[str], BrainServices]
    """Build the services. Receives the system instruction string."""
    tools: tuple[ToolId, ...] = ()
    """Registry tool IDs selected by this brain. Always-on and user-configured
    tools are added centrally by the tool runtime."""
    warmup: Callable[[], Awaitable[None]] | None = None
    """Optional: eagerly download/load this brain's slow first-run resources
    once at process startup, so nothing downloads or cold-starts mid-session
    (and, in wake-gated mode, progress is visible before the wake word). Called
    by ``bot.py`` before any session. Fail-fast: raise a clear, actionable error
    on a known-broken setup (e.g. a required local server not running) rather
    than deferring to a confusing mid-conversation failure. Brains without slow
    startup work leave it ``None``."""
