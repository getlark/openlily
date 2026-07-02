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
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from loguru import logger
from pipecat.adapters.schemas.direct_function import DirectFunction
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.services.llm_service import LLMService


class BrainName(StrEnum):
    """Canonical names of the selectable brains -- the single source of truth.

    Used everywhere a brain is named (each ``SPEC.name``, the ``BRAINS`` registry
    keys, the ``default_brain`` override, and ``DEFAULT_BRAIN``). ``StrEnum``
    members are real strings, so they stay drop-in for logging, dict keys, and
    matching YAML/env values, while giving the type checker and pydantic one
    definition to enforce. Add a brain by adding a member here, registering its
    ``SPEC`` in ``brains/__init__.py``, and (optionally) a section in
    ``brains/overrides.py``.
    """

    OPENAI_STANDARD = "openai_standard"
    CARTESIA_OPENAI = "cartesia_openai"
    OPENAI_REALTIME = "openai_realtime"
    LOCAL_WHISPER_OLLAMA_KOKORO = "local_whisper_ollama_kokoro"


@dataclass
class BrainServices:
    """The service objects a brain contributes to the pipeline.

    For a realtime brain only ``llm`` is set (the speech-to-speech service does
    STT + LLM + TTS itself); ``stt`` and ``tts`` stay ``None``.
    """

    llm: Any
    stt: Any | None = None
    tts: Any | None = None


@dataclass
class ToolBundle:
    """Tools contributed by one source (a brain provider or a generic tool).

    Pure data -- the raw ingredients of an LLM tool set, kept separate so
    bundles merge by plain concatenation:

    - ``standard_tools``: direct-function callables and/or advertise-only
      ``FunctionSchema``s (e.g. MCP tools, whose handlers are registered on the
      LLM via ``registrations``).
    - ``custom_tools``: provider-hosted tools keyed by adapter (e.g. OpenAI's
      hosted ``web_search``).
    - ``instructions``: short prompt snippets describing this source's tools, so
      the system prompt mentions a capability only when it's actually wired in.
    - ``registrations``: async callbacks that wire LLM-dependent handlers onto
      the built LLM (e.g. MCP's ``register_tools_schema``). Run once the LLM
      exists, since the system prompt -- which depends on ``instructions`` -- has
      to be composed before the LLM is constructed.
    - ``cleanups``: async callbacks to release resources at session end (e.g.
      closing an MCP connection).

    Merging, conversion to a ``ToolsSchema``, registration, and cleanup all live
    in the free functions below, so this stays a dumb container.
    """

    standard_tools: list[FunctionSchema | DirectFunction] = field(default_factory=list)
    custom_tools: dict[AdapterType, list[dict[str, Any]]] = field(default_factory=dict)
    instructions: list[str] = field(default_factory=list)
    registrations: list[Callable[[LLMService], Awaitable[None]]] = field(default_factory=list)
    cleanups: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


def merge_tool_bundles(*bundles: ToolBundle) -> ToolBundle:
    """Concatenate several bundles into one.

    Standard tools, instructions, registrations, and cleanups are appended in
    order; custom tools are merged per adapter type. Callables are left
    untouched (not converted to schemas), so no handlers are lost in the merge.
    """
    merged = ToolBundle()
    for bundle in bundles:
        merged.standard_tools.extend(bundle.standard_tools)
        for adapter_type, tools in bundle.custom_tools.items():
            merged.custom_tools.setdefault(adapter_type, []).extend(tools)
        merged.instructions.extend(bundle.instructions)
        merged.registrations.extend(bundle.registrations)
        merged.cleanups.extend(bundle.cleanups)
    return merged


def tools_schema_from_bundle(bundle: ToolBundle) -> ToolsSchema | None:
    """Materialize a bundle into the single ``ToolsSchema`` ``LLMContext`` wants.

    Returns ``None`` when the bundle has no tools, so the caller can build a
    context with no tools at all.
    """
    if not bundle.standard_tools and not bundle.custom_tools:
        return None
    return ToolsSchema(
        standard_tools=bundle.standard_tools,
        custom_tools=bundle.custom_tools or None,
    )


async def register_tool_bundle(bundle: ToolBundle, llm: LLMService) -> None:
    """Run a bundle's LLM-dependent registrations against the built ``llm``.

    Used by tools (e.g. MCP) that must register their handlers on the LLM, which
    only exists after the system prompt is composed from ``instructions``.
    """
    for register in bundle.registrations:
        await register(llm)


async def close_tool_bundle(bundle: ToolBundle) -> None:
    """Run a bundle's cleanups in reverse (LIFO) order, each guarded.

    A failure in one cleanup is logged and doesn't skip the rest.
    """
    for cleanup in reversed(bundle.cleanups):
        try:
            await cleanup()
        except Exception:
            logger.exception("Tool cleanup failed")


@dataclass(frozen=True)
class BrainSpec:
    """Describes a selectable brain for the agent."""

    name: BrainName
    is_realtime: bool
    """When True, the pipeline omits STT/TTS (the LLM is a speech-to-speech model)."""
    build: Callable[[str], BrainServices]
    """Build the services. Receives the system instruction string."""
    setup_tools: Callable[[], Awaitable[ToolBundle]] | None = None
    """Optional async factory for this brain's tools. Returns a ``ToolBundle``
    with the tools, prompt snippets, and any registrations/cleanups. Runs before
    the LLM is built (the prompt depends on the bundle's ``instructions``), so it
    takes no ``llm``; tools that must register on the LLM do so via the bundle's
    ``registrations``. Only this brain's tools apply."""
    warmup: Callable[[], Awaitable[None]] | None = None
    """Optional: eagerly download/load this brain's slow first-run resources
    once at process startup, so nothing downloads or cold-starts mid-session
    (and, in wake-gated mode, progress is visible before the wake word). Called
    by ``bot.py`` before any session. Fail-fast: raise a clear, actionable error
    on a known-broken setup (e.g. a required local server not running) rather
    than deferring to a confusing mid-conversation failure. Brains without slow
    startup work leave it ``None``."""
