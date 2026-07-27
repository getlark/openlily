"""Runtime tool bundle data and lifecycle helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pipecat.adapters.schemas.direct_function import DirectFunction
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.services.llm_service import LLMService


@dataclass(frozen=True)
class ToolGuidance:
    """One prompt snippet paired with the LLM-visible function name(s) it covers.

    ``tool_names`` are the names the LLM sees in its tool schema (e.g.
    ``end_session``, ``web_search``, or every function an MCP server exposes),
    so the rendered guidance (see :func:`render_tool_guidance`) tells the model
    exactly which functions each snippet is about.
    """

    tool_names: tuple[str, ...]
    text: str


def render_tool_guidance(guidance: Sequence[ToolGuidance]) -> str:
    """Render guidance entries as one ``<ToolGuidance>`` block.

    Each entry becomes a ``<Tool name="...">text</Tool>`` line, with the
    covered function names joined by commas. Returns ``""`` when no tools
    contribute guidance, so consumers can inject the block unconditionally.
    """
    if not guidance:
        return ""
    entries = "\n".join(
        f'<Tool name="{", ".join(entry.tool_names)}">{entry.text.strip()}</Tool>'
        for entry in guidance
    )
    return f"<ToolGuidance>\n{entries}\n</ToolGuidance>"


@dataclass
class ToolBundle:
    """Tools contributed by one registry entry.

    Pure data: direct or advertised tools, provider-hosted tools, prompt
    guidance, LLM-dependent registrations, and lifecycle cleanups.
    """

    standard_tools: list[FunctionSchema | DirectFunction] = field(default_factory=list)
    custom_tools: dict[AdapterType, list[dict[str, Any]]] = field(default_factory=dict)
    instructions: list[ToolGuidance] = field(default_factory=list)
    registrations: list[Callable[[LLMService], Awaitable[None]]] = field(default_factory=list)
    cleanups: list[Callable[[], Awaitable[None]]] = field(default_factory=list)


def merge_tool_bundles(*bundles: ToolBundle) -> ToolBundle:
    """Concatenate several bundles into one."""
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
    """Materialize a bundle into the single schema the LLM context expects."""
    if not bundle.standard_tools and not bundle.custom_tools:
        return None
    return ToolsSchema(
        standard_tools=bundle.standard_tools,
        custom_tools=bundle.custom_tools or None,
    )


async def register_tool_bundle(bundle: ToolBundle, llm: LLMService) -> None:
    """Run LLM-dependent registrations after the LLM has been built."""
    for register in bundle.registrations:
        await register(llm)


async def close_tool_bundle(bundle: ToolBundle) -> None:
    """Run cleanups in reverse order without allowing one failure to skip others."""
    for cleanup in reversed(bundle.cleanups):
        try:
            await cleanup()
        except Exception:
            logger.exception("Tool cleanup failed")


__all__ = [
    "ToolBundle",
    "ToolGuidance",
    "close_tool_bundle",
    "merge_tool_bundles",
    "register_tool_bundle",
    "render_tool_guidance",
    "tools_schema_from_bundle",
]
