"""Tool lifecycle orchestration driven by the central registry."""

from __future__ import annotations

import time
from collections.abc import Iterable

from loguru import logger

from .bundle import ToolBundle, merge_tool_bundles
from .contracts import ToolBackend, ToolId, ToolName, ToolSpec
from .mcp_pool import MCPToolsPool
from .registry import (
    always_on_tools,
    get_configurable_tool,
    get_tool_spec,
)


def _enabled_specs(enabled_tool_names: Iterable[ToolName | str] = ()) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for name in enabled_tool_names:
        tool_name = name if isinstance(name, ToolName) else ToolName(name)
        specs.append(get_configurable_tool(tool_name))
    return specs


def _assert_configured(specs: Iterable[ToolSpec]) -> None:
    for spec in specs:
        if spec.is_configured is not None and not spec.is_configured():
            name = spec.configurable_name
            assert name is not None
            raise RuntimeError(
                f"Tool {name.value!r} is enabled in brains.yaml but is not "
                f"configured. Set {spec.requirement} (in .env), or remove "
                f"{name.value!r} from the 'tools' list."
            )


def _selected_specs(
    brain_tool_ids: Iterable[ToolId | str],
    enabled_tool_names: Iterable[ToolName | str] = (),
) -> list[ToolSpec]:
    specs = [*always_on_tools(), *(get_tool_spec(tool_id) for tool_id in brain_tool_ids)]
    specs.extend(_enabled_specs(enabled_tool_names))

    # A malformed BrainSpec should not wire a tool twice. Keep registry order
    # stable while deduplicating by canonical ID.
    unique: dict[ToolId | str, ToolSpec] = {}
    for spec in specs:
        unique.setdefault(spec.id, spec)
    return list(unique.values())


async def setup_tools(
    brain_tool_ids: Iterable[ToolId | str] = (),
    enabled_tool_names: Iterable[ToolName | str] = (),
) -> ToolBundle:
    """Build all tools active for one brain and one session.

    ``brain_tool_ids`` are the tool ids a brain declares (``BrainSpec.tools``);
    ``enabled_tool_names`` are the optional configurable tools to turn on (what
    ``brains.yaml``'s ``tools:`` list holds for the local CLI). The always-on
    tools are added automatically.
    """
    t0 = time.monotonic()
    specs = _selected_specs(brain_tool_ids, enabled_tool_names)
    _assert_configured(specs)

    pool = MCPToolsPool.get()
    bundles: list[ToolBundle] = []
    for spec in specs:
        if spec.backend is ToolBackend.MCP:
            if not pool.is_ready(spec.id):
                raise RuntimeError(
                    f"Tool {spec.id.value!r} is enabled but not in the MCP pool. "
                    "warmup_tools() must run before setup_tools()."
                )
            bundles.append(pool.session_bundle(spec.id))
        else:
            bundles.append(await spec.setup())

    elapsed = time.monotonic() - t0
    logger.info(f"Tools bundle ready in {elapsed:.2f}s")
    return merge_tool_bundles(*bundles)


async def warmup_tools(
    brain_tool_ids: Iterable[ToolId | str] = (),
    enabled_tool_names: Iterable[ToolName | str] = (),
) -> None:
    """Validate enabled tools and eagerly start selected MCP connections."""
    enabled_specs = _enabled_specs(enabled_tool_names)
    _assert_configured(enabled_specs)
    selected = [*(get_tool_spec(tool_id) for tool_id in brain_tool_ids), *enabled_specs]
    await MCPToolsPool.get().warmup(selected)


async def shutdown_tools() -> None:
    """Close process-scoped tool resources."""
    await MCPToolsPool.get().shutdown()


__all__ = [
    "setup_tools",
    "shutdown_tools",
    "warmup_tools",
]
