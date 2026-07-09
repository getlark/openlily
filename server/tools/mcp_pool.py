"""Process-level pool of warmed MCP tool connections.

Enabled MCP-backed generic tools (browser, notion, x) are started once at
process launch via ``warmup_generic_tools()`` and reused across sessions.
Per-session ``setup_generic_tools()`` pulls cached schemas from the pool
instead of spawning fresh MCP servers on every wake word.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.mcp_service import MCPClient

from brains.base import ToolBundle, ToolName
from brains.config import get_enabled_tools

from .mcp_bundle import mcp_tool_bundle

_MCP_OPTIONAL_TOOLS = frozenset({ToolName.BROWSER, ToolName.NOTION, ToolName.X})
MCP_OPTIONAL_TOOLS = _MCP_OPTIONAL_TOOLS


def _mcp_connectors() -> dict[
    ToolName,
    tuple[Callable[[], Awaitable[tuple[MCPClient, ToolsSchema]]], list[str]],
]:
    """Lazy import to avoid circular imports with tool setup modules."""
    from .browser import BROWSER_INSTRUCTION, _connect_browser_mcp
    from .notion import NOTION_INSTRUCTION, _connect_notion_mcp
    from .x import X_INSTRUCTION, _connect_x_mcp

    return {
        ToolName.BROWSER: (_connect_browser_mcp, [BROWSER_INSTRUCTION]),
        ToolName.NOTION: (_connect_notion_mcp, [NOTION_INSTRUCTION]),
        ToolName.X: (_connect_x_mcp, [X_INSTRUCTION]),
    }


# Actionable fail-fast messages when warmup cannot connect an enabled tool.
_WARMUP_FAILURE_HINTS: dict[ToolName, str] = {
    ToolName.BROWSER: (
        "Is Node.js/npx installed, and is a browser listening on BROWSER_CDP_ENDPOINT "
        "(Chrome started with --remote-debugging-port)?"
    ),
    ToolName.NOTION: (
        "Is Node.js/npx installed, and is NOTION_ACCESS_TOKEN a valid integration token?"
    ),
    ToolName.X: (
        "Is X_APP_BEARER_TOKEN a valid App-only Bearer token, and is the network reachable?"
    ),
}


@dataclass
class _PooledMCP:
    mcp: MCPClient
    tools_schema: ToolsSchema
    instructions: list[str]


class MCPToolsPool:
    """Singleton holding warmed MCP connections for the process lifetime."""

    _instance: MCPToolsPool | None = None

    def __init__(self) -> None:
        self._pooled: dict[ToolName, _PooledMCP] = {}
        self._warmed = False

    @classmethod
    def get(cls) -> MCPToolsPool:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def warmup(self) -> None:
        """Start all enabled MCP tools in parallel; fail-fast on any error."""
        if self._warmed:
            return

        enabled_mcp = [name for name in get_enabled_tools() if name in _MCP_OPTIONAL_TOOLS]
        if not enabled_mcp:
            self._warmed = True
            return

        t0 = time.monotonic()
        logger.info(f"Warming up MCP tools ({', '.join(n.value for n in enabled_mcp)})...")

        connectors = _mcp_connectors()

        async def _warm_one(name: ToolName) -> tuple[ToolName, MCPClient, ToolsSchema]:
            connect, _ = connectors[name]
            try:
                mcp, schema = await connect()
            except Exception as e:
                hint = _WARMUP_FAILURE_HINTS.get(name, "")
                raise RuntimeError(
                    f"Tool {name.value!r} is enabled in brains.yaml but failed to "
                    f"connect to its MCP server. {hint}"
                ) from e
            return name, mcp, schema

        results = await asyncio.gather(*[_warm_one(name) for name in enabled_mcp])

        for name, mcp, schema in results:
            _, instructions = connectors[name]
            self._pooled[name] = _PooledMCP(
                mcp=mcp,
                tools_schema=schema,
                instructions=instructions,
            )
            logger.info(f"{name.value} MCP ready: {len(schema.standard_tools)} tools")

        elapsed = time.monotonic() - t0
        logger.info(f"MCP tools warmed in {elapsed:.2f}s")
        self._warmed = True

    def is_ready(self, name: ToolName) -> bool:
        return self._warmed and name in self._pooled

    def session_bundle(self, name: ToolName) -> ToolBundle:
        """Return a per-session bundle that reuses a pooled MCP connection."""
        pooled = self._pooled[name]
        return mcp_tool_bundle(
            pooled.mcp,
            pooled.tools_schema,
            pooled.instructions,
            close_on_cleanup=False,
        )

    async def shutdown(self) -> None:
        """Close all pooled MCP connections (LIFO)."""
        for name in reversed(list(self._pooled)):
            pooled = self._pooled[name]
            try:
                await pooled.mcp.close()
            except Exception:
                logger.exception(f"MCP pool shutdown failed for {name.value!r}")
        self._pooled.clear()
        self._warmed = False


async def _warmup_mcp_pool() -> None:
    """Eagerly start enabled MCP tools once per process."""
    await MCPToolsPool.get().warmup()


async def _shutdown_mcp_pool() -> None:
    """Close pooled MCP connections at process exit."""
    await MCPToolsPool.get().shutdown()


def is_mcp_tool_pooled(name: ToolName) -> bool:
    """Whether ``name`` is an MCP-backed tool with a warmed pool entry."""
    return MCPToolsPool.get().is_ready(name)


def pooled_session_bundle(name: ToolName) -> ToolBundle:
    """Session bundle from the pool; caller must ensure ``is_ready(name)``."""
    return MCPToolsPool.get().session_bundle(name)


__all__ = [
    "MCPToolsPool",
    "MCP_OPTIONAL_TOOLS",
    "is_mcp_tool_pooled",
    "pooled_session_bundle",
    "_shutdown_mcp_pool",
    "_warmup_mcp_pool",
]
