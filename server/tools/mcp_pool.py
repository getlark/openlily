"""Process-level pool of warmed MCP tool connections.

Enabled MCP-backed generic tools (browser, notion, x) are started once at
process launch via the tool runtime and reused across sessions.
Per-session ``setup_tools()`` pulls cached schemas from the pool
instead of spawning fresh MCP servers on every wake word.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.mcp_service import MCPClient

from .bundle import ToolBundle
from .contracts import ToolBackend, ToolId, ToolSpec
from .mcp_bundle import mcp_tool_bundle


@dataclass
class _PooledMCP:
    mcp: MCPClient
    tools_schema: ToolsSchema
    instructions: list[str]


class MCPToolsPool:
    """Singleton holding warmed MCP connections for the process lifetime."""

    _instance: MCPToolsPool | None = None

    def __init__(self) -> None:
        self._pooled: dict[ToolId, _PooledMCP] = {}
        self._warmed = False

    @classmethod
    def get(cls) -> MCPToolsPool:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def warmup(self, specs: list[ToolSpec]) -> None:
        """Start selected MCP tools in parallel; fail-fast on any error."""
        if self._warmed:
            return

        enabled_mcp = [spec for spec in specs if spec.backend is ToolBackend.MCP]
        if not enabled_mcp:
            self._warmed = True
            return

        t0 = time.monotonic()
        logger.info(f"Warming up MCP tools ({', '.join(spec.id.value for spec in enabled_mcp)})...")

        async def _warm_one(spec: ToolSpec) -> tuple[ToolSpec, MCPClient, ToolsSchema]:
            assert spec.mcp_connect is not None
            try:
                mcp, schema = await spec.mcp_connect()
            except Exception as e:
                raise RuntimeError(
                    f"Tool {spec.id.value!r} is enabled in brains.yaml but failed to "
                    f"connect to its MCP server. {spec.warmup_failure_hint}"
                ) from e
            return spec, mcp, schema

        results = await asyncio.gather(
            *[_warm_one(spec) for spec in enabled_mcp],
            return_exceptions=True,
        )

        failures: list[BaseException] = []
        successful: list[tuple[ToolSpec, MCPClient, ToolsSchema]] = []
        for result in results:
            if isinstance(result, BaseException):
                failures.append(result)
            else:
                successful.append(result)

        if failures:
            # All connectors have finished. Close any that succeeded before
            # surfacing the first actionable failure, otherwise a partial
            # warmup would leak child processes or hosted connections.
            await asyncio.gather(
                *(mcp.close() for _, mcp, _ in reversed(successful)),
                return_exceptions=True,
            )
            raise failures[0]

        for spec, mcp, schema in successful:
            assert spec.mcp_instructions is not None
            self._pooled[spec.id] = _PooledMCP(
                mcp=mcp,
                tools_schema=schema,
                instructions=spec.mcp_instructions(),
            )
            logger.info(f"{spec.id.value} MCP ready: {len(schema.standard_tools)} tools")

        elapsed = time.monotonic() - t0
        logger.info(f"MCP tools warmed in {elapsed:.2f}s")
        self._warmed = True

    def is_ready(self, tool_id: ToolId) -> bool:
        return self._warmed and tool_id in self._pooled

    def session_bundle(self, tool_id: ToolId) -> ToolBundle:
        """Return a per-session bundle that reuses a pooled MCP connection."""
        pooled = self._pooled[tool_id]
        return mcp_tool_bundle(
            pooled.mcp,
            pooled.tools_schema,
            pooled.instructions,
            close_on_cleanup=False,
        )

    async def shutdown(self) -> None:
        """Close all pooled MCP connections (LIFO)."""
        for tool_id in reversed(list(self._pooled)):
            pooled = self._pooled[tool_id]
            try:
                await pooled.mcp.close()
            except Exception:
                logger.exception(f"MCP pool shutdown failed for {tool_id.value!r}")
        self._pooled.clear()
        self._warmed = False


__all__ = ["MCPToolsPool"]
