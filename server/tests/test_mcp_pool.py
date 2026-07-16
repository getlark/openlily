"""Network-free tests for registry-driven MCP pooling."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from openlily.tools.bundle import ToolBundle
from openlily.tools.contracts import (
    ToolActivation,
    ToolBackend,
    ToolId,
    ToolName,
    ToolSpec,
)
from openlily.tools.mcp_pool import MCPToolsPool


async def _empty_setup() -> ToolBundle:
    return ToolBundle()


@pytest.fixture
def pool() -> Iterator[MCPToolsPool]:
    MCPToolsPool._instance = None
    instance = MCPToolsPool.get()
    yield instance
    MCPToolsPool._instance = None


def _mcp_spec(connect: AsyncMock) -> ToolSpec:
    return ToolSpec(
        id=ToolId.X,
        activation=ToolActivation.CONFIGURED,
        backend=ToolBackend.MCP,
        setup=_empty_setup,
        configurable_name=ToolName.X,
        is_configured=lambda: True,
        requirement="X_APP_BEARER_TOKEN",
        mcp_connect=connect,
        mcp_instructions=lambda: ["X capability"],
        warmup_failure_hint="Check the X credentials.",
    )


async def test_warmup_reuses_connection_for_session_bundle(pool: MCPToolsPool) -> None:
    mcp = AsyncMock()
    schema = ToolsSchema(standard_tools=[])
    connect = AsyncMock(return_value=(mcp, schema))

    await pool.warmup([_mcp_spec(connect)])
    await pool.warmup([_mcp_spec(connect)])

    assert connect.await_count == 1
    assert pool.is_ready(ToolId.X)
    bundle = pool.session_bundle(ToolId.X)
    assert bundle.instructions == ["X capability"]
    assert len(bundle.cleanups) == 1
    await bundle.cleanups[0]()
    mcp.close.assert_not_awaited()

    await pool.shutdown()
    mcp.close.assert_awaited_once()
    assert not pool.is_ready(ToolId.X)


async def test_warmup_failure_uses_registry_hint(pool: MCPToolsPool) -> None:
    connect = AsyncMock(side_effect=OSError("offline"))

    with pytest.raises(RuntimeError) as exc:
        await pool.warmup([_mcp_spec(connect)])

    assert "Check the X credentials." in str(exc.value)
    assert not pool.is_ready(ToolId.X)


async def test_partial_warmup_closes_successful_connections(pool: MCPToolsPool) -> None:
    mcp = AsyncMock()
    successful_connect = AsyncMock(return_value=(mcp, ToolsSchema(standard_tools=[])))
    failed_connect = AsyncMock(side_effect=OSError("offline"))
    browser_spec = replace(
        _mcp_spec(successful_connect),
        id=ToolId.BROWSER,
        configurable_name=ToolName.BROWSER,
    )

    with pytest.raises(RuntimeError):
        await pool.warmup([browser_spec, _mcp_spec(failed_connect)])

    mcp.close.assert_awaited_once()
    assert not pool.is_ready(ToolId.BROWSER)
