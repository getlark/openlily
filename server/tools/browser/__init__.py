"""Generic browser tool, backed by the Playwright MCP server.

Unlike the per-brain web tools in ``tools/web/``, this is brain-agnostic and
selected through the central registry. It launches Microsoft's
Playwright MCP server locally over stdio (``npx @playwright/mcp``) and exposes
its full default toolset (navigate, click, type, snapshot, ...) so the agent can
drive a real browser.

The MCP server is a child process warmed once and pooled for the process. It
doesn't launch its own browser: it attaches to an
already-running one over the Chrome DevTools Protocol (``--cdp-endpoint``, from
``BROWSER_CDP_ENDPOINT``). That browser is started and owned externally, so it
persists independently when the MCP pool closes at process shutdown.

Browser tools are opt-in. Enabling them without ``BROWSER_CDP_ENDPOINT``, without
Node.js/``npx``, or without a listening browser is a fail-fast startup error.
Direct ``setup_browser_tools`` calls retain their graceful empty-bundle behavior.
"""

from __future__ import annotations

import os

from loguru import logger
from mcp import StdioServerParameters
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.mcp_service import MCPClient

from ..bundle import ToolBundle
from ..contracts import ToolActivation, ToolBackend, ToolId, ToolName, ToolSpec
from ..mcp_bundle import mcp_tool_bundle, prefix_tool_descriptions
from .config import (
    BROWSER_CDP_ENDPOINT_ENV,
    BROWSER_MCP_COMMAND,
    build_browser_mcp_args,
    is_configured,
)

# Prompt snippet describing the browser capability. Attached to the bundle so
# the system prompt mentions the browser only when the tools are actually wired
# in (e.g. it's omitted when the MCP server fails to start).
BROWSER_INSTRUCTION = """
You can control a web browser to navigate to sites, read page contents, and take actions on a page (clicking, typing, filling forms). Your browser tools are the ones whose descriptions begin with the [Browser] tag; reach for them only when you actually need to drive a real browser. It's slower than web search, so reach for it when search and fetch fall short: when you need something precise, live details from a specific page, or when a task requires interacting with a site rather than just reading it. When using browser tool try to perform the complete the task the user gave you --  it might require multiple steps to complete the task. For example, if the users asks to search for something, it's beter to find the relevant page and opoen that that shows relevant information rather than just doing an initial google search and asking the user to navigate to the page.
"""


async def _connect_browser_mcp() -> tuple[MCPClient, ToolsSchema]:
    """Start the Playwright MCP server and return the client plus tool schemas."""
    if not os.getenv(BROWSER_CDP_ENDPOINT_ENV):
        raise RuntimeError(
            f"{BROWSER_CDP_ENDPOINT_ENV} is required for browser tools. Set it in .env, "
            "or remove 'browser' from the 'tools' list in brains.yaml."
        )

    args = build_browser_mcp_args()
    mcp = MCPClient(
        server_params=StdioServerParameters(command=BROWSER_MCP_COMMAND, args=args),
    )
    await mcp.start()
    tools = prefix_tool_descriptions(await mcp.get_tools_schema(), "Browser")
    return mcp, tools


async def setup_browser_tools() -> ToolBundle:
    """Start the Playwright MCP server and bundle its tools.

    Browser tools are opt-in: if ``BROWSER_CDP_ENDPOINT`` is unset, returns an
    empty ``ToolBundle`` without spawning the MCP server.

    Otherwise discovers the tool schemas up front (so the bundle can carry them
    and its prompt snippet before the LLM exists), and defers registering the
    handlers onto the LLM to the bundle's ``registrations`` -- run once the LLM
    is built. The bundle's cleanup closes the connection (stopping the server
    process; the externally-managed browser it attached to stays open). On any
    startup failure, logs a warning and returns an empty ``ToolBundle`` so the
    session continues without browser tools.
    """
    if not os.getenv(BROWSER_CDP_ENDPOINT_ENV):
        logger.info(
            f"Browser tools skipped: {BROWSER_CDP_ENDPOINT_ENV} not set. Set it to a "
            "running browser's CDP endpoint (e.g. http://localhost:9222) to enable."
        )
        return ToolBundle()

    args = build_browser_mcp_args()
    try:
        mcp, tools = await _connect_browser_mcp()
    except Exception:
        logger.exception(
            "Browser tools unavailable: failed to start Playwright MCP server "
            f"({BROWSER_MCP_COMMAND} {' '.join(args)}). Is Node.js/npx installed, and "
            f"is a browser listening on {BROWSER_CDP_ENDPOINT_ENV} (Chrome started with "
            "--remote-debugging-port)? Continuing without browser tools."
        )
        return ToolBundle()

    return mcp_tool_bundle(
        mcp,
        tools,
        [BROWSER_INSTRUCTION],
        close_on_cleanup=True,
        ready_log=f"Browser tools ready: {len(tools.standard_tools)} Playwright MCP tools",
    )


SPEC = ToolSpec(
    id=ToolId.BROWSER,
    activation=ToolActivation.CONFIGURED,
    backend=ToolBackend.MCP,
    setup=setup_browser_tools,
    configurable_name=ToolName.BROWSER,
    is_configured=is_configured,
    requirement="BROWSER_CDP_ENDPOINT",
    mcp_connect=_connect_browser_mcp,
    mcp_instructions=lambda: [BROWSER_INSTRUCTION],
    warmup_failure_hint=(
        "Is Node.js/npx installed, and is a browser listening on "
        "BROWSER_CDP_ENDPOINT (Chrome started with --remote-debugging-port)?"
    ),
)


__all__ = [
    "BROWSER_INSTRUCTION",
    "SPEC",
    "_connect_browser_mcp",
    "setup_browser_tools",
]
