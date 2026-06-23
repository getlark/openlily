"""Generic browser tool, backed by the Playwright MCP server.

Unlike the per-brain web tools in ``tools/web/``, this is brain-agnostic: it's
layered onto every brain centrally in ``bot.py``. It launches Microsoft's
Playwright MCP server locally over stdio (``npx @playwright/mcp``) and exposes
its full default toolset (navigate, click, type, snapshot, ...) so the agent can
drive a real browser.

The MCP server is a child process started per session and torn down via the
returned bundle's cleanup. It doesn't launch its own browser: it attaches to an
already-running one over the Chrome DevTools Protocol (``--cdp-endpoint``, from
``BROWSER_CDP_ENDPOINT``). That browser is started and owned externally, so it
persists across sessions -- tearing down the per-session MCP server leaves the
browser open, and the next session reconnects to it.

Browser tools are opt-in: when ``BROWSER_CDP_ENDPOINT`` is unset they're skipped
without spawning the server. They also require Node.js/``npx`` on the host; if
that's missing (or the server fails to connect, e.g. no browser is listening on
the endpoint), the tools are skipped and the session runs without them.
"""

from __future__ import annotations

import os

from loguru import logger
from mcp import StdioServerParameters
from pipecat.services.llm_service import LLMService
from pipecat.services.mcp_service import MCPClient

from brains.base import ToolBundle

from .config import (
    BROWSER_CDP_ENDPOINT_ENV,
    BROWSER_MCP_COMMAND,
    build_browser_mcp_args,
)

# Prompt snippet describing the browser capability. Attached to the bundle so
# the system prompt mentions the browser only when the tools are actually wired
# in (e.g. it's omitted when the MCP server fails to start).
BROWSER_INSTRUCTION = """
You can control a web browser to navigate to sites, read page contents, and take actions on a page (clicking, typing, filling forms). It's slower than web search, so reach for it when search and fetch fall short: when you need something precise, live details from a specific page, or when a task requires interacting with a site rather than just reading it. When using browser tool try to perform the complete the task the user gave you --  it might require multiple steps to complete the task. For example, if the users asks to search for something, it's beter to find the relevant page and opoen that that shows relevant information rather than just doing an initial google search and asking the user to navigate to the page.
"""


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
    mcp = MCPClient(
        server_params=StdioServerParameters(command=BROWSER_MCP_COMMAND, args=args),
    )
    try:
        await mcp.start()
        tools = await mcp.get_tools_schema()
    except Exception:
        logger.exception(
            "Browser tools unavailable: failed to start Playwright MCP server "
            f"({BROWSER_MCP_COMMAND} {' '.join(args)}). Is Node.js/npx installed, and "
            f"is a browser listening on {BROWSER_CDP_ENDPOINT_ENV} (Chrome started with "
            "--remote-debugging-port)? Continuing without browser tools."
        )
        await mcp.close()
        return ToolBundle()

    async def register(llm: LLMService) -> None:
        await mcp.register_tools_schema(tools, llm)

    logger.info(
        f"Browser tools ready: {len(tools.standard_tools)} Playwright MCP tools"
    )
    return ToolBundle(
        standard_tools=list(tools.standard_tools),
        instructions=[BROWSER_INSTRUCTION],
        registrations=[register],
        cleanups=[mcp.close],
    )


__all__ = ["BROWSER_INSTRUCTION", "setup_browser_tools"]
