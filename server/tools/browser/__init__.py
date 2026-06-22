"""Generic browser tool, backed by the Playwright MCP server.

Unlike the per-brain web tools in ``tools/web/``, this is brain-agnostic: it's
layered onto every brain centrally in ``bot.py``. It launches Microsoft's
Playwright MCP server locally over stdio (``npx @playwright/mcp``) and exposes
its full default toolset (navigate, click, type, snapshot, ...) so the agent can
drive a real browser.

The MCP server is a child process started per session and torn down via the
returned bundle's cleanup. Requires Node.js/``npx`` on the host; if it's missing
(or the server fails to start), the tools are skipped and the session runs
without them.
"""

from __future__ import annotations

from loguru import logger
from mcp import StdioServerParameters
from pipecat.services.llm_service import LLMService
from pipecat.services.mcp_service import MCPClient

from brains.base import ToolBundle

from .config import BROWSER_MCP_COMMAND, build_browser_mcp_args

# Prompt snippet describing the browser capability. Attached to the bundle so
# the system prompt mentions the browser only when the tools are actually wired
# in (e.g. it's omitted when the MCP server fails to start).
BROWSER_INSTRUCTION = (
    "You can control a web browser to navigate to sites, read page contents, "
    "and take actions on a page (clicking, typing, filling forms). It's slower "
    "than web search, so reach for it when search and fetch fall short: when you "
    "need something precise, live details from a specific page, or when a task requires "
    "interacting with a site rather than just reading it."
)


async def setup_browser_tools() -> ToolBundle:
    """Start the Playwright MCP server and bundle its tools.

    Discovers the tool schemas up front (so the bundle can carry them and its
    prompt snippet before the LLM exists), and defers registering the handlers
    onto the LLM to the bundle's ``registrations`` -- run once the LLM is built.
    The bundle's cleanup closes the connection (stopping the server process). On
    any startup failure, logs a warning and returns an empty ``ToolBundle`` so
    the session continues without browser tools.
    """
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
            f"({BROWSER_MCP_COMMAND} {' '.join(args)}). "
            "Is Node.js/npx installed? Continuing without browser tools."
        )
        await mcp.close()
        return ToolBundle()

    async def register(llm: LLMService) -> None:
        await mcp.register_tools_schema(tools, llm)

    logger.info(f"Browser tools ready: {len(tools.standard_tools)} Playwright MCP tools")
    return ToolBundle(
        standard_tools=list(tools.standard_tools),
        instructions=[BROWSER_INSTRUCTION],
        registrations=[register],
        cleanups=[mcp.close],
    )


__all__ = ["BROWSER_INSTRUCTION", "setup_browser_tools"]
