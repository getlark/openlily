"""Agent tools for the Pipecat bot.

Two flavors of tools live here:

- **Per-brain tools** (e.g. ``tools/web/``): a brain selects and owns these in its
  ``setup_tools``; they may be provider-specific (e.g. OpenAI's hosted
  ``web_search``).
- **Generic tools** (e.g. ``tools/browser/``, ``tools/email/``): brain-agnostic
  tools layered onto every brain centrally in ``bot.py``. ``setup_generic_tools``
  aggregates them into a single ``ToolBundle``; add a new generic tool by
  registering its setup here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from brains.base import ToolBundle, merge_tool_bundles

from .browser import setup_browser_tools
from .email import setup_email_tools
from .session import setup_session_tools
from .x import setup_x_tools

# The generic tools wired onto every brain. Each entry is a ``setup_*`` coroutine
# function returning a ``ToolBundle``. Edit this list to change which tools are
# included -- comment out or remove an entry to drop a tool, append one to add it.
GENERIC_TOOL_SETUPS: list[Callable[[], Awaitable[ToolBundle]]] = [
    setup_session_tools,
    # X (Twitter) search/lookup via X's hosted MCP; skipped unless
    # X_APP_BEARER_TOKEN is set.
    # setup_x_tools,
    # Note: include this if you want the agent to be able to use your local browser
    # setup_browser_tools,
    # Note: include this if you want the agent to be able to send emails
    # setup_email_tools,
]


async def setup_generic_tools() -> ToolBundle:
    """Build the brain-agnostic tools shared by every brain.

    Runs every setup in ``GENERIC_TOOL_SETUPS`` and merges the results into one
    ``ToolBundle``. To change which tools are included, edit that list rather
    than this function.
    """
    bundles = [await setup() for setup in GENERIC_TOOL_SETUPS]
    return merge_tool_bundles(*bundles)


__all__ = ["setup_generic_tools"]
