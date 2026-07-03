"""Agent tools for the Pipecat bot.

Two flavors of tools live here:

- **Per-brain tools** (e.g. ``tools/web/``): a brain selects and owns these in its
  ``setup_tools``; they may be provider-specific (e.g. OpenAI's hosted
  ``web_search``).
- **Generic tools** (e.g. ``tools/browser/``, ``tools/email/``): brain-agnostic
  tools layered onto every brain centrally in ``bot.py``. ``setup_generic_tools``
  aggregates them into a single ``ToolBundle``.

Which generic tools run is config-driven, not code-driven: the ``session`` tool
is always on (see ``_ALWAYS_ON_SETUPS``), and the *optional* ones are enabled by
name via the ``tools`` list in ``brains.yaml`` (see ``brains/overrides.py`` and
``get_enabled_tools``). Enabling a tool whose credentials are missing is a
fail-fast startup error rather than a silent skip -- if you asked for it, we
won't quietly run without it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from brains.base import ToolBundle, ToolName, merge_tool_bundles
from brains.config import get_enabled_tools

from .browser import setup_browser_tools
from .browser.config import is_configured as browser_is_configured
from .email import email_is_configured, setup_email_tools
from .session import setup_session_tools
from .x import setup_x_tools
from .x.config import is_configured as x_is_configured

# Always-on generic tools, wired onto every brain regardless of config. The
# session (end-session) tool is a critical control, not an opt-in capability, so
# it lives here and can't be disabled via brains.yaml.
_ALWAYS_ON_SETUPS: list[Callable[[], Awaitable[ToolBundle]]] = [
    setup_session_tools,
]


@dataclass(frozen=True)
class _OptionalTool:
    """An optional generic tool: how to build it, and whether it's usable.

    ``is_configured`` is checked *before* ``setup`` so an enabled-but-unconfigured
    tool fails fast with an actionable message (naming ``requirement``) rather
    than attempting setup and silently degrading.
    """

    setup: Callable[[], Awaitable[ToolBundle]]
    is_configured: Callable[[], bool]
    # Human-readable hint naming what to set, for the fail-fast error message.
    requirement: str


# The optional generic tools, keyed by the name used in brains.yaml's ``tools``
# list. Add a new optional tool by implementing its ``setup_*`` + a config-presence
# check, adding a ``ToolName`` member, and registering it here.
_OPTIONAL_TOOLS: dict[ToolName, _OptionalTool] = {
    ToolName.BROWSER: _OptionalTool(
        setup=setup_browser_tools,
        is_configured=browser_is_configured,
        requirement="BROWSER_CDP_ENDPOINT",
    ),
    ToolName.EMAIL: _OptionalTool(
        setup=setup_email_tools,
        is_configured=email_is_configured,
        requirement="USER_EMAIL and the email provider's credentials",
    ),
    ToolName.X: _OptionalTool(
        setup=setup_x_tools,
        is_configured=x_is_configured,
        requirement="X_APP_BEARER_TOKEN",
    ),
}


async def setup_generic_tools() -> ToolBundle:
    """Build the brain-agnostic tools: the always-on set plus the enabled optionals.

    The always-on tools (e.g. session) run unconditionally. The optional tools
    are those named in ``brains.yaml``'s ``tools`` list (see ``get_enabled_tools``);
    each is checked for its credentials first and raises a ``RuntimeError`` if
    it's enabled but unconfigured -- a fail-fast startup error, surfaced through
    ``_build_pipeline`` in ``bot.py``. Results are merged into one ``ToolBundle``.
    """
    bundles = [await setup() for setup in _ALWAYS_ON_SETUPS]

    for name in get_enabled_tools():
        tool = _OPTIONAL_TOOLS[name]
        if not tool.is_configured():
            raise RuntimeError(
                f"Tool {name.value!r} is enabled in brains.yaml but is not "
                f"configured. Set {tool.requirement} (in .env), or remove "
                f"{name.value!r} from the 'tools' list."
            )
        bundles.append(await tool.setup())

    return merge_tool_bundles(*bundles)


__all__ = ["setup_generic_tools"]
