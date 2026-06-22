"""Web search tools for the voice agent.

Selects a web-search provider (Exa for now) and asks it to create the agent's
web tools. Each provider owns its tools' schemas, so the schema matches exactly
what that provider supports. Adding a provider is a new module under
``tools/web/`` plus a registry entry here and a line in ``config.py``.

The tools are wired in only when the selected provider's credentials are
configured; otherwise ``setup_web_tools`` logs a warning and returns an empty
bundle, so the session runs without web search -- the same graceful degradation
the email tool uses.
"""

from __future__ import annotations

from loguru import logger

from brains.base import ToolBundle

from ..base import ToolProvider
from .config import WEB_SEARCH_PROVIDER
from .exa import ExaProvider

# Prompt snippet describing the web search/fetch capability. Provider-agnostic
# (the model just needs to know the capability exists), so both the hosted
# web_search brains and the Exa-tool brain attach it to their bundle.
WEB_SEARCH_INSTRUCTION = (
    "You can search the web and read web pages to answer questions that need "
    "external information. These are fast, so prefer them for quick, "
    "casual lookups if you have multiple options."
)

# Registry of available providers. Add new ones here; select via config.py.
_PROVIDERS: dict[str, type[ToolProvider]] = {
    "exa": ExaProvider,
}


def _provider_cls() -> type[ToolProvider]:
    try:
        return _PROVIDERS[WEB_SEARCH_PROVIDER]
    except KeyError:
        raise ValueError(
            f"Unknown web search provider {WEB_SEARCH_PROVIDER!r}; available: {sorted(_PROVIDERS)}"
        )


def setup_web_tools() -> ToolBundle:
    """Build the web tools from the selected provider, if it's configured.

    Returns an empty ``ToolBundle`` (so the session continues without web
    search) when the selected provider's credentials are missing, logging a
    warning that says what to set.
    """
    provider_cls = _provider_cls()

    if not provider_cls.is_configured():
        logger.warning(
            f"Web search tool unavailable: set the {WEB_SEARCH_PROVIDER!r} "
            "provider's credentials to enable it. Continuing without web search."
        )
        return ToolBundle()

    tools = provider_cls().create_tools()

    logger.info(f"Web search tool ready (provider={WEB_SEARCH_PROVIDER})")
    return ToolBundle(
        standard_tools=list(tools),
        instructions=[WEB_SEARCH_INSTRUCTION],
    )


__all__ = ["WEB_SEARCH_INSTRUCTION", "setup_web_tools"]
