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
from pipecat.adapters.schemas.tools_schema import AdapterType

from ..base import ToolProvider
from ..bundle import ToolBundle
from ..contracts import ToolActivation, ToolBackend, ToolId, ToolSpec
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


def hosted_web_search_bundle(search_context_size: str = "low") -> ToolBundle:
    """Bundle a provider-hosted ``web_search`` tool (no local handler).

    ``web_search`` is a server-side tool of the OpenAI-style Responses API: the
    model runs the search itself and reads the results, so there's nothing to
    register or clean up. Shared by every brain whose LLM speaks that API --
    OpenAI's own models (``openai_standard``, ``cartesia_openai``) and Meta's
    OpenAI-compatible Responses endpoint (``cartesia_meta``) -- so no brain has
    to import another's tool setup.

    The adapter key stays ``AdapterType.OPENAI`` even for Meta: it just tags the
    tool as OpenAI-shaped, which Meta's compatible API accepts. ``search_context_size``
    (``low`` | ``medium`` | ``high``) trades breadth for latency; the ``low``
    default keeps voice turns fast and concise.
    """
    web_search = {"type": "web_search", "search_context_size": search_context_size}
    return ToolBundle(
        # Keyed by wire *protocol*, not vendor: AdapterType.OPENAI is the bucket the
        # OpenAI adapter reads (Chat Completions / Responses / Realtime), so every
        # OpenAI-shaped LLM finds the tool here -- including Meta's compatible API.
        # Tools under any other adapter key are ignored by the OpenAI adapter.
        custom_tools={AdapterType.OPENAI: [web_search]},
        instructions=[WEB_SEARCH_INSTRUCTION],
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


async def _setup_hosted_web() -> ToolBundle:
    return hosted_web_search_bundle()


async def _setup_exa_web() -> ToolBundle:
    return setup_web_tools()


HOSTED_WEB_SPEC = ToolSpec(
    id=ToolId.WEB_HOSTED,
    activation=ToolActivation.BRAIN,
    backend=ToolBackend.HOSTED,
    setup=_setup_hosted_web,
)

EXA_WEB_SPEC = ToolSpec(
    id=ToolId.WEB_EXA,
    activation=ToolActivation.BRAIN,
    backend=ToolBackend.LOCAL,
    setup=_setup_exa_web,
)


__all__ = [
    "EXA_WEB_SPEC",
    "HOSTED_WEB_SPEC",
    "WEB_SEARCH_INSTRUCTION",
    "hosted_web_search_bundle",
    "setup_web_tools",
]
