"""Exa-backed web search provider.

Owns the agent's ``web_search``/``web_fetch`` tools and implements them with the
official Exa Python SDK (``exa-py``), so the tools' schemas and behavior match
what Exa supports and we control the knobs that matter for a voice assistant:
content freshness, small token-efficient excerpts, and bounded latency.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable

from exa_py import AsyncExa
from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from ..base import ToolProvider

# Defaults for the model-controllable knobs. Small on purpose: a spoken answer
# needs a few good sources with short excerpts, not a page of links, which keeps
# context small and the turn fast. The model can raise them when a task needs
# broader coverage or more detail.
_DEFAULT_NUM_RESULTS = 8
_DEFAULT_HIGHLIGHT_MAX_CHARS = 400
_DEFAULT_TEXT_MAX_CHARS = 3000
# Upper bound on results so a stray large value can't blow up latency/context
# (Exa itself allows up to 100).
_MAX_NUM_RESULTS = 25
# Bound a live crawl so a fresh fetch can't stall the conversation turn (ms).
_LIVECRAWL_TIMEOUT_MS = 8000
# Overall per-call timeout. The SDK's HTTP client defaults to a 600s timeout,
# far too long for a voice turn, so we bound each call ourselves.
_CALL_TIMEOUT_S = 12.0

_DEFAULT_FRESHNESS = "recent"
# Map the freshness knob to Exa's ``max_age_hours`` -- the max age of cached
# content in hours, where 0 = always livecrawl and -1 = cache only.
_MAX_AGE_HOURS = {"live": 0, "recent": 24, "any": -1}

# Exa's only credential, read from the environment. ``is_configured`` reports
# whether it's present so the web tools can be skipped with a warning when it
# isn't, rather than raising at construction.
EXA_API_KEY_ENV = "EXA_API_KEY"


def _max_age_hours(freshness: str) -> int:
    return _MAX_AGE_HOURS.get(freshness, _MAX_AGE_HOURS[_DEFAULT_FRESHNESS])


class ExaProvider(ToolProvider):
    """Web search/fetch tools backed by the Exa Python SDK."""

    def __init__(self) -> None:
        api_key = os.getenv(EXA_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"{EXA_API_KEY_ENV} is required for the Exa web search provider")
        self._exa = AsyncExa(api_key=api_key)

    @classmethod
    def is_configured(cls) -> bool:
        return bool(os.getenv(EXA_API_KEY_ENV))

    def create_tools(self) -> list[Callable[..., Awaitable[None]]]:
        # Freshness is a plain string rather than a ``Literal`` because Pipecat's
        # direct-function schema generation doesn't emit JSON-Schema enums for
        # ``Literal`` types (it falls back to an untyped value). A documented
        # string plus defensive mapping (_max_age_hours) is more reliable.
        async def web_search(
            params: FunctionCallParams,
            query: str,
            freshness: str = _DEFAULT_FRESHNESS,
            num_results: int = _DEFAULT_NUM_RESULTS,
            max_excerpt_chars: int = _DEFAULT_HIGHLIGHT_MAX_CHARS,
        ) -> None:
            """Search the web and return a few relevant sources with short excerpts.

            Use this to answer questions that need current or external
            information you don't already know.

            Args:
                query: A natural-language search query describing what to find.
                freshness: How fresh the results must be. Use "live" for
                    time-sensitive things that change constantly: weather, sports
                    scores, breaking news, stock prices, or anything happening
                    "now", "today", or "latest". Use "recent" (the default) for
                    general up-to-date information. Use "any" for stable facts
                    where speed matters more than recency.
                num_results: How many results to return. Keep it small (the
                    default) for a quick spoken answer; raise it only when the
                    task needs broader coverage.
                max_excerpt_chars: Maximum length of each result's excerpt. Keep
                    it small (the default) to stay fast and concise; raise it
                    when you need more detail from each source.
            """
            try:
                payload = await self._search(query, freshness, num_results, max_excerpt_chars)
            except Exception:
                logger.exception("web_search failed (query={!r}, freshness={!r})", query, freshness)
                await params.result_callback(
                    {"error": "The web search failed. Tell the user and offer to try again."}
                )
                return
            await params.result_callback(payload)

        async def web_fetch(
            params: FunctionCallParams,
            url: str,
            focus: str = "",
            freshness: str = _DEFAULT_FRESHNESS,
            max_chars: int = _DEFAULT_TEXT_MAX_CHARS,
        ) -> None:
            """Fetch the contents of a specific web page.

            Use this to read a page you already have the URL for (often one
            returned by web_search), e.g. to get details the search excerpts
            didn't cover.

            Args:
                url: The full URL of the page to read.
                focus: Optional. What you want from the page; when set, returns a
                    short summary targeted at it instead of the full text. Prefer
                    setting this to keep the response small and fast.
                freshness: How fresh the page must be. Use "live" to force a
                    fresh fetch (bypassing any cache), "recent" (the default)
                    otherwise, or "any" to allow older cached content for speed.
                max_chars: Maximum characters of page text to return. Ignored
                    when focus is set (a summary is returned instead). Keep it
                    small (the default) for speed; raise it for more detail.
            """
            try:
                payload = await self._fetch(url, focus, freshness, max_chars)
            except Exception:
                logger.exception("web_fetch failed (url={!r}, freshness={!r})", url, freshness)
                await params.result_callback(
                    {"error": "Could not fetch that page. Tell the user and offer an alternative."}
                )
                return
            await params.result_callback(payload)

        return [web_search, web_fetch]

    async def _search(
        self, query: str, freshness: str, num_results: int, max_excerpt_chars: int
    ) -> dict:
        resp = await asyncio.wait_for(
            self._exa.search(
                query,
                # "auto" lets Exa route between neural and keyword search --
                # favors result quality/reliability over the snappier "fast".
                type="auto",
                num_results=max(1, min(num_results, _MAX_NUM_RESULTS)),
                contents={
                    "highlights": {
                        "query": query,
                        "max_characters": max(1, max_excerpt_chars),
                    },
                    "max_age_hours": _max_age_hours(freshness),
                    "livecrawl_timeout": _LIVECRAWL_TIMEOUT_MS,
                },
            ),
            timeout=_CALL_TIMEOUT_S,
        )
        return {
            "results": [
                {
                    "title": r.title or "",
                    "url": r.url,
                    "published_date": r.published_date,
                    "excerpts": r.highlights or [],
                }
                for r in resp.results
            ]
        }

    async def _fetch(self, url: str, focus: str, freshness: str, max_chars: int) -> dict:
        kwargs: dict = {
            "max_age_hours": _max_age_hours(freshness),
            "livecrawl_timeout": _LIVECRAWL_TIMEOUT_MS,
        }
        if focus:
            # A focused summary is pre-condensed -- less for the model to read.
            kwargs["summary"] = {"query": focus}
        else:
            kwargs["text"] = {"max_characters": max(1, max_chars)}
        resp = await asyncio.wait_for(
            self._exa.get_contents([url], **kwargs), timeout=_CALL_TIMEOUT_S
        )
        results = resp.results or []
        if not results:
            return {"title": "", "url": url, "content": ""}
        r = results[0]
        content = r.summary if focus else r.text
        return {
            "title": r.title or "",
            "url": r.url or url,
            "content": (content or "").strip(),
        }
