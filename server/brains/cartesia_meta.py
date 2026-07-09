"""Cartesia + Meta cascade brain: Cartesia STT -> Meta LLM -> Cartesia TTS.

Same shape as ``cartesia_openai`` (Cartesia ``ink-2`` STT, ``sonic-3.5`` TTS, and
an OpenAI Responses LLM with a hosted ``web_search`` tool), but the LLM is served
by the Meta Model API (``muse-spark-1.1``) instead of OpenAI. Meta's Responses API
is OpenAI-compatible, and Pipecat has no dedicated Meta service, so the LLM is
built with ``OpenAIResponsesHttpLLMService`` pointed at Meta's base URL.

We use the *HTTP* Responses service, not the default ``OpenAIResponsesLLMService``:
that one streams over a persistent WebSocket whose URL (``wss://api.openai.com/v1/responses``)
is a separate arg from ``base_url`` and defaults to OpenAI, so a ``base_url`` override
alone is ignored -- it would hit OpenAI with the Meta key (a 401). Meta only exposes
the HTTP Responses endpoint (``{base_url}/responses``), which is what this variant
uses. The trade-off is losing the WebSocket's ``previous_response_id`` latency
optimization.

Web search uses Meta's hosted search grounding, which -- like OpenAI's -- is a
Responses-API feature: the model runs the search server-side (billed under
``META_API_KEY``, no separate key). The tool is built by the shared
``hosted_web_search_bundle`` helper, so this brain doesn't depend on any other.
The always-on generic tools (email/notion/x, end-session) still apply via
``setup_generic_tools`` too.
"""

from __future__ import annotations

from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from env import require_env
from tools.web import hosted_web_search_bundle

from .base import BrainName, BrainServices, BrainSpec, ToolBundle
from .overrides import get_brain_overrides

# Cartesia TTS requires an explicit voice ID (no service default).
CARTESIA_VOICE_ID = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"

# Meta Model API base URL (https://ai.developer.meta.com). Its HTTP Responses API
# is OpenAI-compatible and serves muse-spark-1.1, so ``OpenAIResponsesHttpLLMService``
# (and its hosted ``web_search`` tool) can talk to it via ``{base_url}/responses``.
META_BASE_URL = "https://api.meta.ai/v1"


def build(system_instruction: str) -> BrainServices:
    ov = get_brain_overrides().cartesia_meta

    stt = CartesiaSTTService(
        api_key=require_env("CARTESIA_API_KEY", "Set it to use the cartesia_meta brain."),
        settings=CartesiaSTTService.Settings(
            # Cartesia's latest streaming STT model. Pin to English so the model
            # never language-guesses on short or noisy input and starts emitting
            # non-English words -- mirrors the other cascade brains.
            model=ov.stt.model or "ink-2",
            language=Language.EN.value,
        ),
    )

    llm = OpenAIResponsesHttpLLMService(
        api_key=require_env("META_API_KEY", "Set it to use the cartesia_meta brain."),
        base_url=META_BASE_URL,
        settings=OpenAIResponsesHttpLLMService.Settings(
            model=ov.llm.model or "muse-spark-1.1",
            system_instruction=system_instruction,
        ),
    )

    tts = CartesiaTTSService(
        api_key=require_env("CARTESIA_API_KEY", "Set it to use the cartesia_meta brain."),
        settings=CartesiaTTSService.Settings(
            # Cartesia's latest TTS model.
            model=ov.tts.model or "sonic-3.5",
            voice=ov.tts.voice or CARTESIA_VOICE_ID,
        ),
        # Strip markup before synthesis so the voice never reads it aloud -- the
        # LLM can still emit markdown links, code, or tables. Same filter as the
        # other cascade brains.
        text_filters=[
            MarkdownTextFilter(
                params=MarkdownTextFilter.InputParams(
                    filter_code=True,
                    filter_tables=True,
                )
            ),
        ],
    )

    return BrainServices(llm=llm, stt=stt, tts=tts)


async def setup_tools() -> ToolBundle:
    """Attach Meta's hosted ``web_search`` tool (see ``hosted_web_search_bundle``)."""
    return hosted_web_search_bundle()


SPEC = BrainSpec(
    name=BrainName.CARTESIA_META,
    is_realtime=False,
    build=build,
    setup_tools=setup_tools,
)
