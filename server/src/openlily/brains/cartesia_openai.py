"""Cartesia + OpenAI cascade brain: Cartesia STT -> OpenAI LLM -> Cartesia TTS.

Same shape as ``openai_standard`` -- and the same OpenAI Responses LLM
(``gpt-5.4-mini`` with OpenAI's hosted ``web_search`` tool) -- but speech in/out
runs on Cartesia's latest models (``ink-2`` STT, ``sonic-3.5`` TTS). The hosted
web-search tool comes from the shared ``hosted_web_search_bundle`` helper.
"""

from __future__ import annotations

from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from openlily.env import require_env
from openlily.tools.contracts import ToolId

from .base import BrainName, BrainServices, BrainSpec
from .overrides import get_brain_overrides

# Cartesia TTS requires an explicit voice ID (no service default).
CARTESIA_VOICE_ID = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"


def build(system_instruction: str) -> BrainServices:
    ov = get_brain_overrides().cartesia_openai

    stt = CartesiaSTTService(
        api_key=require_env("CARTESIA_API_KEY", "Set it to use the cartesia_openai brain."),
        settings=CartesiaSTTService.Settings(
            # Cartesia's latest streaming STT model. Pin to English so the model
            # never language-guesses on short or noisy input and starts emitting
            # non-English words -- mirrors the openai_standard brain.
            model=ov.stt.model or "ink-2",
            language=Language.EN.value,
        ),
    )

    llm = OpenAIResponsesLLMService(
        api_key=require_env("OPENAI_API_KEY", "Set it to use the cartesia_openai brain."),
        settings=OpenAIResponsesLLMService.Settings(
            model=ov.llm.model or "gpt-5.4-mini",
            system_instruction=system_instruction,
        ),
    )

    tts = CartesiaTTSService(
        api_key=require_env("CARTESIA_API_KEY", "Set it to use the cartesia_openai brain."),
        settings=CartesiaTTSService.Settings(
            # Cartesia's latest TTS model.
            model=ov.tts.model or "sonic-3.5",
            voice=ov.tts.voice or CARTESIA_VOICE_ID,
        ),
        # Strip markup before synthesis so the voice never reads it aloud -- the
        # LLM can still emit markdown links, code, or tables. Same filter as the
        # openai_standard TTS.
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


SPEC = BrainSpec(
    name=BrainName.CARTESIA_OPENAI,
    is_realtime=False,
    build=build,
    tools=(ToolId.WEB_HOSTED,),
)
