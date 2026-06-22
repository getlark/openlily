"""All-OpenAI cascade brain: OpenAI STT -> OpenAI LLM -> OpenAI TTS.

The LLM is the OpenAI Responses API service so the model can use OpenAI's
hosted ``web_search`` tool (OpenAI runs the search server-side; there is no
local function callback). See ``setup_tools``.
"""

from __future__ import annotations

from pipecat.adapters.schemas.tools_schema import AdapterType
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from env import require_env
from tools.web import WEB_SEARCH_INSTRUCTION

from .base import BrainName, BrainServices, BrainSpec, ToolBundle
from .overrides import get_brain_overrides


def build(system_instruction: str) -> BrainServices:
    ov = get_brain_overrides().openai_standard

    stt = OpenAISTTService(
        api_key=require_env("OPENAI_API_KEY", "Set it to use the openai_standard brain."),
        settings=OpenAISTTService.Settings(
            model=ov.stt.model or "gpt-4o-transcribe",
            # Pin to English (the default, but explicit here) so the model never
            # language-guesses on short or noisy input and starts emitting
            # non-English words -- mirrors the realtime brain's transcription
            # config. temperature=0 keeps transcripts deterministic.
            language=Language.EN,
            temperature=0.0,
            # Bias transcription toward natural, English conversational speech.
            # The prompt anchors the model's expected style/vocabulary, which
            # reduces drift into non-English or hallucinated text on short or
            # noisy input (it doesn't fully prevent silence hallucinations --
            # VAD tuning upstream handles those).
            prompt=(
                "The following is a clear English conversation between a person "
                "and their personal voice assistant. The person speaks naturally "
                "in everyday English."
            ),
        ),
    )

    tts = OpenAITTSService(
        api_key=require_env("OPENAI_API_KEY", "Set it to use the openai_standard brain."),
        settings=OpenAITTSService.Settings(
            model=ov.tts.model or "gpt-4o-mini-tts",
            voice=ov.tts.voice or "marin",
            # Steer delivery for a voice assistant: natural, conversational, and
            # not rushed. gpt-4o-mini-tts honors free-form acting instructions.
            instructions=(
                "Speak in a warm, natural, conversational tone, as a friendly "
                "personal assistant. Use a calm, even pace and clear "
                "enunciation. Do not sound robotic."
            ),
        ),
        # Strip markup before synthesis so the voice never reads it aloud. The LLM
        # can still emit markdown links (e.g. "[docs.pipecat.ai](https://...)"),
        # code, or tables; this filter converts links to just their text, drops
        # the URL, and removes code blocks/tables. Filters run after sentence
        # aggregation, so each link/block reaches the filter intact.
        text_filters=[
            MarkdownTextFilter(
                params=MarkdownTextFilter.InputParams(
                    filter_code=True,
                    filter_tables=True,
                )
            ),
        ],
    )

    llm = OpenAIResponsesLLMService(
        api_key=require_env("OPENAI_API_KEY", "Set it to use the openai_standard brain."),
        settings=OpenAIResponsesLLMService.Settings(
            model=ov.llm.model or "gpt-5.4-mini",
            system_instruction=system_instruction,
        ),
    )

    return BrainServices(llm=llm, stt=stt, tts=tts)


async def setup_tools() -> ToolBundle:
    """Attach OpenAI's hosted web search tool to the LLM context.

    ``web_search`` is a server-side tool of the Responses API: the model runs
    the search itself and reads the results, so there's no local handler to
    register or clean up. ``search_context_size`` is kept low to keep voice
    turns fast and concise; raise it if answers need broader coverage.
    """
    web_search = {"type": "web_search", "search_context_size": "low"}
    return ToolBundle(
        custom_tools={AdapterType.OPENAI: [web_search]},
        instructions=[WEB_SEARCH_INSTRUCTION],
    )


SPEC = BrainSpec(
    name=BrainName.OPENAI_STANDARD,
    is_realtime=False,
    build=build,
    setup_tools=setup_tools,
)
