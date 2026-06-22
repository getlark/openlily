"""OpenAI Realtime (GPT speech-to-speech) brain.

The realtime model handles STT + LLM + TTS internally, so the pipeline omits the
separate STT/TTS stages (see ``BrainSpec.is_realtime`` and the pipeline in
``bot.py``).
"""

from __future__ import annotations

from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    InputAudioNoiseReduction,
    InputAudioTranscription,
    SessionProperties,
    TurnDetection,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from env import require_env
from tools.web import setup_web_tools

from .base import BrainName, BrainServices, BrainSpec, ToolBundle
from .overrides import get_brain_overrides


def build(system_instruction: str) -> BrainServices:
    ov = get_brain_overrides().openai_realtime

    session_properties = SessionProperties(
        audio=AudioConfiguration(
            input=AudioInput(
                # Pin transcription to English and a concrete model so the
                # model never language-guesses on short or noisy input and
                # starts emitting non-English words.
                transcription=InputAudioTranscription(
                    model=ov.stt.model or "gpt-4o-transcribe",
                    language="en",
                ),
                # The expected noise is *other people talking*. server_vad
                # gates on loudness, so the closer/louder primary speaker wins
                # and quieter background chatter is rejected -- a raised
                # threshold trades a little snappiness for that robustness.
                # Nudge threshold toward 0.65-0.7 if background voices still
                # trigger turns.
                turn_detection=TurnDetection(
                    type="server_vad",
                    threshold=0.6,
                    prefix_padding_ms=300,
                    silence_duration_ms=600,
                ),
                # Keep near_field even for mixed mic setups: far_field would
                # make the model *more* sensitive to distant voices, the
                # opposite of what we want with background chatter.
                noise_reduction=InputAudioNoiseReduction(type="near_field"),
            )
        ),
    )

    llm = OpenAIRealtimeLLMService(
        api_key=require_env("OPENAI_API_KEY", "Set it to use the openai_realtime brain."),
        settings=OpenAIRealtimeLLMService.Settings(
            session_properties=session_properties,
            system_instruction=system_instruction,
            model=ov.llm.model or "gpt-realtime-2",
        ),
    )

    return BrainServices(llm=llm)


async def setup_tools() -> ToolBundle:
    """Build the agent's web search/fetch tools for the session.

    Skipped (an empty bundle) when the web provider isn't configured, so the
    session still runs without web search.
    """
    return setup_web_tools()


SPEC = BrainSpec(
    name=BrainName.OPENAI_REALTIME,
    is_realtime=True,
    build=build,
    setup_tools=setup_tools,
)
