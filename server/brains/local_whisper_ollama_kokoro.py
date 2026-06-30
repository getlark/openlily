"""Fully-local cascade brain: MLX Whisper STT -> Ollama LLM -> Kokoro TTS.

Everything runs on-device, with no external API and no API keys:

- STT: ``WhisperSTTServiceMLX`` (Apple Silicon MLX Whisper), default model
  ``mlx-community/whisper-large-v3-turbo``. On the first run the weights are
  pulled from Hugging Face and cached.
- LLM: ``OLLamaLLMService`` -- an OpenAI-compatible client pointed at a local
  ``ollama serve`` (default ``http://localhost:11434/v1``, overridable via
  ``OLLAMA_BASE_URL``). Default model ``gemma4:e4b``; pull it first with
  ``ollama pull gemma4:e4b``.
- TTS: ``KokoroTTSService``, default voice ``af_heart``. The model/voices file
  is downloaded and cached on first use.

No tools are attached (``setup_tools=None``): the existing web/browser/email
tools all reach out to cloud services, which would break the "fully local"
contract. Model names and the TTS voice can still be overridden per the usual
``brains.yaml`` mechanism (see ``brains/overrides.py``).
"""

from __future__ import annotations

import os

from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from .base import BrainName, BrainServices, BrainSpec
from .overrides import get_brain_overrides

# Kokoro requires an explicit voice (no service default). ``af_heart`` is a
# warm, natural English voice from the Kokoro v1.0 voice pack.
KOKORO_VOICE = "af_heart"

# Where the local Ollama server lives. The OpenAI-compatible endpoint is the
# ``/v1`` path. Overridable so a remote or non-default Ollama host can be used.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def build(system_instruction: str) -> BrainServices:
    ov = get_brain_overrides().local_whisper_ollama_kokoro

    stt = WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(
            # MLX Whisper large-v3-turbo: strong accuracy, fast on Apple
            # Silicon. Pin to English so the model never language-guesses on
            # short or noisy input and starts emitting non-English words --
            # mirrors the other cascade brains.
            model=ov.stt.model or MLXModel.LARGE_V3_TURBO.value,
            language=Language.EN,
        ),
    )

    llm = OLLamaLLMService(
        base_url=OLLAMA_BASE_URL,
        settings=OLLamaLLMService.Settings(
            # gemma4:e4b is the default Gemma 4 edge variant -- a good
            # quality/latency balance for local voice. Use gemma4:e2b for a
            # lighter/faster model on lower-memory machines.
            model=ov.llm.model or "gemma4:e4b",
            system_instruction=system_instruction,
        ),
    )

    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice=ov.tts.voice or KOKORO_VOICE,
            language=Language.EN,
        ),
        # Strip markup before synthesis so the voice never reads it aloud --
        # same filter as the other cascade brains.
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
    name=BrainName.LOCAL_WHISPER_OLLAMA_KOKORO,
    is_realtime=False,
    build=build,
    setup_tools=None,
)
