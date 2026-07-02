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

import httpx
from loguru import logger
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from .base import BrainName, BrainServices, BrainSpec
from .overrides import _Cascade, get_brain_overrides

# Kokoro requires an explicit voice (no service default). ``af_heart`` is a
# warm, natural English voice from the Kokoro v1.0 voice pack.
KOKORO_VOICE = "af_heart"

# Default Ollama model: the Gemma 4 edge variant -- a good quality/latency
# balance for local voice. Use gemma4:e2b for a lighter/faster model.
DEFAULT_LLM_MODEL = "gemma4:e4b"

# Where the local Ollama server lives. The OpenAI-compatible endpoint is the
# ``/v1`` path. Overridable so a remote or non-default Ollama host can be used.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def _build_stt(ov: _Cascade) -> WhisperSTTServiceMLX:
    return WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(
            # MLX Whisper large-v3-turbo: strong accuracy, fast on Apple
            # Silicon. Pin to English so the model never language-guesses on
            # short or noisy input and starts emitting non-English words --
            # mirrors the other cascade brains.
            model=ov.stt.model or MLXModel.LARGE_V3_TURBO.value,
            language=Language.EN,
        ),
    )


def _build_tts(ov: _Cascade) -> KokoroTTSService:
    return KokoroTTSService(
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


def build(system_instruction: str) -> BrainServices:
    ov = get_brain_overrides().local_whisper_ollama_kokoro

    stt = _build_stt(ov)

    llm = OLLamaLLMService(
        base_url=OLLAMA_BASE_URL,
        settings=OLLamaLLMService.Settings(
            model=ov.llm.model or DEFAULT_LLM_MODEL,
            system_instruction=system_instruction,
        ),
    )

    tts = _build_tts(ov)

    return BrainServices(llm=llm, stt=stt, tts=tts)


async def _warmup_ollama(model: str) -> None:
    """Preload the Ollama model into memory, failing loudly if it can't work.

    Hitting Ollama's native ``/api/generate`` with an empty prompt loads the
    model into memory, so the first reply of the first session isn't a cold
    start. We deliberately don't pin it (no ``keep_alive: -1``): Ollama unloads
    it after its default idle timeout (~5 min), so it doesn't hold memory while
    the assistant sits idle. A later session that starts cold just pays a
    one-time reload on its first reply.

    Raises a clear, actionable ``RuntimeError`` on the two known-broken cases
    (server not running, model not pulled) so the user can fix it and relaunch
    rather than hit a confusing failure once the LLM is first invoked.
    """
    host = OLLAMA_BASE_URL.removesuffix("/v1")
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{host}/api/generate", json={"model": model})
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Ollama is not reachable at {host}. Start it (`ollama serve`) and "
            f"pull the model (`ollama pull {model}`), then relaunch. Set "
            f"OLLAMA_BASE_URL to use a different host."
        ) from e

    if resp.status_code == 404:
        # Server is up but the model isn't downloaded.
        raise RuntimeError(
            f"Ollama model {model!r} is not available. Run `ollama pull {model}` "
            f"and relaunch."
        )
    resp.raise_for_status()


async def warmup() -> None:
    """Download and load all three local models up front (see ``BrainSpec.warmup``).

    Fail-fast: any error (Ollama down/missing model, or a failed Whisper/Kokoro
    download) propagates and aborts startup with a clear message.
    """
    ov = get_brain_overrides().local_whisper_ollama_kokoro

    # STT: a dummy transcription on a short silence buffer forces the MLX
    # Whisper weights to download and load into the process-wide model cache
    # (mlx_whisper's ModelHolder), so the first real utterance doesn't stall.
    logger.info("Warming up speech-to-text (MLX Whisper)...")
    stt = _build_stt(ov)
    silence = b"\x00\x00" * 8000  # ~0.5s of 16 kHz, 16-bit mono PCM
    async for _ in stt.run_stt(silence):
        pass

    # TTS: constructing the service downloads the Kokoro model/voices and loads
    # the ONNX model. Discarded here; the per-session build reloads it from the
    # now-cached local files (no network).
    logger.info("Warming up text-to-speech (Kokoro)...")
    _build_tts(ov)

    # LLM: preload the Ollama model into memory (and fail loudly if it can't work).
    model = ov.llm.model or DEFAULT_LLM_MODEL
    logger.info(f"Warming up LLM (ollama: {model})...")
    await _warmup_ollama(model)


SPEC = BrainSpec(
    name=BrainName.LOCAL_WHISPER_OLLAMA_KOKORO,
    is_realtime=False,
    build=build,
    setup_tools=None,
    warmup=warmup,
)
