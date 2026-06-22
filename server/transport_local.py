"""Local-audio transport for the terminal voice CLI.

Pipecat's ``LocalAudioTransport`` reads the system mic and plays to the system
speakers via PyAudio, so the bot process itself is the voice client - no network,
no separate client app.

On a single device the bot would otherwise hear its own voice from the speakers
(self-transcription, false interruptions). Two always-on layers close that gap:

* **Half-duplex gating** (``GatedLocalAudioTransport``): while the bot is speaking
  (any ``TTSAudioRawFrame`` is written, plus a short tail), captured mic audio is
  dropped before it enters the pipeline, so neither STT nor VAD ever sees the
  echo. This is the primary defense and is pure Python. The trade-off is no
  barge-in while the bot talks; a ``_on_gated_audio`` hook is left as the seam for
  a future wake-word barge-in.
* **The WebRTC Audio Processing Module** (``_APM``): the same AEC + noise
  suppression + auto gain control that browsers and LiveKit use, run over the mic
  (near-end) and the bot's TTS playback (far-end reference). It cleans the user's
  near-end audio and covers the gate's reopen boundary. We use
  ``livekit.rtc.AudioProcessingModule`` purely as a standalone DSP module: no
  LiveKit room, server, API key, or network is involved.

Both are mandatory: ``livekit`` is a declared dependency and the transport fails
loudly (rather than degrading quality) if it is missing.

On top of the gate, **wake-word barge-in** (``WAKE_WORD_BARGE_IN``, on by
default) restores the ability to interrupt: while the bot speaks, the gated (but
AEC-cleaned) mic audio is tapped into an openWakeWord detector via the
``_on_gated_audio`` hook, and a wake word ("alexa", or ``$WAKE_MODELS``) pushes a
pipeline interruption that stops the bot. Set it False for pure half-duplex.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import pyaudio
from loguru import logger
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionWorkerFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransport,
    LocalAudioTransportParams,
)

# The bot's mic capture rate. 16 kHz is what OpenAI STT expects and is plenty
# for speech; the APM processes capture at this rate.
LOCAL_AUDIO_IN_SAMPLE_RATE = 16000
# OpenAI TTS and the OpenAI Realtime model both emit 24 kHz audio; matching it
# here avoids an extra resample on the way to the speakers.
LOCAL_AUDIO_OUT_SAMPLE_RATE = 24000

# Keep the mic gated this long after the last bot audio frame, so the playback
# buffer drains and brief room echo decays before we listen again. Too small
# lets trailing echo leak; too large clips the start of the user's next turn.
GATE_TAIL_SECS = 0.3
# AEC far/near delay hint (ms) handed to the APM's set_stream_delay_ms.
APM_STREAM_DELAY_MS = 100
# Wake-word barge-in: while the bot speaks (mic gated), a wake word ("alexa", or
# $WAKE_MODELS) can still interrupt it. Set False for pure half-duplex - the bot
# can't be interrupted mid-utterance and openWakeWord is never even imported.
WAKE_WORD_BARGE_IN = False
# On a barge-in, force the mic gate open for this long, ignoring bot audio. This
# has to outlast the trailing TTS flush that the interruption drains (each
# flushed frame would otherwise call mark_speaking and re-close the gate),
# leaving the mic open for the command spoken right after the wake word. The
# interrupted bot stays silent through this window (it only speaks again after
# the user's command -> STT -> LLM -> TTS round trip), so there's no echo to gate
# during it; afterwards normal gating resumes and the bot's next reply re-gates.
BARGE_IN_OPEN_SECS = 2.0


class _APM:
    """Thread-safe wrapper around the WebRTC Audio Processing Module.

    Feeds the mic (near-end) through ``process_stream`` and the bot's playback
    (far-end) through ``process_reverse_stream``, both as exact 10 ms frames, so
    the echo canceller can subtract the bot's own voice from the mic input.
    """

    def __init__(
        self,
        *,
        echo_cancellation: bool = True,
        noise_suppression: bool = True,
        high_pass_filter: bool = True,
        auto_gain_control: bool = True,
        stream_delay_ms: int = APM_STREAM_DELAY_MS,
    ) -> None:
        try:
            from livekit import rtc
        except ImportError as e:
            raise RuntimeError(
                "The local voice CLI requires the 'livekit' package for echo "
                "cancellation and noise suppression (declared in "
                "server/pyproject.toml). Run `uv sync` before starting the bot."
            ) from e

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=echo_cancellation,
            noise_suppression=noise_suppression,
            high_pass_filter=high_pass_filter,
            auto_gain_control=auto_gain_control,
        )
        self._aec = echo_cancellation
        self._stream_delay_ms = stream_delay_ms
        self._delay_set = False
        self._lock = threading.Lock()
        self._reverse_buf = bytearray()
        self._warned_capture = False
        self._warned_render = False

    def _ensure_delay_locked(self) -> None:
        # set_stream_delay_ms must be called iff echo processing is enabled.
        if self._aec and not self._delay_set:
            try:
                self._apm.set_stream_delay_ms(self._stream_delay_ms)
            except Exception as e:  # noqa: BLE001 - degrade gracefully
                logger.warning(f"APM set_stream_delay_ms failed: {e}")
            self._delay_set = True

    def process_capture(
        self, audio: bytes, sample_rate: int, num_channels: int
    ) -> bytes:
        """Run AEC/NS/AGC over a mic buffer; returns the cleaned audio."""
        bytes_per_10ms = (sample_rate // 100) * num_channels * 2
        if bytes_per_10ms <= 0 or len(audio) % bytes_per_10ms != 0:
            return audio  # can't frame cleanly; pass through untouched
        samples_per_channel = sample_rate // 100
        out = bytearray()
        with self._lock:
            self._ensure_delay_locked()
            for i in range(0, len(audio), bytes_per_10ms):
                chunk = bytearray(audio[i : i + bytes_per_10ms])
                frame = self._rtc.AudioFrame(
                    chunk, sample_rate, num_channels, samples_per_channel
                )
                try:
                    self._apm.process_stream(frame)
                    out += bytes(frame.data)
                except Exception as e:  # noqa: BLE001 - degrade gracefully
                    if not self._warned_capture:
                        logger.warning(
                            f"APM process_stream failed; passing through: {e}"
                        )
                        self._warned_capture = True
                    out += chunk
        return bytes(out)

    def process_render(self, audio: bytes, sample_rate: int, num_channels: int) -> None:
        """Feed played audio to the APM as the echo-cancellation reference."""
        if not self._aec:
            return
        bytes_per_10ms = (sample_rate // 100) * num_channels * 2
        if bytes_per_10ms <= 0:
            return
        samples_per_channel = sample_rate // 100
        with self._lock:
            self._ensure_delay_locked()
            self._reverse_buf += audio
            while len(self._reverse_buf) >= bytes_per_10ms:
                chunk = bytearray(self._reverse_buf[:bytes_per_10ms])
                del self._reverse_buf[:bytes_per_10ms]
                frame = self._rtc.AudioFrame(
                    chunk, sample_rate, num_channels, samples_per_channel
                )
                try:
                    self._apm.process_reverse_stream(frame)
                except Exception as e:  # noqa: BLE001 - degrade gracefully
                    if not self._warned_render:
                        logger.warning(f"APM process_reverse_stream failed: {e}")
                        self._warned_render = True


class _SpeakerGate:
    """Tracks whether the bot is currently speaking, for half-duplex gating.

    ``mark_speaking`` is called from the output side whenever bot audio is
    written; ``is_gated`` is polled from the PyAudio input callback (a different
    thread), hence the lock. The gate stays closed for ``tail_secs`` after the
    last bot audio so the playback buffer drains before we listen again.
    """

    def __init__(self, tail_secs: float = GATE_TAIL_SECS) -> None:
        self._tail = tail_secs
        self._last_speaking = 0.0
        # Absolute monotonic deadline until which the gate is forced open and bot
        # audio is ignored (set by a barge-in). 0.0 means no active window.
        self._open_until = 0.0
        self._lock = threading.Lock()

    def mark_speaking(self) -> None:
        with self._lock:
            # Inside a barge-in's forced-open window, ignore bot audio so the
            # interruption's trailing TTS flush can't re-close the gate and clip
            # the user's command.
            if time.monotonic() < self._open_until:
                return
            self._last_speaking = time.monotonic()

    def open_window(self, secs: float) -> None:
        """Force the gate open for ``secs``, ignoring bot audio (e.g. on barge-in).

        Clears the speaking mark and holds the gate open until the window
        elapses; after that, normal tail-based gating resumes (so the bot's next
        reply re-closes it).
        """
        with self._lock:
            self._open_until = time.monotonic() + secs
            self._last_speaking = 0.0

    def is_gated(self) -> bool:
        with self._lock:
            if time.monotonic() < self._open_until:
                return False
            return (time.monotonic() - self._last_speaking) < self._tail


class _GatedInputTransport(LocalAudioInputTransport):
    """Local mic input that drops captured audio while the bot is speaking.

    ``_preprocess_capture`` and ``_on_gated_audio`` are hooks: the APM subclass
    overrides the former to AEC-clean the mic, and the latter is the seam for a
    future wake-word barge-in (mic audio is still available there while gated).
    """

    def __init__(self, py_audio, params, gate: _SpeakerGate):
        super().__init__(py_audio, params)
        self._gate = gate
        self._was_gated = False

    def _preprocess_capture(self, in_data: bytes) -> bytes:
        return in_data

    def _on_gated_audio(self, audio: bytes) -> None:
        # Reserved for a future wake-word barge-in tap. While gated, the audio is
        # dropped from the pipeline but still surfaced here.
        pass

    def _audio_in_callback(self, in_data, frame_count, time_info, status):
        audio = self._preprocess_capture(in_data)
        gated = self._gate.is_gated()
        if gated != self._was_gated:
            logger.debug(
                f"mic gate {'CLOSED (bot speaking)' if gated else 'OPEN (listening)'}"
            )
            self._was_gated = gated
        if gated:
            self._on_gated_audio(audio)
            return (None, pyaudio.paContinue)
        frame = InputAudioRawFrame(
            audio=audio,
            sample_rate=self._sample_rate,
            num_channels=self._params.audio_in_channels,
        )
        asyncio.run_coroutine_threadsafe(
            self.push_audio_frame(frame), self.get_event_loop()
        )
        return (None, pyaudio.paContinue)


class _GatedOutputTransport(LocalAudioOutputTransport):
    """Local speaker output that marks the gate while the bot is speaking.

    Only ``TTSAudioRawFrame`` (the bot's spoken audio, for both the cascade and
    realtime brains) closes the gate - the plain ``OutputAudioRawFrame`` readiness
    chime does not, so it won't gate the user's first turn.
    """

    def __init__(self, py_audio, params, gate: _SpeakerGate):
        super().__init__(py_audio, params)
        self._gate = gate

    async def write_audio_frame(self, frame) -> bool:
        if isinstance(frame, TTSAudioRawFrame):
            self._gate.mark_speaking()
        return await super().write_audio_frame(frame)


class _APMInputTransport(_GatedInputTransport):
    def __init__(self, py_audio, params, gate: _SpeakerGate, apm: _APM):
        super().__init__(py_audio, params, gate)
        self._apm = apm

    def _preprocess_capture(self, in_data: bytes) -> bytes:
        return self._apm.process_capture(
            in_data, self._sample_rate, self._params.audio_in_channels
        )


class _APMOutputTransport(_GatedOutputTransport):
    def __init__(self, py_audio, params, gate: _SpeakerGate, apm: _APM):
        super().__init__(py_audio, params, gate)
        self._apm = apm

    async def write_audio_frame(self, frame) -> bool:
        # Feed a copy of the played audio to the APM as the far-end reference;
        # the gated base then marks the speaker gate and plays the original audio.
        self._apm.process_render(
            frame.audio, self._sample_rate, self._params.audio_out_channels
        )
        return await super().write_audio_frame(frame)


class _BargeInInputTransport(_APMInputTransport):
    """APM mic input that lets a wake word interrupt the bot while gated.

    While the bot speaks the mic is gated (audio dropped from the pipeline), but
    the AEC-cleaned capture is still tapped here into a wake-word detector. On a
    hit we push an ``InterruptionWorkerFrame`` upstream; the worker turns it into
    a pipeline-wide interruption that stops the bot. The bot then stops emitting
    TTS, so the gate reopens on its own and the user's follow-up reaches STT.
    """

    def __init__(self, py_audio, params, gate: _SpeakerGate, apm: _APM):
        super().__init__(py_audio, params, gate, apm)
        self._detector = None

    async def start(self, frame):
        await super().start(frame)
        if self._detector is None:
            # Imported lazily so a pure half-duplex run never loads openWakeWord.
            from barge_in import BargeInDetector

            self._detector = BargeInDetector(on_detect=self._trigger_barge_in)
            self._detector.start()

    async def cleanup(self):
        if self._detector is not None:
            self._detector.stop()
            self._detector = None
        await super().cleanup()

    def _on_gated_audio(self, audio: bytes) -> None:
        if self._detector is not None:
            self._detector.feed(audio)

    def _trigger_barge_in(self) -> None:
        # Called from the detector's worker thread. Force the gate open for a
        # window (not just a one-shot reset) so the command spoken right after
        # the wake word is heard, instead of being dropped: a one-shot open is
        # immediately undone by the interruption's trailing TTS flush, which
        # re-marks the gate and re-closes it over the start of the command. The
        # window ignores that flush. Then request a pipeline interruption to stop
        # the bot (hop back to the event loop).
        self._gate.open_window(BARGE_IN_OPEN_SECS)
        asyncio.run_coroutine_threadsafe(
            self.push_frame(InterruptionWorkerFrame(), FrameDirection.UPSTREAM),
            self.get_event_loop(),
        )


class GatedLocalAudioTransport(LocalAudioTransport):
    """LocalAudioTransport with half-duplex mic gating and a shared WebRTC APM.

    ``apm`` may be ``None`` as a code-level seam (gating without the APM); the
    builder always provides one. ``barge_in`` adds wake-word barge-in on top of
    the APM input; set it False (via ``WAKE_WORD_BARGE_IN``) for pure half-duplex.
    """

    def __init__(
        self,
        params: LocalAudioTransportParams,
        gate: _SpeakerGate,
        apm: _APM | None = None,
        barge_in: bool = False,
    ):
        super().__init__(params)
        self._gate = gate
        self._apm = apm
        self._barge_in = barge_in

    def input(self):
        if not self._input:
            if self._apm is not None and self._barge_in:
                self._input = _BargeInInputTransport(
                    self._pyaudio, self._params, self._gate, self._apm
                )
            elif self._apm is not None:
                self._input = _APMInputTransport(
                    self._pyaudio, self._params, self._gate, self._apm
                )
            else:
                self._input = _GatedInputTransport(
                    self._pyaudio, self._params, self._gate
                )
        return self._input

    def output(self):
        if not self._output:
            if self._apm is not None:
                self._output = _APMOutputTransport(
                    self._pyaudio, self._params, self._gate, self._apm
                )
            else:
                self._output = _GatedOutputTransport(
                    self._pyaudio, self._params, self._gate
                )
        return self._output


def build_local_transport() -> LocalAudioTransport:
    """Build the local-audio transport for the terminal voice CLI.

    Always uses half-duplex gating plus the WebRTC APM (AEC + NS + AGC). Raises
    loudly if ``livekit`` (required for the APM) is unavailable, rather than
    silently running at degraded quality.
    """
    params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=LOCAL_AUDIO_IN_SAMPLE_RATE,
        audio_out_sample_rate=LOCAL_AUDIO_OUT_SAMPLE_RATE,
    )

    apm = _APM(stream_delay_ms=APM_STREAM_DELAY_MS)  # raises loudly if livekit missing
    gate = _SpeakerGate()
    logger.info(
        "Local audio: half-duplex gating + WebRTC APM (AEC + noise suppression + AGC)"
        + ("; wake-word barge-in enabled" if WAKE_WORD_BARGE_IN else "")
    )
    return GatedLocalAudioTransport(params, gate, apm, barge_in=WAKE_WORD_BARGE_IN)


def close_local_transport(transport: LocalAudioTransport) -> None:
    """Release the transport's PyAudio instance.

    Pipecat's ``LocalAudioTransport`` creates a PyAudio handle in ``__init__`` but
    never terminates it (only the input/output stream processors are closed on
    session end). When a long-lived process builds a fresh transport per session
    (the wake-gated run mode), that would leak a PortAudio initialization each
    time, so terminate it here once the session's streams are already closed.
    """
    with contextlib.suppress(Exception):
        transport._pyaudio.terminate()
