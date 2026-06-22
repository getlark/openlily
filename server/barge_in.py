"""Wake-word barge-in for the local voice transport.

While the bot is speaking the mic is half-duplex gated (see ``transport_local``):
captured audio is dropped from the pipeline so the bot can't hear itself. That
removes barge-in - the user can't interrupt mid-utterance. This module restores
it for a single deliberate signal: a wake word. The gated (but AEC-cleaned) mic
audio is tapped into an openWakeWord detector running on a background thread; on
a hit, a callback fires that interrupts the bot.

A wake word is the right trigger here because it is rare and specific - the bot's
own TTS won't say it, and it's scored on already-echo-cancelled audio - so it
survives the gate without reintroducing the self-interruption problem.

This is the bridge between the Pipecat-agnostic ``wakeword`` package and the
transport; it deliberately does not live inside ``wakeword`` (which stays free of
any Pipecat/transport coupling).
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
import time
from typing import Callable

import numpy as np
from loguru import logger

from wakeword import WakeWordEngine

# openWakeWord expects 16 kHz mono int16 audio in 80 ms (1280-sample) chunks.
FRAME_LENGTH = 1280
# If gated audio hasn't been fed for this long (e.g. between bot turns), reset
# the engine's rolling buffer so stale, discontinuous audio can't false-fire.
GAP_RESET_SECS = 0.5
# openWakeWord detection threshold (matches WakeWordEngine's default). Raise it
# if the bot's own speech ever self-triggers a barge-in.
BARGE_IN_THRESHOLD = 0.5
# Bound the hand-off queue so a slow/loading detector can't accumulate unbounded
# audio; dropping frames is fine since detection is best-effort.
_MAX_QUEUED_CHUNKS = 64


def resolve_wake_models() -> list[str]:
    """Resolve the openWakeWord model(s) from $WAKE_MODELS, else the default.

    Mirrors ``bot._wake_models`` so barge-in uses the same wake word as the
    wake-gated launcher without a new env var.
    """
    raw = os.getenv("WAKE_MODELS")
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
        if models:
            return models
    return ["alexa"]


class BargeInDetector:
    """Runs openWakeWord on tapped mic audio and fires ``on_detect`` on a hit.

    Audio is handed in from the PyAudio callback thread via ``feed`` and consumed
    on a dedicated worker thread, so the (CPU-bound) wake-word inference never
    runs in the realtime audio callback. The engine is constructed inside the
    worker thread so its model load doesn't block pipeline startup.
    """

    def __init__(
        self,
        on_detect: Callable[[], None],
        models: list[str] | None = None,
        threshold: float = BARGE_IN_THRESHOLD,
    ) -> None:
        self._on_detect = on_detect
        self._models = models or resolve_wake_models()
        self._threshold = threshold
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=_MAX_QUEUED_CHUNKS)
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="barge-in-wakeword", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        # Unblock the worker's blocking get() so it can exit promptly.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def feed(self, audio: bytes) -> None:
        """Hand a mic buffer (16 kHz mono int16) to the detector; best-effort."""
        try:
            self._queue.put_nowait(audio)
        except queue.Full:
            pass  # detector is loading or behind; dropping is fine

    def _run(self) -> None:
        try:
            engine = WakeWordEngine(models=self._models, threshold=self._threshold)
        except Exception:
            logger.exception("Barge-in wake-word engine failed to load; disabled")
            return
        logger.info(f"Barge-in wake-word detector ready (models={self._models})")

        buf = np.empty(0, dtype=np.int16)
        last_fed = time.monotonic()
        while self._running:
            audio = self._queue.get()
            if audio is None:
                break
            now = time.monotonic()
            if now - last_fed > GAP_RESET_SECS:
                # Gap since the last gated window: drop stale context.
                engine.reset()
                buf = np.empty(0, dtype=np.int16)
            last_fed = now

            buf = np.concatenate([buf, np.frombuffer(audio, dtype=np.int16)])
            while len(buf) >= FRAME_LENGTH:
                frame = buf[:FRAME_LENGTH]
                buf = buf[FRAME_LENGTH:]
                label = engine.process(frame.tolist())
                if label is not None:
                    logger.info(f"Barge-in wake word '{label}' detected; interrupting bot")
                    engine.reset()
                    buf = np.empty(0, dtype=np.int16)
                    try:
                        self._on_detect()
                    except Exception:
                        logger.exception("Barge-in on_detect callback failed")
                    break
