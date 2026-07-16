"""Audio sources for the wake-word listener.

The listener consumes frames from an ``AudioSource`` -- anything that can yield
fixed-size chunks of 16 kHz mono int16 PCM. This indirection keeps the engine and
listener backend-agnostic: this package ships a ``PyAudioSource`` (the natural
choice here, since Pipecat's local transport already brings PyAudio + portaudio),
and any other backend (e.g. PvRecorder) can be dropped in by implementing the
same protocol.
"""

from __future__ import annotations

import queue
from typing import Protocol, runtime_checkable

from loguru import logger

# openWakeWord expects 16 kHz mono int16 audio in 80 ms (1280-sample) chunks.
SAMPLE_RATE = 16000
FRAME_LENGTH = 1280

# How long ``read()`` waits for the next captured frame before declaring the
# stream stalled. In healthy operation a frame arrives every ~80 ms, so a couple
# of seconds of total silence from the device means the audio unit isn't actually
# running (typical after the machine sleeps/wakes - see AudioSourceStalled).
READ_TIMEOUT_SECS = 2.0
# Cap on buffered frames so a slow/blocked consumer can't grow memory without
# bound; when full the oldest frame is dropped (fine for wake detection, which
# only cares about recent audio).
MAX_BUFFERED_FRAMES = 50


class AudioSourceStalled(RuntimeError):
    """Raised by ``AudioSource.read()`` when no audio arrives in time.

    Signals that the underlying capture stream has gone silent (e.g. the OS put
    the audio device to sleep on a laptop-lid close and CoreAudio couldn't start
    the audio unit on wake). Callers should tear the source down and rebuild it
    rather than block forever on a dead stream.
    """


@runtime_checkable
class AudioSource(Protocol):
    """A source of fixed-size int16 PCM frames for wake-word detection."""

    frame_length: int

    def start(self) -> None:
        """Open the underlying device/stream and begin capturing."""
        ...

    def read(self) -> list[int]:
        """Block until one ``frame_length`` chunk is available; return it."""
        ...

    def stop(self) -> None:
        """Stop capturing and release the underlying device/stream."""
        ...


class PyAudioSource:
    """An ``AudioSource`` backed by PyAudio.

    Opens a 16 kHz mono int16 input stream and reads exact ``frame_length``
    chunks. ``start()``/``stop()`` fully open and close the stream so the mic is
    released between uses (e.g. when a voice session takes over the device).

    Capture is callback-driven: PortAudio delivers frames on its own thread into
    a bounded queue, and ``read()`` pops from that queue with a timeout. This is
    what lets a dead stream be detected - a blocking ``Stream.read()`` would hang
    forever if the audio unit silently failed to start (the macOS post-sleep
    ``-10863`` case) - so ``read()`` raises ``AudioSourceStalled`` instead of
    blocking, and the caller can rebuild the source.
    """

    frame_length = FRAME_LENGTH

    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_length: int = FRAME_LENGTH,
        read_timeout_secs: float = READ_TIMEOUT_SECS,
    ) -> None:
        self._device_index = device_index
        self._sample_rate = sample_rate
        self.frame_length = frame_length
        self._read_timeout_secs = read_timeout_secs
        self._pa = None
        self._stream = None
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=MAX_BUFFERED_FRAMES)

    def _on_audio(self, in_data, _frame_count, _time_info, _status):
        import pyaudio

        # Keep the queue near real time: if the consumer fell behind, drop the
        # oldest frame to make room rather than blocking PortAudio's thread.
        try:
            self._queue.put_nowait(in_data)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(in_data)
            except (queue.Empty, queue.Full):
                pass
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        import pyaudio

        if self._stream is not None:
            return
        # Release any handle left over from a failed prior attempt before
        # allocating a new one. start() is retried (e.g. while the device
        # recovers from sleep) and opening can fail *after* PyAudio() init - so
        # without this, a previous attempt's live, un-terminated PyAudio would be
        # orphaned here on each retry, leaking a PortAudio init every time (it
        # has no finalizer; only terminate() releases it).
        if self._pa is not None:
            self.stop()
        # Fresh queue per start so stale frames from a previous (e.g. stalled)
        # session can't leak into the new one.
        self._queue = queue.Queue(maxsize=MAX_BUFFERED_FRAMES)
        # Build into locals and only publish to self once both succeed, so a
        # failed open() can't leave a half-initialized source: terminate the init
        # we just created and re-raise, leaving self._pa/self._stream as None for
        # the caller to retry cleanly.
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._sample_rate,
                frames_per_buffer=self.frame_length,
                input=True,
                input_device_index=self._device_index,
                stream_callback=self._on_audio,
            )
        except Exception:
            try:
                pa.terminate()
            except Exception:
                logger.debug(
                    "Ignoring error while terminating PortAudio after a failed "
                    "open",
                    exc_info=True,
                )
            raise
        self._pa = pa
        self._stream = stream

    def read(self) -> list[int]:
        import numpy as np

        if self._stream is None:
            raise RuntimeError("PyAudioSource.read() called before start()")
        try:
            data = self._queue.get(timeout=self._read_timeout_secs)
        except queue.Empty as e:
            raise AudioSourceStalled(
                f"No audio for {self._read_timeout_secs}s; capture stream appears dead"
            ) from e
        return np.frombuffer(data, dtype=np.int16).tolist()

    def stop(self) -> None:
        # Clear the handles up front (capturing locals) so the source is left in
        # a clean, reopenable state even if releasing a handle throws - which it
        # can when the OS has yanked the device out from under PortAudio (the
        # post-sleep case). Otherwise a half-failed stop would leave self._stream
        # set, making the next start() a no-op (a permanent wedge), or orphan an
        # un-terminated PyAudio (PyAudio has no finalizer, so that leaks the
        # PortAudio init). Each handle is still released best-effort below.
        stream, self._stream = self._stream, None
        pa, self._pa = self._pa, None
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                logger.debug(
                    "Ignoring error while closing wake-word capture stream",
                    exc_info=True,
                )
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                logger.debug(
                    "Ignoring error while terminating PortAudio", exc_info=True
                )
