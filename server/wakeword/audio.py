"""Audio sources for the wake-word listener.

The listener consumes frames from an ``AudioSource`` -- anything that can yield
fixed-size chunks of 16 kHz mono int16 PCM. This indirection keeps the engine and
listener backend-agnostic: this package ships a ``PyAudioSource`` (the natural
choice here, since Pipecat's local transport already brings PyAudio + portaudio),
and any other backend (e.g. PvRecorder) can be dropped in by implementing the
same protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# openWakeWord expects 16 kHz mono int16 audio in 80 ms (1280-sample) chunks.
SAMPLE_RATE = 16000
FRAME_LENGTH = 1280


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
    """

    frame_length = FRAME_LENGTH

    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_length: int = FRAME_LENGTH,
    ) -> None:
        self._device_index = device_index
        self._sample_rate = sample_rate
        self.frame_length = frame_length
        self._pa = None
        self._stream = None

    def start(self) -> None:
        import pyaudio

        if self._stream is not None:
            return
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            frames_per_buffer=self.frame_length,
            input=True,
            input_device_index=self._device_index,
        )

    def read(self) -> list[int]:
        import numpy as np

        if self._stream is None:
            raise RuntimeError("PyAudioSource.read() called before start()")
        # exception_on_overflow=False: a transient input overflow shouldn't crash
        # an always-on listener; dropping a frame is fine for wake detection.
        data = self._stream.read(self.frame_length, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16).tolist()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
