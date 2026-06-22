"""Portable wake-word detection.

A small, Pipecat-agnostic package (depends only on ``openwakeword``, ``numpy``,
and ``pyaudio``): a pure detection ``WakeWordEngine``, a pluggable ``AudioSource``
(default ``PyAudioSource``), and a blocking ``WakeWordListener`` that ties them
together. Copy this folder into another project to reuse it.
"""

from __future__ import annotations

from .audio import AudioSource, PyAudioSource
from .engine import WakeWordEngine
from .listener import WakeWordListener

__all__ = [
    "AudioSource",
    "PyAudioSource",
    "WakeWordEngine",
    "WakeWordListener",
]
