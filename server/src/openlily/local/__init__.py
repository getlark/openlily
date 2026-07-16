"""Local-audio building blocks for running openlily on the machine's mic/speakers.

These are the pieces the terminal voice CLI uses but that a cloud deployment does
not need: the gated local-audio transport (AEC/NS/AGC + half-duplex gating), the
portable wake-word detection package, and wake-word barge-in. They live in their
own subpackage so importing the core library never pulls in PyAudio/openWakeWord.
"""

from __future__ import annotations

from .transport import (
    GatedLocalAudioTransport,
    build_local_transport,
    close_local_transport,
)

__all__ = [
    "GatedLocalAudioTransport",
    "build_local_transport",
    "close_local_transport",
]
