"""Synthesize a short readiness chime so the user knows the bot is listening.

Mirrors the LiveKit client's chime (livekit-client/.../sound.py): a quick rising
two-note "ding" with a bell-like exponential decay. Here we only *generate* the
PCM; playback goes through Pipecat's output transport (as a
``ReadinessChimeFrame``) rather than a separate audio device, so it routes
through the same speakers. Unlike bot speech, the chime is deliberately *not*
fed to the WebRTC APM as an echo-cancellation reference: as a session's first
sound it would mis-adapt the cold echo canceller and swallow the user's first
sentence (see ``transport_local.py``). The half-duplex gate keeps its echo out
of the capture path instead.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass

from pipecat.frames.frames import OutputAudioRawFrame

SAMPLE_RATE = 44100
# (frequency_hz, duration_seconds) per note. A quick rising two-note "ding".
_NOTES = ((988.0, 0.10), (1318.5, 0.18))
_AMPLITUDE = 0.3
# Exponential decay so each note sounds like a bell rather than a flat beep.
_DECAY = 9.0


def chime_pcm() -> tuple[bytes, int]:
    """Return the readiness chime as (mono 16-bit PCM bytes, sample_rate).

    The output transport resamples to its own rate, so the 44.1 kHz here is just
    the synthesis rate.
    """
    samples = array.array("h")
    for freq, duration in _NOTES:
        frame_count = int(SAMPLE_RATE * duration)
        for i in range(frame_count):
            t = i / SAMPLE_RATE
            envelope = math.exp(-t * _DECAY)
            value = _AMPLITUDE * envelope * math.sin(2.0 * math.pi * freq * t)
            samples.append(int(max(-1.0, min(1.0, value)) * 32767))
    return samples.tobytes(), SAMPLE_RATE


@dataclass
class ReadinessChimeFrame(OutputAudioRawFrame):
    """The readiness chime, as its own output-frame type for special handling.

    It is a plain output frame (not a ``TTSAudioRawFrame``), so it still does not
    count as bot speech - no idle-timer reset, no interruption logic. It is a
    distinct subclass so the local transport (see ``transport_local.py``) can give
    it the two-layer treatment a session's first sound needs on a hardware
    speakerphone, where the chime otherwise swallows the user's first sentence:

    1. The half-duplex gate closes the mic while it plays (plus a short tail), so
       the chime's echo never reaches VAD/STT in the pipeline.
    2. It is *not* fed to the echo canceller as a far-end reference, so the cold,
       freshly-built canceller can't mis-adapt to the loud tone and then
       over-suppress the user's near-end speech for the next few seconds.

    The soft "working" cue (``working_sound.py``) stays a plain
    ``OutputAudioRawFrame`` and gets neither treatment.
    """


# A low, soft "blip" cue played occasionally while the bot is busy (slow
# reasoning or a tool call), so the user hears an unobtrusive sign of life
# instead of dead silence. The intent is classy and heavy: a couple of deep,
# warm tones spread out over a long gap, not a chatty tick. The long gap between
# cycles is realized by the caller sleeping (see working_sound.py) rather than
# baked into the buffer - the output transport plays audio FIFO, so queuing
# seconds of silence would sit in front of the bot's real speech and delay it.
#
# (frequency_hz, duration_seconds, relative_gain, gap_after_seconds): two low,
# soft blips - a deeper one answering a slightly higher one - then the within-
# cycle spacing. Frequencies sit in the low register for weight.
_WS_BLIPS = (
    (164.81, 0.30, 1.00, 0.38),  # E3
    (130.81, 0.42, 0.85, 0.00),  # C3, lower and a touch longer -> "heavier"
)
# Soft: kept well below the chime's 0.3 so the cue is gentle on echo
# cancellation / barge-in detection.
_WS_AMPLITUDE = 0.4
# Gentle (small) decay for a warm, sustained tone with a heavy tail, rather than
# a sharp pluck.
_WS_DECAY = 3.2
# Short fades in/out of each tone so the soft onset/cutoff doesn't click.
_WS_ATTACK_SECS = 0.012
_WS_RELEASE_SECS = 0.04
# A faint second harmonic adds warmth/body without making it a flat sine.
_WS_PARTIAL_GAIN = 0.18


def _render_tone(
    samples: array.array, freq: float, duration: float, gain: float
) -> None:
    """Append one soft, click-free tone (fundamental + faint 2nd harmonic)."""
    frame_count = int(SAMPLE_RATE * duration)
    attack = max(1, int(SAMPLE_RATE * _WS_ATTACK_SECS))
    release = max(1, int(SAMPLE_RATE * _WS_RELEASE_SECS))
    for i in range(frame_count):
        t = i / SAMPLE_RATE
        envelope = math.exp(-t * _WS_DECAY)
        if i < attack:
            envelope *= i / attack
        remaining = frame_count - i
        if remaining < release:
            envelope *= remaining / release
        wave = math.sin(2.0 * math.pi * freq * t) + _WS_PARTIAL_GAIN * math.sin(
            2.0 * math.pi * 2.0 * freq * t
        )
        value = _WS_AMPLITUDE * gain * envelope * wave
        samples.append(int(max(-1.0, min(1.0, value)) * 32767))


def working_sound_pcm() -> tuple[bytes, int]:
    """Return one short "blip" cycle as (mono 16-bit PCM bytes, sample_rate).

    Just the blips (plus the brief within-cycle spacing) - the long, spread-out
    gap between repeats is the caller's job (sleep), not baked into the buffer,
    so the cue never sits in front of the bot's speech in the output's FIFO. The
    output transport resamples to its own rate, so 44.1 kHz here is just the
    synthesis rate.
    """
    samples = array.array("h")
    for freq, duration, gain, gap_after in _WS_BLIPS:
        _render_tone(samples, freq, duration, gain)
        if gap_after:
            samples.extend([0] * int(SAMPLE_RATE * gap_after))
    return samples.tobytes(), SAMPLE_RATE
