"""openWakeWord wake-word engine.

A pure detection engine: feed it 16 kHz mono int16 PCM frames and it returns the
label of a detected wake word (or ``None``). It owns no audio device and has no
knowledge of how frames are captured -- that's the job of an ``AudioSource`` (see
``audio.py``). This keeps the engine trivially portable: it depends only on
``openwakeword`` + ``numpy``.

Copied (and lightly trimmed) from the standalone ``wake_word_detector`` project
so this package can be lifted into any project without pulling in Pipecat.
"""

from __future__ import annotations

# openWakeWord's pretrained wake phrases (pass any of these by name as a model):
#   "alexa"        -> "Alexa"
#   "hey_jarvis"   -> "Hey Jarvis"
#   "hey_mycroft"  -> "Hey Mycroft"
#   "hey_rhasspy"  -> "Hey Rhasspy"
# (It also ships "timer"/"weather" command models, which aren't wake phrases.)
# Custom models can be passed as .onnx/.tflite file paths instead of a name.
DEFAULT_MODEL = "alexa"


class WakeWordEngine:
    """openWakeWord wake-word engine (open source, no API key)."""

    def __init__(
        self,
        models: list[str] | None = None,
        threshold: float = 0.5,
        inference_framework: str = "onnx",
    ) -> None:
        import numpy as np
        import openwakeword.utils
        from openwakeword.model import Model

        self._np = np
        self._threshold = threshold

        models = models or [DEFAULT_MODEL]

        # Pretrained models referenced by name need their files downloaded once.
        # File paths (custom models) are passed straight through.
        builtin = [m for m in models if not m.endswith((".onnx", ".tflite"))]
        if builtin:
            try:
                openwakeword.utils.download_models(builtin)
            except Exception:
                # Fall back to downloading the full default set if a targeted
                # download isn't supported by the installed version.
                openwakeword.utils.download_models()

        self._model = Model(
            wakeword_models=models,
            inference_framework=inference_framework,
        )

    def process(self, pcm: list[int]) -> str | None:
        """Score one frame of audio; return the wake-word label if one fired."""
        from typing import cast

        frame = self._np.array(pcm, dtype=self._np.int16)
        scores = cast("dict[str, float]", self._model.predict(frame))
        best_label: str | None = None
        best_score = self._threshold
        for label, score in scores.items():
            if score >= best_score:
                best_label = label
                best_score = score
        return best_label

    def reset(self) -> None:
        """Clear the model's rolling audio/feature buffers (~10s).

        Call this after a wake word fires and listening pauses, so audio heard
        before the pause (e.g. the wake word itself) can't immediately re-fire a
        detection when listening resumes.
        """
        self._model.reset()
