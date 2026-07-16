"""Generic end-session tool.

Brain-agnostic and always on: lets the user dismiss a voice session immediately
("stop", "never mind", false wake, etc.) instead of waiting for the idle
timeout. Declared always-on in the central tool registry.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams

from ..bundle import ToolBundle
from ..contracts import ToolActivation, ToolBackend, ToolId, ToolSpec

END_SESSION_INSTRUCTION = (
    "You can end the voice session immediately. Call end_session when the user "
    "dismisses you: stop, shut up, never mind, cancel, go away, or a false wake "
    "they want to ignore. Call it right away and do not speak afterward. Sometimes "
    "you might hear some voice notes after the user asks you to quit  -- it is fine to "
    "ignore the content after since the user has already asked you to quit."
    "They can always start a new session if they want to continue the conversation."
)


async def end_session(params: FunctionCallParams, reason: str = "") -> None:
    """End the voice session immediately.

    Use when the user dismisses the assistant or wants silence: stop, shut up,
    never mind, cancel, go away, or a false wake they want to ignore. Do not
    keep talking after calling this.

    Args:
        reason: Brief note on why the session is ending (for logs only).
    """
    if reason:
        logger.info(f"end_session called (reason={reason!r})")
    else:
        logger.info("end_session called")

    await params.result_callback(
        {"status": "ended", "reason": reason or None},
        properties=FunctionCallResultProperties(run_llm=False),
    )
    await params.pipeline_worker.cancel(reason="user dismissed session")


async def setup_session_tools() -> ToolBundle:
    """Return the always-on end-session tool."""
    return ToolBundle(
        standard_tools=[end_session],
        instructions=[END_SESSION_INSTRUCTION],
    )


SPEC = ToolSpec(
    id=ToolId.SESSION,
    activation=ToolActivation.ALWAYS,
    backend=ToolBackend.LOCAL,
    setup=setup_session_tools,
)


__all__ = ["END_SESSION_INSTRUCTION", "SPEC", "end_session", "setup_session_tools"]
