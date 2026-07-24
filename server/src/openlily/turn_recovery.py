"""Recover from "ghost" user turns that cancel a response and leave it dead.

The failure this fixes: while the bot is thinking (not yet speaking), a raw VAD
false-trigger -- a breath, cough, or background noise -- can start a user turn.
Starting a turn broadcasts an interruption, which cancels the in-flight LLM
generation. Since the user said nothing, no transcript ever arrives; the turn
controller's stop watchdog force-ends the turn after a few seconds, but the
user aggregator's ``push_aggregation()`` early-returns on an empty aggregation,
so no context frame is pushed and the LLM never re-runs. The user's question
sits unanswered in the context forever.

This module wires the recovery: subscribe to the user aggregator's
``on_user_turn_stopped`` event, and when a turn ends with no user content while
the context still ends in an unanswered user message (or a tool call that the
interruption cancelled), re-trigger inference by queueing an ``LLMRunFrame``
into the pipeline. The frame is queued (not a direct context/method call) so it
stays ordered with everything else flowing through the aggregator.

Loop guards:

- A turn that produced real user content never re-triggers (normal flow
  already runs the LLM), and it resets the attempt counter.
- No re-trigger while an assistant response is already in flight (tracked via
  the assistant aggregator's turn events).
- At most ``MAX_CONSECUTIVE_RECOVERY_ATTEMPTS`` consecutive re-triggers, reset
  by any completed (un-interrupted, non-empty) assistant response, so a broken
  LLM can't cause infinite regeneration. Each attempt additionally requires a
  fresh empty turn stop, so this can never tight-loop.

Realtime caveat (verified against pipecat-ai 1.4.0): in realtime mode
(``realtime_service_mode=True``) ``message.content`` is always ``None`` at turn
stop -- the user message is committed later, when the assistant response
starts -- so "did the user actually say something" is checked against the
aggregator's pending (not yet flushed) aggregation instead. Also note that in
1.4.0 the OpenAI Realtime service only creates a response for its *initial*
context frame or a new tool result, so the re-triggered ``LLMRunFrame`` is a
harmless no-op there (the context still updates); cascade brains recover
fully. If pipecat later honors mid-session run frames for realtime services,
this recovery starts working there with no changes here.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMAssistantAggregator,
    LLMUserAggregator,
    UserTurnStoppedMessage,
)

# Upper bound on *consecutive* recovery re-triggers (reset by any completed
# assistant response or any user turn with real content). Each attempt also
# requires a fresh empty turn stop -- i.e. another VAD false-trigger -- so this
# only bounds pathological cases like an LLM that errors on every run.
MAX_CONSECUTIVE_RECOVERY_ATTEMPTS = 3

# Sentinel pipecat writes as a tool message's content when an interruption
# cancels the call mid-flight (see LLMAssistantAggregator._handle_function_call_cancel).
_CANCELLED_TOOL_RESULT = "CANCELLED"


def has_unanswered_user_message(context: LLMContext) -> bool:
    """True if the context's conversation tail is still waiting on the bot.

    Walks messages from the end, skipping provider-specific and
    system/developer messages. The tail is "unanswered" when it is a user
    message (the interruption cancelled the response before any assistant text
    landed) or a tool result the interruption cancelled (the response died
    mid-tool-call). A trailing assistant message -- or a tool result that
    completed normally, which pipecat itself follows up on -- means nothing is
    owed.
    """
    for message in reversed(context.get_messages()):
        if isinstance(message, LLMSpecificMessage):
            continue
        role = message.get("role")
        if role in ("system", "developer"):
            continue
        if role == "user":
            return True
        if role == "tool":
            return message.get("content") == _CANCELLED_TOOL_RESULT
        return False
    return False


@dataclass
class EmptyTurnRecovery:
    """Re-runs the LLM when an empty user turn left a question unanswered.

    Wire it with :func:`wire_empty_turn_recovery`; the public ``on_*`` methods
    are the event-handler entry points (kept separate from the wiring so the
    decision logic is unit-testable with fakes).
    """

    user_aggregator: LLMUserAggregator
    assistant_aggregator: LLMAssistantAggregator
    context: LLMContext
    max_attempts: int = MAX_CONSECUTIVE_RECOVERY_ATTEMPTS

    _attempts: int = 0
    _response_in_flight: bool = False

    async def on_user_turn_stopped(self, message: UserTurnStoppedMessage) -> None:
        """Decide whether this turn stop needs a recovery re-trigger."""
        # Pending aggregation covers realtime mode, where content is always
        # None at turn stop and any transcript that has already arrived sits
        # in the aggregator awaiting the deferred flush on assistant-response
        # start. In cascade mode the aggregation was flushed before this event
        # fired, so content alone decides.
        pending = self.user_aggregator.aggregation_string().strip()
        if message.content or pending:
            # The turn carried real user content; the normal flow runs the LLM.
            self._attempts = 0
            return

        if self._response_in_flight:
            return

        if not has_unanswered_user_message(self.context):
            return

        if self._attempts >= self.max_attempts:
            logger.warning(
                f"Empty user turn left an unanswered question, but recovery already "
                f"re-ran the LLM {self._attempts} times without a completed response; "
                "giving up on this question"
            )
            return

        self._attempts += 1
        logger.warning(
            f"User turn ended with no content but the context ends in an unanswered "
            f"user message (likely a VAD false-trigger cancelled the response); "
            f"re-running the LLM (attempt {self._attempts}/{self.max_attempts})"
        )
        await self.user_aggregator.queue_frame(LLMRunFrame())

    def on_assistant_turn_started(self) -> None:
        """Track that a response is now in flight."""
        self._response_in_flight = True

    def on_assistant_turn_stopped(self, message: AssistantTurnStoppedMessage) -> None:
        """Track response completion; a real completed answer resets the cap."""
        self._response_in_flight = False
        if message.content and not message.interrupted:
            self._attempts = 0


def wire_empty_turn_recovery(
    user_aggregator: LLMUserAggregator,
    assistant_aggregator: LLMAssistantAggregator,
    context: LLMContext,
) -> EmptyTurnRecovery:
    """Subscribe an :class:`EmptyTurnRecovery` to the aggregator pair's events."""
    recovery = EmptyTurnRecovery(
        user_aggregator=user_aggregator,
        assistant_aggregator=assistant_aggregator,
        context=context,
    )

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def _on_user_turn_stopped(_aggregator, _strategy, message: UserTurnStoppedMessage):
        await recovery.on_user_turn_stopped(message)

    @assistant_aggregator.event_handler("on_assistant_turn_started")
    async def _on_assistant_turn_started(_aggregator):
        recovery.on_assistant_turn_started()

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def _on_assistant_turn_stopped(_aggregator, message: AssistantTurnStoppedMessage):
        recovery.on_assistant_turn_stopped(message)

    return recovery


__all__ = [
    "MAX_CONSECUTIVE_RECOVERY_ATTEMPTS",
    "EmptyTurnRecovery",
    "has_unanswered_user_message",
    "wire_empty_turn_recovery",
]
