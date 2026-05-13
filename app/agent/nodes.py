# app/agent/nodes.py

"""
LangGraph node factory functions for the customer support agent.

Each public function is a factory: it accepts a dependency and returns a
closure that satisfies the LangGraph node signature `(state) -> dict`.

The returned dict is merged into AgentState by the graph runtime — nodes
only need to return the fields they write.

Nodes follow the codebase-wide graceful degradation pattern:
  - Memory retrieval failures (Redis, Chroma) degrade to empty context.
  - LLM failure in generate_response increments retry_count and sets error;
    the routing edge decides whether to retry or route to log_failure.
  - handle_fallback and log_failure never raise.
"""

import logging
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.agent.state import AgentState
from app.agent.prompts import get_system_prompt, build_context_prompt
from app.guardrails.pipeline import GuardrailPipeline
from app.memory.redis_memory import RedisMemoryManager
from app.memory.vector_memory import VectorMemoryManager

logger = logging.getLogger(__name__)

_GENERATION_ERROR_RESPONSE = (
    "I'm sorry, I'm having trouble processing your request right now. "
    "Please try again in a moment."
)


# ── Node factories ────────────────────────────────────────────────────────────

def make_classify_intent(pipeline: GuardrailPipeline) -> Callable[[AgentState], dict]:
    """Stage 0-2 safety + intent classification via GuardrailPipeline."""

    def classify_intent(state: AgentState) -> dict:
        result = pipeline.run(state["message"])
        return {
            "intent": result.intent,
            "is_on_topic": result.is_on_topic,
            "blocked_reason": result.blocked_reason,
            "fallback_response": result.fallback_response,
            "error": None,
        }

    return classify_intent


def make_retrieve_short_term(redis_manager: RedisMemoryManager) -> Callable[[AgentState], dict]:
    """Fetch the recent conversation buffer from Redis. Fails OPEN (returns empty string)."""

    def retrieve_short_term(state: AgentState) -> dict:
        try:
            history = redis_manager.get_formatted_history(state["session_id"])
        except Exception as exc:
            logger.warning(
                "Short-term memory retrieval failed for session %s: %s",
                state["session_id"], exc,
            )
            history = ""
        return {"short_term_history": history}

    return retrieve_short_term


def make_retrieve_long_term(vector_manager: VectorMemoryManager) -> Callable[[AgentState], dict]:
    """Retrieve relevant past-session summaries from Chroma. Fails OPEN (returns empty string)."""

    def retrieve_long_term(state: AgentState) -> dict:
        try:
            context = vector_manager.retrieve_as_text(state["user_id"], state["message"])
        except Exception as exc:
            logger.warning(
                "Long-term memory retrieval failed for user %s: %s",
                state["user_id"], exc,
            )
            context = ""
        return {"long_term_context": context}

    return retrieve_long_term


def build_context(state: AgentState) -> dict:
    """Assemble long-term context + short-term history + current message into final_prompt."""
    final_prompt = build_context_prompt(
        message=state["message"],
        short_term_history=state.get("short_term_history", ""),  # type: ignore[arg-type]
        long_term_context=state.get("long_term_context", ""),  # type: ignore[arg-type]
    )
    return {"final_prompt": final_prompt}


def make_generate_response(llm: BaseChatModel) -> Callable[[AgentState], dict]:
    """
    Call the main LLM with intent-specific system prompt + assembled context.

    On success: sets response, clears error.
    On failure: increments retry_count, sets error (routing edge decides next step).
    """

    def generate_response(state: AgentState) -> dict:
        system_prompt = get_system_prompt(state["intent"])
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["final_prompt"]),
        ]
        try:
            response = llm.invoke(messages).content
            return {
                "response": response,
                "error": None,
                "retry_count": state.get("retry_count", 0),  # type: ignore[arg-type]
            }
        except Exception as exc:
            retry_count = state.get("retry_count", 0) + 1  # type: ignore[operator]
            logger.warning(
                "LLM generation failed (attempt %d) for session %s: %s",
                retry_count, state["session_id"], exc,
            )
            return {
                "response": "",
                "error": str(exc),
                "retry_count": retry_count,
            }

    return generate_response


def handle_fallback(state: AgentState) -> dict:
    """
    Return the pre-generated canned response for blocked messages.

    The fallback_response was set by GuardrailPipeline.run() and stored in
    state by classify_intent; no additional LLM call is made here.
    """
    return {
        "response": (
            state.get("fallback_response")  # type: ignore[arg-type]
            or "I can only assist with e-commerce customer support questions."
        ),
    }


def make_save_memory(redis_manager: RedisMemoryManager) -> Callable[[AgentState], None]:
    """Persist the completed turn to Redis. Failure is logged but does not abort the turn."""

    def save_memory(state: AgentState) -> None:
        try:
            redis_manager.save_turn(
                session_id=state["session_id"],
                user_id=state["user_id"],
                human_message=state["message"],
                assistant_message=state["response"],
                intent=state.get("intent", ""),  # type: ignore[arg-type]
            )
        except Exception as exc:
            logger.warning(
                "Failed to save turn to Redis for session %s: %s",
                state["session_id"], exc,
            )

    return save_memory


def log_failure(state: AgentState) -> dict:
    """
    Terminal error node: logs the failure and returns a canned error response.

    Reached only after generate_response exhausts all retries (max_retries).
    LangSmith captures the full graph trace automatically via LANGCHAIN_TRACING_V2.
    """
    logger.error(
        "Agent generation failed after %d attempt(s) — session=%s user=%s error=%r",
        state.get("retry_count", 0),  # type: ignore[arg-type]
        state.get("session_id", "unknown"),  # type: ignore[arg-type]
        state.get("user_id", "unknown"),  # type: ignore[arg-type]
        state.get("error", "unknown"),  # type: ignore[arg-type]
    )
    return {"response": _GENERATION_ERROR_RESPONSE}
