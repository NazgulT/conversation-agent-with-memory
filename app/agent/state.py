# app/agent/state.py

from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Immutable per-turn state threaded through the LangGraph graph.

    Fields are populated progressively as nodes execute:
      - session_id / user_id / message: provided by the caller before graph entry
      - intent … fallback_response: written by classify_intent
      - short_term_history: written by retrieve_short_term
      - long_term_context: written by retrieve_long_term
      - final_prompt: written by build_context
      - response: written by generate_response, handle_fallback, or log_failure
      - retry_count / error: managed by generate_response for the retry loop
    """

    # ── Input (always required) ───────────────────────────────────────────────
    session_id: str
    user_id: str
    message: str

    # ── classify_intent outputs ───────────────────────────────────────────────
    intent: str
    is_on_topic: bool
    blocked_reason: Optional[str]
    fallback_response: Optional[str]

    # ── Memory retrieval outputs ──────────────────────────────────────────────
    short_term_history: str
    long_term_context: str

    # ── build_context output ──────────────────────────────────────────────────
    final_prompt: str

    # ── Response and retry tracking ───────────────────────────────────────────
    response: str
    retry_count: int
    error: Optional[str]
