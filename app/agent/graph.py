# app/agent/graph.py

"""
CustomerSupportAgent — LangGraph-based conversational agent (Phase 4).

Graph topology:

    START
      └─► classify_intent
            ├─(blocked)────────────► handle_fallback ──────────────────► END
            └─(on-topic)──────────► retrieve_short_term
                                          └─► retrieve_long_term
                                                    └─► build_context
                                                              └─► generate_response
                                                                    ├─(error, retries left)──► [retry]
                                                                    ├─(error, no retries)────► log_failure ──► END
                                                                    └─(success)──────────────► save_memory ──► END

Design decisions:
  - Stateless per-turn: the graph processes one message and returns.
    Session lifecycle (end_session) is Phase 5's responsibility.
  - All dependencies are injectable for testing (DI pattern from architectural_patterns.md).
  - Memory retrieval nodes fail OPEN: a Redis or Chroma error produces empty context,
    not a blocked message.
  - generate_response retries up to max_retries times on LLM failure before routing
    to log_failure, which returns a canned error response.
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import (
    make_classify_intent,
    make_retrieve_short_term,
    make_retrieve_long_term,
    build_context,
    make_generate_response,
    handle_fallback,
    make_save_memory,
    log_failure,
)
from app.guardrails.pipeline import GuardrailPipeline
from app.memory.redis_memory import RedisMemoryManager
from app.memory.vector_memory import VectorMemoryManager
from app.llm.provider import get_llm
from app.config import get_settings

logger = logging.getLogger(__name__)

def _route_after_classify(state: AgentState) -> str:
    return "retrieve_short_term" if state["is_on_topic"] else "handle_fallback"


def _make_route_after_generate(max_retries: int) -> object:
    def _route(state: AgentState) -> str:
        if not state.get("error"):
            return "save_memory"
        # retry_count was incremented inside generate_response before this edge runs
        if state.get("retry_count", 0) <= max_retries:  # type: ignore[operator]
            return "generate_response"
        return "log_failure"

    return _route


class CustomerSupportAgent:
    """
    Processes one conversational turn at a time.

    The compiled LangGraph is built once in __init__ and reused across all
    process_turn() calls. Each invocation is a fresh, isolated state dict.

    Args:
        pipeline       : GuardrailPipeline (injected for tests)
        redis_manager  : RedisMemoryManager (injected for tests)
        vector_manager : VectorMemoryManager (injected for tests)
        llm            : Main chat LLM (injected for tests)

    Usage:
        agent = CustomerSupportAgent()
        response = agent.process_turn(
            session_id="sess_abc123",
            user_id="usr_xyz",
            message="Where is my order #12345?",
        )
    """

    def __init__(
        self,
        pipeline: Optional[GuardrailPipeline] = None,
        redis_manager: Optional[RedisMemoryManager] = None,
        vector_manager: Optional[VectorMemoryManager] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        settings = get_settings()
        self._pipeline = pipeline or GuardrailPipeline()
        self._redis_manager = redis_manager or RedisMemoryManager()
        self._vector_manager = vector_manager or VectorMemoryManager()
        self._llm = llm or get_llm()
        self._max_retries: int = settings.max_retries
        self._graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # ── Nodes ─────────────────────────────────────────────────────────────
        workflow.add_node("classify_intent", make_classify_intent(self._pipeline))
        workflow.add_node("retrieve_short_term", make_retrieve_short_term(self._redis_manager))
        workflow.add_node("retrieve_long_term", make_retrieve_long_term(self._vector_manager))
        workflow.add_node("build_context", build_context)
        workflow.add_node("generate_response", make_generate_response(self._llm))
        workflow.add_node("handle_fallback", handle_fallback)
        workflow.add_node("save_memory", make_save_memory(self._redis_manager))
        workflow.add_node("log_failure", log_failure)

        # ── Entry point ───────────────────────────────────────────────────────
        workflow.set_entry_point("classify_intent")

        # ── Edges ─────────────────────────────────────────────────────────────
        workflow.add_conditional_edges(
            "classify_intent",
            _route_after_classify,
            {
                "retrieve_short_term": "retrieve_short_term",
                "handle_fallback": "handle_fallback",
            },
        )

        workflow.add_edge("retrieve_short_term", "retrieve_long_term")
        workflow.add_edge("retrieve_long_term", "build_context")
        workflow.add_edge("build_context", "generate_response")

        workflow.add_conditional_edges(
            "generate_response",
            _make_route_after_generate(self._max_retries),
            {
                "save_memory": "save_memory",
                "generate_response": "generate_response",
                "log_failure": "log_failure",
            },
        )

        workflow.add_edge("handle_fallback", END)
        workflow.add_edge("save_memory", END)
        workflow.add_edge("log_failure", END)

        return workflow.compile()

    def process_turn(
        self,
        session_id: str,
        user_id: str,
        message: str,
    ) -> str:
        """
        Process one conversational turn and return the agent's response.

        Never raises — any unhandled error surfaces via log_failure's canned
        response so the caller always receives a string.
        """
        initial_state: AgentState = {
            "session_id": session_id,
            "user_id": user_id,
            "message": message,
            "intent": "",
            "is_on_topic": True,
            "blocked_reason": None,
            "fallback_response": None,
            "short_term_history": "",
            "long_term_context": "",
            "final_prompt": "",
            "response": "",
            "retry_count": 0,
            "error": None,
        }
        result = self._graph.invoke(initial_state)
        return result["response"]
