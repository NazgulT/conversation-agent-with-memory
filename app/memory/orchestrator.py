# app/memory/orchestrator.py

"""
Memory Orchestrator

Coordinates the session-end memory flow:
  1. Get turns from Redis  (RedisMemoryManager)
  2. Summarise turns       (ConversationSummariser)
  3. Store summary         (VectorMemoryManager)
  4. Clear Redis session   (RedisMemoryManager)

This function is called:
  - By the DELETE /sessions/{id} endpoint (Phase 5)
  - By a background inactivity job (Phase 8 — 30min timeout)
  - Directly in tests and the manual test script

It is NOT called during a live conversation — only at session end.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from app.memory.redis_memory import RedisMemoryManager
from app.memory.vector_memory import VectorMemoryManager
from app.memory.summariser import ConversationSummariser
from app.schemas.memory import MemorySummary

logger = logging.getLogger(__name__)


@dataclass
class SessionEndResult:
    """
    Result of the session-end orchestration flow.
    Returned so callers (API endpoints, tests) can inspect what happened.
    """
    session_id: str
    user_id: str
    turns_processed: int
    summary: Optional[MemorySummary]
    stored_in_chroma: bool
    redis_cleared: bool
    skipped: bool           # True if conversation was too short to summarise
    error: Optional[str]    # Non-None if something went wrong


def end_session(
    session_id: str,
    user_id: str,
    redis_manager: Optional[RedisMemoryManager] = None,
    vector_manager: Optional[VectorMemoryManager] = None,
    summariser: Optional[ConversationSummariser] = None,
) -> SessionEndResult:
    """
    Run the complete session-end memory flow for one session.

    All three dependencies are injectable for testing.
    If not provided, each creates its own default instance.

    This function never raises exceptions — errors are captured into
    the returned SessionEndResult so the caller can decide how to
    handle them (log, alert, retry).

    Args:
        session_id     : The Redis session to end
        user_id        : The user who owns this session
        redis_manager  : Optional injected RedisMemoryManager
        vector_manager : Optional injected VectorMemoryManager
        summariser     : Optional injected ConversationSummariser

    Returns:
        SessionEndResult with full details of what happened
    """
    # Initialise dependencies if not injected
    redis_mgr  = redis_manager  or RedisMemoryManager()
    vector_mgr = vector_manager or VectorMemoryManager()
    summ       = summariser     or ConversationSummariser()

    result = SessionEndResult(
        session_id=session_id,
        user_id=user_id,
        turns_processed=0,
        summary=None,
        stored_in_chroma=False,
        redis_cleared=False,
        skipped=False,
        error=None,
    )

    try:
        # Step 1: Get all turns from Redis
        turns = redis_mgr.get_history(session_id)
        result.turns_processed = len(turns)

        if len(turns) < 2:
            logger.info(
                "Session %s has %d turns — too short to summarise, clearing Redis only.",
                session_id, len(turns)
            )
            result.skipped = True
            redis_mgr.clear_session(session_id)
            result.redis_cleared = True
            return result

        # Step 2: Summarise the turns (LLM call — llama3.2:3b)
        logger.info("Summarising session %s (%d turns)...", session_id, len(turns))
        summary = summ.summarise(
            turns=turns,
            user_id=user_id,
            session_id=session_id,
        )

        if summary is None:
            result.skipped = True
            redis_mgr.clear_session(session_id)
            result.redis_cleared = True
            return result

        result.summary = summary

        # Step 3: Store the summary in Chroma
        stored_summary = vector_mgr.store_summary(summary)
        result.stored_in_chroma = True
        result.summary = stored_summary

        # Step 4: Clear the Redis session
        # NOTE: This is done AFTER Chroma storage succeeds.
        # If Chroma storage fails, the exception is caught below and
        # Redis is NOT cleared — the data is preserved for retry.
        redis_mgr.clear_session(session_id)
        result.redis_cleared = True

        logger.info(
            "Session %s ended: %d turns → summary stored → Redis cleared. "
            "topics=%s resolution=%s",
            session_id,
            len(turns),
            summary.topics,
            summary.resolution,
        )

    except Exception as e:
        result.error = str(e)
        logger.error("Session end failed for %s: %s", session_id, e, exc_info=True)
        # Do NOT clear Redis if something failed — data is preserved

    return result