# app/memory/redis_memory.py

"""
RedisMemoryManager

Manages short-term conversation history for the customer support agent.
All conversation data is stored in Redis using List and Hash data structures.

Key design:
  session:{session_id}:history  →  Redis List of JSON-serialised ConversationTurn
  session:{session_id}:meta     →  Redis Hash of SessionMeta fields

TTL (time-to-live) is applied to both keys so stale sessions auto-expire
without any manual cleanup. Default: 24 hours.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import redis

from app.config import get_settings
from app.schemas.memory import ConversationTurn, SessionMeta

logger = logging.getLogger(__name__)


class RedisMemoryManager:
    """
    Handles all Redis read/write operations for short-term conversation memory.

    Usage:
        manager = RedisMemoryManager()

        # Save a completed turn
        manager.save_turn(
            session_id="ses_abc",
            user_id="usr_123",
            human_message="Where is my order?",
            assistant_message="Can you share your order number?",
            intent="ORDER",
        )

        # Retrieve last 10 turns before building the LLM prompt
        turns = manager.get_history("ses_abc", n=10)

        # Format turns as a readable dialogue string
        text = manager.format_as_dialogue(turns)
    """

    def __init__(self, client: Optional[redis.Redis] = None):
        """
        Accepts an optional pre-built Redis client.
        If none is provided, creates one from config.

        Accepting an injected client makes testing easy:
        you can pass in a mock Redis client in unit tests
        without touching real Redis at all.
        """
        self._settings = get_settings()

        if client is not None:
            self._client = client
        else:
            self._client = redis.from_url(
                self._settings.redis_url,
                db=self._settings.redis_db,
                decode_responses=True,      # return str, not bytes
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )

    # ── Key helpers ───────────────────────────────────────────────────────────

    def _history_key(self, session_id: str) -> str:
        """Redis key for the conversation history list."""
        return f"session:{session_id}:history"

    def _meta_key(self, session_id: str) -> str:
        """Redis key for the session metadata hash."""
        return f"session:{session_id}:meta"

    # ── Core operations ───────────────────────────────────────────────────────

    def save_turn(
        self,
        session_id: str,
        user_id: str,
        human_message: str,
        assistant_message: str,
        intent: str = "",
    ) -> ConversationTurn:
        """
        Append one completed turn to the session's history in Redis.

        Steps:
          1. Build a ConversationTurn object with all fields populated
          2. Serialise it to a JSON string
          3. RPUSH it to the history list (append to the right/end)
          4. LTRIM the list to keep only the last buffer_memory_size turns
             (this is what makes it a "sliding window" buffer)
          5. (Re)set the TTL on both keys so the session clock resets
             on every new message
          6. Update session metadata (turn count, last active)

        The LTRIM is the key operation that enforces the buffer window.
        After every save, the list contains at most buffer_memory_size turns.
        Older turns are permanently removed from short-term memory.
        (They will be summarised and moved to long-term memory in Phase 2.)

        Returns the saved ConversationTurn for confirmation.
        """
        history_key = self._history_key(session_id)
        meta_key = self._meta_key(session_id)

        # Get current turn count for the turn_index field
        current_length = self._client.llen(history_key)

        turn = ConversationTurn(
            role_human=human_message,
            role_assistant=assistant_message,
            intent=intent,
            turn_index=current_length,  # 0-based index
            session_id=session_id,
            user_id=user_id,
        )

        # Use a Redis pipeline: batch all commands into one network round-trip
        # This is more efficient than sending each command separately
        pipe = self._client.pipeline()

        # Append the serialised turn to the list
        pipe.rpush(history_key, turn.to_json())

        # Trim to keep only the last buffer_memory_size turns
        # -buffer_memory_size means "from the Nth-from-last position"
        pipe.ltrim(
            history_key,
            -self._settings.buffer_memory_size,
            -1
        )

        # Reset TTL on history key (24h from now)
        pipe.expire(history_key, self._settings.redis_ttl_seconds)

        # Upsert session metadata
        now = datetime.now(timezone.utc).isoformat()
        pipe.hset(meta_key, mapping={
            "session_id": session_id,
            "user_id": user_id,
            "last_active": now,
        })

        # HSETNX = set only if field does not already exist
        # This preserves the original created_at timestamp
        pipe.hsetnx(meta_key, "created_at", now)

        # Increment turn_count in the hash
        pipe.hincrby(meta_key, "turn_count", 1)

        # Reset TTL on meta key too
        pipe.expire(meta_key, self._settings.redis_ttl_seconds)

        pipe.execute()

        logger.debug(
            "Saved turn %d for session %s (intent: %s)",
            turn.turn_index,
            session_id,
            intent or "unclassified",
        )

        return turn

    def get_history(
        self,
        session_id: str,
        n: Optional[int] = None,
    ) -> list[ConversationTurn]:
        """
        Retrieve the last N turns for a session.

        If n is None, uses buffer_memory_size from config.
        If n is larger than what's stored, returns everything stored.
        If the session does not exist, returns an empty list (not an error).

        The list is returned in chronological order (oldest first) —
        this is the natural order for building an LLM prompt.
        """
        n = n or self._settings.buffer_memory_size
        history_key = self._history_key(session_id)

        # LRANGE with negative indices counts from the end
        # -n = nth from last, -1 = last item → returns last n items in order
        raw_turns = self._client.lrange(history_key, -n, -1)

        if not raw_turns:
            return []

        turns = []
        for raw in raw_turns:
            try:
                turns.append(ConversationTurn.from_json(raw))
            except Exception as e:
                # Skip malformed entries rather than crashing
                logger.warning("Skipping malformed turn in session %s: %s", session_id, e)

        return turns

    def get_session_meta(self, session_id: str) -> Optional[SessionMeta]:
        """
        Retrieve metadata for a session.
        Returns None if the session does not exist.
        """
        meta_key = self._meta_key(session_id)
        data = self._client.hgetall(meta_key)

        if not data:
            return None

        return SessionMeta(
            session_id=data.get("session_id", session_id),
            user_id=data.get("user_id", ""),
            created_at=data.get("created_at", ""),
            turn_count=int(data.get("turn_count", 0)),
            last_active=data.get("last_active", ""),
        )

    def session_exists(self, session_id: str) -> bool:
        """Check if a session has any history stored."""
        return bool(self._client.exists(self._history_key(session_id)))

    def get_turn_count(self, session_id: str) -> int:
        """Return the number of turns currently stored for a session."""
        return self._client.llen(self._history_key(session_id))

    def clear_session(self, session_id: str) -> None:
        """
        Delete all history and metadata for a session.

        Called when:
        - A conversation formally ends (DELETE /sessions/{id} in Phase 5)
        - Before that endpoint exists, called manually in tests

        Note: In Phase 2, the caller is responsible for summarising the
        session and saving it to the vector store BEFORE calling this.
        Clearing without summarising loses the conversation permanently.
        """
        pipe = self._client.pipeline()
        pipe.delete(self._history_key(session_id))
        pipe.delete(self._meta_key(session_id))
        pipe.execute()
        logger.debug("Cleared session %s from Redis", session_id)

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_as_dialogue(self, turns: list[ConversationTurn]) -> str:
        """
        Convert a list of ConversationTurns into a readable dialogue string
        suitable for insertion into an LLM prompt.

        Output format:
            Human: Where is my order?
            Assistant: Can you share your order number?
            Human: It's #12345.
            Assistant: Let me look that up for you.

        This is the format most instruction-tuned models (llama3.2, gemma2)
        are trained to understand as a conversation history.

        Empty assistant replies (partially saved turns) are skipped.
        """
        if not turns:
            return ""

        lines = []
        for turn in turns:
            lines.append(f"Human: {turn.role_human.strip()}")
            if turn.role_assistant.strip():
                lines.append(f"Assistant: {turn.role_assistant.strip()}")

        return "\n".join(lines)

    def get_formatted_history(
        self,
        session_id: str,
        n: Optional[int] = None,
    ) -> str:
        """
        Convenience method: get history and format it in one call.

        This is what the retrieve_short_term node in Phase 4 will call:
            context = memory_manager.get_formatted_history(session_id)
        """
        turns = self.get_history(session_id, n=n)
        return self.format_as_dialogue(turns)

    # ── Health check ──────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """
        Check if Redis is reachable.
        Used by the /health endpoint in Phase 5.
        Returns True if Redis responds, False if not.
        """
        try:
            return self._client.ping()
        except Exception:
            return False