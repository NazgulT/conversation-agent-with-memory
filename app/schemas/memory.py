# app/schemas/memory.py

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """
    One complete exchange: one human message + one assistant reply.

    Both sides are stored together so a single LRANGE call returns
    complete, paired turns — no interleaving logic needed at retrieval.

    Fields:
        role_human      : What the user said
        role_assistant  : What the agent replied (empty string if not yet
                          generated — turns can be saved in two steps)
        timestamp       : When this turn started (UTC, ISO 8601)
        intent          : Classified intent from the guardrail node
                          (empty in Phase 1, filled in Phase 3)
        turn_index      : Position in the conversation (0-based)
        session_id      : Which session this turn belongs to
        user_id         : Which user this session belongs to
    """

    role_human: str
    role_assistant: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    intent: str = ""
    turn_index: int = 0
    session_id: str = ""
    user_id: str = ""

    def to_json(self) -> str:
        """Serialise to JSON string for storage in Redis."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "ConversationTurn":
        """Deserialise from a Redis-stored JSON string."""
        return cls.model_validate_json(json_str)


class SessionMeta(BaseModel):
    """
    Metadata stored alongside a conversation session.
    Stored as a Redis Hash (not a List).
    """

    session_id: str
    user_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    turn_count: int = 0
    last_active: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )