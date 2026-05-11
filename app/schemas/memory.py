# app/schemas/memory.py

from datetime import datetime, timezone
from typing import Optional, Literal
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


class MemorySummary(BaseModel):
    """
    A compressed summary of one completed conversation session.
    This is what gets stored in Chroma as a long-term memory document.

    It is created by ConversationSummariser at session end and stored
    by VectorMemoryManager. It is retrieved at the start of new
    conversations to give the agent historical context about the user.

    Fields:
        user_id      : Which user this summary belongs to.
                       CRITICAL: used as the Chroma metadata filter.
                       Without this, users would see each other's data.

        session_id   : The original Redis session this summary came from.
                       Allows tracing a Chroma document back to its source.

        summary_text : The 3-5 sentence summary produced by the LLM.
                       This is the text that gets embedded into a vector.

        topics       : List of topic tags e.g. ["order_delay", "refund"].
                       Used in Phase 6 LangSmith logging and future filtering.

        resolution   : Whether the issue was resolved.
                       Allows filtering for unresolved issues in future.

        sentiment    : Customer's overall sentiment in the conversation.
                       Useful for escalation logic in Phase 3.

        created_at   : When this summary was created (session end time).

        chroma_id    : The ID assigned by Chroma after storing.
                       Populated after store_summary() returns.
    """

    user_id: str
    session_id: str
    summary_text: str
    topics: list[str] = Field(default_factory=list)
    resolution: Literal["resolved", "unresolved", "escalated", "unknown"] = "unknown"
    sentiment: Literal["positive", "neutral", "negative", "unknown"] = "unknown"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    chroma_id: Optional[str] = None

    def to_chroma_metadata(self) -> dict:
        """
        Convert to a flat dict for Chroma metadata storage.

        Chroma metadata values must be str, int, float, or bool.
        Lists are not supported — join topics as a comma-separated string.
        """
        return {
            "user_id":    self.user_id,
            "session_id": self.session_id,
            "topics":     ",".join(self.topics),
            "resolution": self.resolution,
            "sentiment":  self.sentiment,
            "created_at": self.created_at,
        }

    @classmethod
    def from_chroma_result(
        cls,
        document: str,
        metadata: dict,
        chroma_id: str,
    ) -> "MemorySummary":
        """
        Reconstruct a MemorySummary from a Chroma query result.
        Used when displaying retrieved memories for debugging.
        """
        topics_raw = metadata.get("topics", "")
        return cls(
            user_id=metadata.get("user_id", ""),
            session_id=metadata.get("session_id", ""),
            summary_text=document,
            topics=topics_raw.split(",") if topics_raw else [],
            resolution=metadata.get("resolution", "unknown"),
            sentiment=metadata.get("sentiment", "unknown"),
            created_at=metadata.get("created_at", ""),
            chroma_id=chroma_id,
        )