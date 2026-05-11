# app/memory/summariser.py

"""
ConversationSummariser

Compresses a list of ConversationTurns (from Redis) into a structured
summary suitable for long-term vector storage.

Uses llama3.2:3b via Ollama — local, offline, no API cost.

The summariser is intentionally separate from VectorMemoryManager because:
  1. Testability — LLM calls can be mocked without touching Chroma
  2. Single responsibility — one class compresses, one class stores
  3. Replaceability — you can swap the summary strategy without
     touching the storage layer (e.g. switch to extractive summarisation)
"""

import json
import logging
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.provider import get_llm
from app.schemas.memory import ConversationTurn, MemorySummary

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = """You are a customer support data analyst.
Your job is to summarise completed support conversations for long-term storage.

Given a conversation transcript, produce a structured JSON summary.

Rules:
- summary_text: 2-4 sentences. Include what the customer's issue was,
  what actions were taken, and the final outcome. Be specific.
- topics: list of 1-4 lowercase topic tags from this list only:
  order_delay, refund, billing, account, login, technical, shipping,
  product, return, cancellation, complaint, general
- resolution: "resolved" if the issue was fixed, "unresolved" if not,
  "escalated" if handed to a human, "unknown" if unclear
- sentiment: "positive" if the customer was satisfied or calm,
  "negative" if frustrated or angry, "neutral" if neither, "unknown" if unclear

Respond with ONLY valid JSON. No explanation. No markdown. No preamble.

JSON format:
{
  "summary_text": "...",
  "topics": ["...", "..."],
  "resolution": "resolved",
  "sentiment": "neutral"
}"""


SUMMARY_USER_TEMPLATE = """Summarise this customer support conversation:

{transcript}

JSON summary:"""


# ── ConversationSummariser ────────────────────────────────────────────────────

class ConversationSummariser:
    """
    Produces structured summaries of completed support conversations.

    Usage:
        summariser = ConversationSummariser()
        summary = summariser.summarise(
            turns=turns,
            user_id="usr_001",
            session_id="ses_abc",
        )
        # summary is a fully populated MemorySummary ready for Chroma storage
    """

    def __init__(self, llm=None):
        """
        Accepts an optional LLM instance for dependency injection.
        If None, uses get_llm() from provider.py (llama3.2:3b via Ollama).
        The injectable parameter makes unit testing clean — pass a mock LLM
        and the tests never touch real Ollama.
        """
        self._llm = llm or get_llm()

    def _build_transcript(self, turns: list[ConversationTurn]) -> str:
        """
        Convert ConversationTurns to a numbered transcript string.

        Format:
            [1] Human: My order hasn't arrived.
                Assistant: I'm sorry. Can you share your order number?
            [2] Human: It's #12345.
                Assistant: Found it. Your package is in transit.
        """
        lines = []
        for turn in turns:
            lines.append(f"[{turn.turn_index + 1}] Human: {turn.role_human.strip()}")
            if turn.role_assistant.strip():
                lines.append(f"     Assistant: {turn.role_assistant.strip()}")
        return "\n".join(lines)

    def _parse_llm_response(self, raw: str) -> dict:
        """
        Parse the LLM's JSON response into a dict.

        llama3.2:3b sometimes wraps JSON in markdown fences (```json ... ```)
        even when told not to. This method strips fences before parsing.

        If parsing fails entirely, returns a safe fallback dict so the
        application does not crash — a degraded summary is better than
        a crashed session-end flow.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        cleaned = cleaned.rstrip("`").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from within the string
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            logger.warning(
                "Could not parse LLM summary response as JSON. "
                "Raw response: %s", raw[:200]
            )
            return {
                "summary_text": "Conversation summary unavailable due to parsing error.",
                "topics": ["general"],
                "resolution": "unknown",
                "sentiment": "unknown",
            }

    def _validate_parsed(self, parsed: dict) -> dict:
        """
        Validate and sanitise the parsed JSON from the LLM.

        The LLM might return topics not in the allowed list, or an
        invalid resolution value. This method enforces the schema.
        """
        allowed_topics = {
            "order_delay", "refund", "billing", "account", "login",
            "technical", "shipping", "product", "return",
            "cancellation", "complaint", "general"
        }
        allowed_resolutions = {"resolved", "unresolved", "escalated", "unknown"}
        allowed_sentiments  = {"positive", "neutral", "negative", "unknown"}

        # Topics: filter to allowed set, default to ["general"]
        raw_topics = parsed.get("topics", [])
        if isinstance(raw_topics, str):
            raw_topics = [raw_topics]
        valid_topics = [t.lower().strip() for t in raw_topics
                        if t.lower().strip() in allowed_topics]
        if not valid_topics:
            valid_topics = ["general"]

        resolution = parsed.get("resolution", "unknown")
        if resolution not in allowed_resolutions:
            resolution = "unknown"

        sentiment = parsed.get("sentiment", "unknown")
        if sentiment not in allowed_sentiments:
            sentiment = "unknown"

        summary_text = str(parsed.get("summary_text", "")).strip()
        if not summary_text:
            summary_text = "No summary could be generated."

        return {
            "summary_text": summary_text,
            "topics":       valid_topics,
            "resolution":   resolution,
            "sentiment":    sentiment,
        }

    def summarise(
        self,
        turns: list[ConversationTurn],
        user_id: str,
        session_id: str,
    ) -> Optional[MemorySummary]:
        """
        Produce a MemorySummary from a list of ConversationTurns.

        Returns None if:
        - turns list is empty (nothing to summarise)
        - turns list has fewer than 2 turns (not enough for a meaningful summary)

        The returned MemorySummary is not yet stored in Chroma.
        Pass it to VectorMemoryManager.store_summary() to persist it.

        Args:
            turns      : List of ConversationTurns from RedisMemoryManager.get_history()
            user_id    : The user ID to attach to the summary (for Chroma filtering)
            session_id : The session ID to attach (for traceability)
        """
        if not turns or len(turns) < 2:
            logger.info(
                "Session %s has fewer than 2 turns — skipping summarisation.",
                session_id
            )
            return None

        transcript = self._build_transcript(turns)

        messages = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=SUMMARY_USER_TEMPLATE.format(
                transcript=transcript
            )),
        ]

        logger.debug("Summarising session %s (%d turns)...", session_id, len(turns))

        try:
            response = self._llm.invoke(messages)
            raw = response.content.strip()
        except Exception as e:
            logger.error("LLM summarisation failed for session %s: %s", session_id, e)
            # Return a minimal fallback summary rather than crashing
            return MemorySummary(
                user_id=user_id,
                session_id=session_id,
                summary_text=f"Conversation with {len(turns)} turns. Summary unavailable.",
                topics=["general"],
                resolution="unknown",
                sentiment="unknown",
            )

        parsed   = self._parse_llm_response(raw)
        validated = self._validate_parsed(parsed)

        summary = MemorySummary(
            user_id=user_id,
            session_id=session_id,
            summary_text=validated["summary_text"],
            topics=validated["topics"],
            resolution=validated["resolution"],
            sentiment=validated["sentiment"],
        )

        logger.debug(
            "Summary created for session %s: topics=%s, resolution=%s, sentiment=%s",
            session_id,
            summary.topics,
            summary.resolution,
            summary.sentiment,
        )

        return summary