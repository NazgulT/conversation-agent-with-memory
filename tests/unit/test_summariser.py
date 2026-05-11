# tests/unit/test_summariser.py

"""
Unit tests for ConversationSummariser.

The LLM is mocked in all tests — no Ollama needed.
Tests verify parsing, validation, and edge cases.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from app.memory.summariser import ConversationSummariser
from app.schemas.memory import ConversationTurn


def make_turns(count: int) -> list[ConversationTurn]:
    """Helper: generate N ConversationTurns for testing."""
    turns = []
    for i in range(count):
        turns.append(ConversationTurn(
            role_human=f"Human message {i}",
            role_assistant=f"Assistant reply {i}",
            turn_index=i,
            session_id="ses_test",
            user_id="usr_test",
        ))
    return turns


def make_mock_llm(json_response: str):
    """Helper: build a mock LLM that returns a fixed JSON string."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content=json_response)
    return mock


VALID_JSON = '''{
    "summary_text": "Customer inquired about a delayed order #12345. Issue traced to warehouse delay. Refund of $12.99 was issued. Customer expressed satisfaction.",
    "topics": ["order_delay", "refund"],
    "resolution": "resolved",
    "sentiment": "positive"
}'''


def test_summarise_returns_memory_summary():
    """summarise() should return a populated MemorySummary."""
    summariser = ConversationSummariser(llm=make_mock_llm(VALID_JSON))
    turns = make_turns(3)
    result = summariser.summarise(turns, user_id="usr_001", session_id="ses_001")

    assert result is not None
    assert result.user_id == "usr_001"
    assert result.session_id == "ses_001"
    assert "delayed order" in result.summary_text
    assert "order_delay" in result.topics
    assert "refund" in result.topics
    assert result.resolution == "resolved"
    assert result.sentiment == "positive"


def test_summarise_returns_none_for_empty_turns():
    """summarise() returns None when no turns are provided."""
    summariser = ConversationSummariser(llm=make_mock_llm(VALID_JSON))
    result = summariser.summarise([], user_id="usr_001", session_id="ses_001")
    assert result is None


def test_summarise_returns_none_for_single_turn():
    """summarise() returns None for fewer than 2 turns — not meaningful."""
    summariser = ConversationSummariser(llm=make_mock_llm(VALID_JSON))
    result = summariser.summarise(make_turns(1), "usr_001", "ses_001")
    assert result is None


def test_summarise_strips_markdown_fences():
    """LLM sometimes wraps JSON in ```json ... ```. Should still parse correctly."""
    fenced_json = f"```json\n{VALID_JSON}\n```"
    summariser = ConversationSummariser(llm=make_mock_llm(fenced_json))
    result = summariser.summarise(make_turns(3), "usr_001", "ses_001")
    assert result is not None
    assert result.resolution == "resolved"


def test_summarise_filters_invalid_topics():
    """Topics not in the allowed set should be dropped, defaulting to ['general']."""
    bad_json = '''{
        "summary_text": "Test summary.",
        "topics": ["pizza", "weather", "not_a_topic"],
        "resolution": "resolved",
        "sentiment": "neutral"
    }'''
    summariser = ConversationSummariser(llm=make_mock_llm(bad_json))
    result = summariser.summarise(make_turns(3), "usr_001", "ses_001")
    assert result.topics == ["general"]


def test_summarise_handles_invalid_resolution():
    """An invalid resolution value should be replaced with 'unknown'."""
    bad_json = '''{
        "summary_text": "Test.",
        "topics": ["general"],
        "resolution": "partially_fixed",
        "sentiment": "neutral"
    }'''
    summariser = ConversationSummariser(llm=make_mock_llm(bad_json))
    result = summariser.summarise(make_turns(3), "usr_001", "ses_001")
    assert result.resolution == "unknown"


def test_summarise_handles_llm_failure_gracefully():
    """If the LLM raises an exception, return a fallback MemorySummary, not a crash."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Ollama timeout")

    summariser = ConversationSummariser(llm=mock_llm)
    result = summariser.summarise(make_turns(4), "usr_001", "ses_001")

    # Should return a summary, not raise an exception
    assert result is not None
    assert result.user_id == "usr_001"
    assert "unavailable" in result.summary_text.lower()


def test_summarise_handles_completely_invalid_json():
    """If LLM returns unparseable text, return a fallback, not a crash."""
    summariser = ConversationSummariser(llm=make_mock_llm("This is not JSON at all."))
    result = summariser.summarise(make_turns(3), "usr_001", "ses_001")
    assert result is not None
    assert result.topics == ["general"]
    assert result.resolution == "unknown"