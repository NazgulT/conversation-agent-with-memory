# tests/unit/test_redis_memory.py

"""
Unit tests for RedisMemoryManager.

Uses fakeredis — no real Redis needed. Tests run anywhere, anytime.
Each test gets its own isolated fakeredis client via the fixture below.
"""

import time
import pytest
import fakeredis

from app.memory.redis_memory import RedisMemoryManager
from app.schemas.memory import ConversationTurn


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def memory() -> RedisMemoryManager:
    """
    Returns a RedisMemoryManager backed by an isolated fakeredis instance.
    Each test function gets a fresh, empty instance — no state leaks between tests.
    """
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    return RedisMemoryManager(client=fake_client)


# ── save_turn ─────────────────────────────────────────────────────────────────

def test_save_turn_returns_correct_turn(memory):
    """save_turn should return the saved turn with all fields populated."""
    turn = memory.save_turn(
        session_id="ses_001",
        user_id="usr_001",
        human_message="Where is my order?",
        assistant_message="Can you share your order number?",
        intent="ORDER",
    )

    assert turn.role_human == "Where is my order?"
    assert turn.role_assistant == "Can you share your order number?"
    assert turn.intent == "ORDER"
    assert turn.session_id == "ses_001"
    assert turn.user_id == "usr_001"
    assert turn.turn_index == 0  # first turn


def test_save_turn_increments_turn_index(memory):
    """Each saved turn should have a sequentially increasing turn_index."""
    for i in range(3):
        turn = memory.save_turn(
            session_id="ses_002",
            user_id="usr_002",
            human_message=f"Message {i}",
            assistant_message=f"Reply {i}",
        )
        assert turn.turn_index == i


def test_save_turn_stores_in_redis(memory):
    """Saved turn should be retrievable from Redis."""
    memory.save_turn(
        session_id="ses_003",
        user_id="usr_003",
        human_message="I want a refund.",
        assistant_message="I can help with that.",
    )

    turns = memory.get_history("ses_003")
    assert len(turns) == 1
    assert turns[0].role_human == "I want a refund."


# ── get_history ───────────────────────────────────────────────────────────────

def test_get_history_empty_session(memory):
    """Retrieving from a non-existent session returns an empty list, not an error."""
    turns = memory.get_history("nonexistent_session")
    assert turns == []


def test_get_history_returns_correct_count(memory):
    """get_history with n=3 returns at most 3 turns."""
    for i in range(5):
        memory.save_turn(
            session_id="ses_004",
            user_id="usr_004",
            human_message=f"Q{i}",
            assistant_message=f"A{i}",
        )

    turns = memory.get_history("ses_004", n=3)
    assert len(turns) == 3


def test_get_history_returns_most_recent_turns(memory):
    """When n < total turns, get_history returns the MOST RECENT ones."""
    messages = ["first", "second", "third", "fourth", "fifth"]
    for msg in messages:
        memory.save_turn(
            session_id="ses_005",
            user_id="usr_005",
            human_message=msg,
            assistant_message=f"reply to {msg}",
        )

    turns = memory.get_history("ses_005", n=2)
    assert len(turns) == 2
    assert turns[0].role_human == "fourth"
    assert turns[1].role_human == "fifth"


def test_get_history_chronological_order(memory):
    """Turns should be returned oldest-first (chronological order for LLM prompt)."""
    for msg in ["alpha", "beta", "gamma"]:
        memory.save_turn(
            session_id="ses_006",
            user_id="usr_006",
            human_message=msg,
            assistant_message="ok",
        )

    turns = memory.get_history("ses_006")
    contents = [t.role_human for t in turns]
    assert contents == ["alpha", "beta", "gamma"]


# ── Buffer window (the sliding window behaviour) ──────────────────────────────

def test_buffer_window_enforced(memory):
    """
    Saving more turns than buffer_memory_size should trim the oldest turns.
    This is the core buffer window behaviour.
    """
    buffer_size = memory._settings.buffer_memory_size  # 10 by default

    # Save more turns than the buffer size
    total_turns = buffer_size + 5
    for i in range(total_turns):
        memory.save_turn(
            session_id="ses_007",
            user_id="usr_007",
            human_message=f"message {i}",
            assistant_message=f"reply {i}",
        )

    # Should only have buffer_size turns left
    turns = memory.get_history("ses_007")
    assert len(turns) == buffer_size

    # The oldest turns should be gone, only the most recent remain
    assert turns[0].role_human == f"message {5}"   # first retained = index 5
    assert turns[-1].role_human == f"message {total_turns - 1}"


def test_buffer_window_does_not_affect_new_sessions(memory):
    """
    The buffer trim on one session should not affect other sessions.
    Redis keys are per-session — there should be no cross-session interference.
    """
    for i in range(15):
        memory.save_turn("ses_A", "usr_001", f"A{i}", f"replyA{i}")

    memory.save_turn("ses_B", "usr_002", "only message", "only reply")

    turns_b = memory.get_history("ses_B")
    assert len(turns_b) == 1
    assert turns_b[0].role_human == "only message"


# ── format_as_dialogue ────────────────────────────────────────────────────────

def test_format_as_dialogue_empty(memory):
    """Formatting an empty list should return an empty string."""
    result = memory.format_as_dialogue([])
    assert result == ""


def test_format_as_dialogue_structure(memory):
    """Formatted output should follow Human:/Assistant: pattern."""
    memory.save_turn("ses_008", "usr_008", "Hello", "Hi there!")
    memory.save_turn("ses_008", "usr_008", "My order is late", "I can help.")

    turns = memory.get_history("ses_008")
    text = memory.format_as_dialogue(turns)

    assert "Human: Hello" in text
    assert "Assistant: Hi there!" in text
    assert "Human: My order is late" in text
    assert "Assistant: I can help." in text


def test_format_as_dialogue_order(memory):
    """Dialogue should be in chronological order in the output string."""
    memory.save_turn("ses_009", "usr_009", "First question", "First answer")
    memory.save_turn("ses_009", "usr_009", "Second question", "Second answer")

    turns = memory.get_history("ses_009")
    text = memory.format_as_dialogue(turns)

    first_pos = text.index("First question")
    second_pos = text.index("Second question")
    assert first_pos < second_pos   # First appears before Second


def test_get_formatted_history_convenience(memory):
    """get_formatted_history should return the same result as get_history + format."""
    memory.save_turn("ses_010", "usr_010", "Test message", "Test reply")

    turns = memory.get_history("ses_010")
    manual = memory.format_as_dialogue(turns)
    convenience = memory.get_formatted_history("ses_010")

    assert manual == convenience


# ── clear_session ─────────────────────────────────────────────────────────────

def test_clear_session_removes_history(memory):
    """clear_session should remove all turns for that session."""
    memory.save_turn("ses_011", "usr_011", "Hello", "Hi")
    assert memory.session_exists("ses_011") is True

    memory.clear_session("ses_011")
    assert memory.session_exists("ses_011") is False
    assert memory.get_history("ses_011") == []


def test_clear_session_removes_metadata(memory):
    """clear_session should remove session metadata too."""
    memory.save_turn("ses_012", "usr_012", "Hello", "Hi")
    memory.clear_session("ses_012")

    meta = memory.get_session_meta("ses_012")
    assert meta is None


def test_clear_nonexistent_session_does_not_raise(memory):
    """Clearing a session that doesn't exist should be a no-op, not an error."""
    memory.clear_session("does_not_exist")  # should not raise


def test_clear_session_does_not_affect_other_sessions(memory):
    """Clearing one session should not touch other sessions."""
    memory.save_turn("ses_A2", "usr_001", "Hello A", "Hi A")
    memory.save_turn("ses_B2", "usr_002", "Hello B", "Hi B")

    memory.clear_session("ses_A2")

    assert memory.session_exists("ses_A2") is False
    assert memory.session_exists("ses_B2") is True


# ── session metadata ──────────────────────────────────────────────────────────

def test_session_meta_created_on_first_turn(memory):
    """Session metadata should be created when the first turn is saved."""
    assert memory.get_session_meta("ses_013") is None

    memory.save_turn("ses_013", "usr_013", "Hello", "Hi")

    meta = memory.get_session_meta("ses_013")
    assert meta is not None
    assert meta.session_id == "ses_013"
    assert meta.user_id == "usr_013"
    assert meta.turn_count == 1


def test_turn_count_increments(memory):
    """turn_count in metadata should increment with each saved turn."""
    for i in range(3):
        memory.save_turn("ses_014", "usr_014", f"Q{i}", f"A{i}")

    meta = memory.get_session_meta("ses_014")
    assert meta.turn_count == 3


# ── session_exists and get_turn_count ────────────────────────────────────────

def test_session_exists_false_before_first_turn(memory):
    assert memory.session_exists("brand_new_session") is False


def test_session_exists_true_after_first_turn(memory):
    memory.save_turn("ses_015", "usr_015", "Hi", "Hello")
    assert memory.session_exists("ses_015") is True


def test_get_turn_count(memory):
    for i in range(4):
        memory.save_turn("ses_016", "usr_016", f"Q{i}", f"A{i}")
    assert memory.get_turn_count("ses_016") == 4


# ── ping ─────────────────────────────────────────────────────────────────────

def test_ping_returns_true_with_fakeredis(memory):
    """ping() should return True when Redis (or fakeredis) is reachable."""
    assert memory.ping() is True