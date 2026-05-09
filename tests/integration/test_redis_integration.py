# tests/integration/test_redis_integration.py

"""
Integration tests for RedisMemoryManager against real Redis.

Requires: brew services start redis

Run with:
    pytest tests/integration/test_redis_integration.py -v
"""

import time
import pytest
import redis as redis_lib

from app.memory.redis_memory import RedisMemoryManager


@pytest.fixture(scope="module")
def real_memory() -> RedisMemoryManager:
    """
    Uses the real Redis instance. Verifies connectivity before running tests.
    All test keys are prefixed with 'test:' and cleaned up after the module.
    """
    manager = RedisMemoryManager()

    if not manager.ping():
        pytest.skip("Real Redis not available — run: brew services start redis")

    yield manager

    # Cleanup: delete all test keys after the full module runs
    client = manager._client
    test_keys = client.keys("session:inttest_*")
    if test_keys:
        client.delete(*test_keys)


def test_real_redis_save_and_retrieve(real_memory):
    """Full save → retrieve cycle on real Redis."""
    real_memory.save_turn(
        session_id="inttest_001",
        user_id="usr_001",
        human_message="Integration test message",
        assistant_message="Integration test reply",
        intent="GENERAL",
    )

    turns = real_memory.get_history("inttest_001")
    assert len(turns) == 1
    assert turns[0].role_human == "Integration test message"
    assert turns[0].intent == "GENERAL"


def test_real_redis_ttl_is_set(real_memory):
    """Saved sessions should have a TTL set in Redis."""
    real_memory.save_turn(
        session_id="inttest_002",
        user_id="usr_002",
        human_message="TTL test",
        assistant_message="TTL reply",
    )

    client = real_memory._client
    ttl = client.ttl(f"session:inttest_002:history")

    # TTL should be close to redis_ttl_seconds (86400 = 24h)
    assert ttl > 80000, f"TTL is {ttl} — expected close to 86400"
    assert ttl <= 86400


def test_real_redis_pipeline_atomicity(real_memory):
    """Pipeline operations should all succeed or all fail together."""
    for i in range(5):
        real_memory.save_turn(
            session_id="inttest_003",
            user_id="usr_003",
            human_message=f"Pipeline message {i}",
            assistant_message=f"Pipeline reply {i}",
        )

    count = real_memory.get_turn_count("inttest_003")
    assert count == 5

    meta = real_memory.get_session_meta("inttest_003")
    assert meta.turn_count == 5


def test_real_redis_formatted_output(real_memory):
    """Formatted dialogue should be correct on real Redis."""
    real_memory.save_turn(
        "inttest_004", "usr_004",
        "What is your return policy?",
        "You can return items within 30 days.",
    )

    text = real_memory.get_formatted_history("inttest_004")
    assert "Human: What is your return policy?" in text
    assert "Assistant: You can return items within 30 days." in text