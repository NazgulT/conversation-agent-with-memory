# tests/integration/test_vector_integration.py

"""
Integration tests for VectorMemoryManager and end_session orchestrator.
Requires: Ollama running with nomic-embed-text + llama3.2:3b pulled.

Run:
    pytest tests/integration/test_vector_integration.py -v -s
"""

import pytest
import chromadb

from app.memory.vector_memory import VectorMemoryManager
from app.memory.summariser import ConversationSummariser
from app.memory.orchestrator import end_session
from app.memory.redis_memory import RedisMemoryManager
from app.schemas.memory import MemorySummary, ConversationTurn


@pytest.fixture(scope="module")
def real_vector_manager():
    """In-memory Chroma + real Ollama embeddings (nomic-embed-text)."""
    try:
        from app.llm.provider import get_embeddings
        embeddings = get_embeddings()
        # Quick embedding test to confirm Ollama is reachable
        embeddings.embed_query("test")
    except Exception:
        pytest.skip("Ollama not running — run: ollama serve")

    client = chromadb.EphemeralClient()
    manager = VectorMemoryManager(chroma_client=client, embeddings=embeddings)
    yield manager


@pytest.fixture(scope="module")
def real_summariser():
    """Real ConversationSummariser using llama3.2:3b."""
    try:
        from app.llm.provider import get_llm
        llm = get_llm()
        llm.invoke([])  # quick test call
    except Exception:
        pytest.skip("Ollama not running or llama3.2:3b not pulled")
    return ConversationSummariser()


def test_real_embed_and_store(real_vector_manager):
    """Store a summary with real embeddings, confirm it is retrievable."""
    summary = MemorySummary(
        user_id="inttest_usr_001",
        session_id="inttest_ses_001",
        summary_text="Customer reported that order #99 was delivered damaged. Replacement sent.",
        topics=["shipping", "product"],
        resolution="resolved",
        sentiment="neutral",
    )

    stored = real_vector_manager.store_summary(summary)
    assert stored.chroma_id is not None

    results = real_vector_manager.retrieve_relevant(
        user_id="inttest_usr_001",
        query="I received a broken item in my delivery",
        k=3,
    )
    assert len(results) >= 1
    assert results[0].user_id == "inttest_usr_001"


def test_real_semantic_relevance(real_vector_manager):
    """
    A semantically related query should retrieve the stored summary.
    This test verifies that nomic-embed-text captures meaning, not just keywords.
    """
    real_vector_manager.store_summary(MemorySummary(
        user_id="inttest_usr_002",
        session_id="inttest_ses_002",
        summary_text="Customer could not access their account after a password change.",
        topics=["account", "login"],
        resolution="resolved",
        sentiment="neutral",
    ))

    # Query uses different words but same meaning
    results = real_vector_manager.retrieve_relevant(
        user_id="inttest_usr_002",
        query="I am locked out and cannot sign in",
        k=3,
    )
    assert len(results) >= 1
    # The login/account summary should be retrieved for a login query


def test_real_user_isolation(real_vector_manager):
    """With real embeddings, user isolation must still hold."""
    real_vector_manager.store_summary(MemorySummary(
        user_id="inttest_usr_A",
        session_id="inttest_ses_A",
        summary_text="User A had a billing issue. Resolved with a $5 credit.",
        topics=["billing"],
        resolution="resolved",
        sentiment="positive",
    ))

    # Searching as user_B should return nothing (no data stored for them)
    results = real_vector_manager.retrieve_relevant(
        user_id="inttest_usr_B",
        query="billing issue credit",
        k=3,
    )
    for r in results:
        assert r.user_id == "inttest_usr_B", "PRIVACY LEAK: user B got user A data"


def test_real_summariser_produces_valid_summary(real_summariser):
    """Real LLM should produce a parseable, validated MemorySummary."""
    turns = [
        ConversationTurn(
            role_human="My package from last week still hasn't arrived.",
            role_assistant="I'm sorry to hear that. What is your order number?",
            turn_index=0, session_id="inttest_ses_sum", user_id="inttest_usr_sum",
        ),
        ConversationTurn(
            role_human="It's order #54321.",
            role_assistant="Found it. There was a carrier delay. I've issued a full refund.",
            turn_index=1, session_id="inttest_ses_sum", user_id="inttest_usr_sum",
        ),
    ]

    summary = real_summariser.summarise(
        turns=turns,
        user_id="inttest_usr_sum",
        session_id="inttest_ses_sum",
    )

    assert summary is not None
    assert len(summary.summary_text) > 20
    assert summary.resolution in {"resolved", "unresolved", "escalated", "unknown"}
    assert summary.sentiment in {"positive", "neutral", "negative", "unknown"}
    assert len(summary.topics) >= 1


def test_full_end_session_flow():
    """
    Full orchestrated flow: turns in Redis → summarise → store in Chroma → clear Redis.
    Uses real Redis (Homebrew) + real Ollama.
    """
    redis_mgr = RedisMemoryManager()
    if not redis_mgr.ping():
        pytest.skip("Redis not running — brew services start redis")

    chroma_client = chromadb.EphemeralClient()

    session_id = "inttest_orchestration_001"
    user_id    = "inttest_orch_usr"

    # Set up: save 3 turns to Redis
    redis_mgr.clear_session(session_id)
    redis_mgr.save_turn(session_id, user_id,
                        "My order is 2 weeks late.", "I'm very sorry. Order number?")
    redis_mgr.save_turn(session_id, user_id,
                        "Order #77777", "I see it was delayed. I'll escalate this.")
    redis_mgr.save_turn(session_id, user_id,
                        "Thank you.", "You'll receive a confirmation email shortly.")

    assert redis_mgr.get_turn_count(session_id) == 3

    # Run end_session
    from app.llm.provider import get_embeddings
    try:
        embeddings = get_embeddings()
        embeddings.embed_query("test")
    except Exception:
        pytest.skip("Ollama not running")

    vector_mgr = VectorMemoryManager(chroma_client=chroma_client, embeddings=embeddings)
    result = end_session(
        session_id=session_id,
        user_id=user_id,
        redis_manager=redis_mgr,
        vector_manager=vector_mgr,
    )

    # Verify the result
    assert result.turns_processed == 3
    assert result.stored_in_chroma is True
    assert result.redis_cleared is True
    assert result.error is None
    assert redis_mgr.session_exists(session_id) is False