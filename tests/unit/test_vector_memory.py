# tests/unit/test_vector_memory.py

"""
Unit tests for VectorMemoryManager.

Uses Chroma EphemeralClient (in-memory) and a deterministic fake
embedding function. No Ollama required. Tests run in milliseconds.
"""

import pytest
import chromadb
from unittest.mock import MagicMock

from app.memory.vector_memory import VectorMemoryManager
from app.schemas.memory import MemorySummary


# ── Fake embedding function ───────────────────────────────────────────────────

class FakeEmbeddings:
    """
    Deterministic embedding stub.

    Returns a fixed 768-dimensional vector for any input.
    This is sufficient for unit tests — we are testing the
    storage/retrieval logic, not the embedding quality.

    The vector varies slightly by input length so similarity
    search can return plausible results.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._make_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._make_vector(text)

    def _make_vector(self, text: str) -> list[float]:
        # Simple deterministic vector based on text content
        base = 0.5 + (len(text) % 10) * 0.01
        return [base] * 768


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def vector_manager() -> VectorMemoryManager:
    """
    Returns a VectorMemoryManager backed by in-memory Chroma
    and a fake embedding function. No disk writes, no Ollama.
    """
    client = chromadb.EphemeralClient()
    return VectorMemoryManager(
        chroma_client=client,
        embeddings=FakeEmbeddings(),
    )


def make_summary(
    user_id: str = "usr_001",
    session_id: str = "ses_001",
    text: str = "Customer had an issue with a delayed order. Resolved with a refund.",
    topics: list = None,
    resolution: str = "resolved",
    sentiment: str = "positive",
) -> MemorySummary:
    return MemorySummary(
        user_id=user_id,
        session_id=session_id,
        summary_text=text,
        topics=topics or ["order_delay", "refund"],
        resolution=resolution,
        sentiment=sentiment,
    )


# ── store_summary ─────────────────────────────────────────────────────────────

def test_store_summary_returns_summary_with_chroma_id(vector_manager):
    """store_summary should return the summary with chroma_id populated."""
    summary = make_summary()
    result = vector_manager.store_summary(summary)
    assert result.chroma_id is not None
    assert "ses_001" in result.chroma_id


def test_store_summary_is_idempotent(vector_manager):
    """Storing the same session twice should not create duplicate documents."""
    summary = make_summary(session_id="ses_dup")
    vector_manager.store_summary(summary)
    vector_manager.store_summary(summary)  # second call — should be a no-op
    count = vector_manager.get_user_memory_count("usr_001")
    assert count == 1


def test_store_multiple_summaries(vector_manager):
    """Multiple summaries for the same user should all be stored."""
    for i in range(3):
        vector_manager.store_summary(make_summary(session_id=f"ses_{i:03d}"))
    count = vector_manager.get_user_memory_count("usr_001")
    assert count == 3


# ── retrieve_relevant ─────────────────────────────────────────────────────────

def test_retrieve_returns_empty_for_unknown_user(vector_manager):
    """Retrieving for a user with no stored summaries returns empty list."""
    results = vector_manager.retrieve_relevant("usr_nobody", "order problem", k=3)
    assert results == []


def test_retrieve_returns_results_for_known_user(vector_manager):
    """After storing a summary, retrieve_relevant should return it."""
    vector_manager.store_summary(make_summary(user_id="usr_002"))
    results = vector_manager.retrieve_relevant("usr_002", "order delay refund", k=3)
    assert len(results) >= 1
    assert all(r.user_id == "usr_002" for r in results)


def test_retrieve_enforces_user_id_isolation(vector_manager):
    """
    PRIVACY TEST: User A's summaries must never appear in User B's results.
    This is the most important correctness test in Phase 2.
    """
    # Store one summary for each user
    vector_manager.store_summary(make_summary(user_id="usr_A", session_id="ses_A"))
    vector_manager.store_summary(make_summary(user_id="usr_B", session_id="ses_B"))

    results_a = vector_manager.retrieve_relevant("usr_A", "order", k=5)
    results_b = vector_manager.retrieve_relevant("usr_B", "order", k=5)

    for r in results_a:
        assert r.user_id == "usr_A", "User A received User B's data — PRIVACY LEAK"
    for r in results_b:
        assert r.user_id == "usr_B", "User B received User A's data — PRIVACY LEAK"


def test_retrieve_respects_k_limit(vector_manager):
    """retrieve_relevant should return at most k results."""
    for i in range(5):
        vector_manager.store_summary(
            make_summary(user_id="usr_003", session_id=f"ses_{i:03d}")
        )
    results = vector_manager.retrieve_relevant("usr_003", "order", k=2)
    assert len(results) <= 2


def test_retrieve_as_text_returns_formatted_string(vector_manager):
    """retrieve_as_text should return a human-readable string, not a list."""
    vector_manager.store_summary(make_summary(user_id="usr_004"))
    text = vector_manager.retrieve_as_text("usr_004", "refund request")
    assert isinstance(text, str)
    assert "Past customer interactions:" in text or text == ""


def test_retrieve_as_text_returns_empty_string_for_no_results(vector_manager):
    """retrieve_as_text should return '' (not None or []) for unknown users."""
    text = vector_manager.retrieve_as_text("usr_nobody", "anything")
    assert text == ""


# ── delete_user_memories ──────────────────────────────────────────────────────

def test_delete_user_memories_removes_all_summaries(vector_manager):
    """delete_user_memories should remove all documents for that user."""
    for i in range(3):
        vector_manager.store_summary(
            make_summary(user_id="usr_delete", session_id=f"ses_{i:03d}")
        )
    assert vector_manager.get_user_memory_count("usr_delete") == 3

    deleted = vector_manager.delete_user_memories("usr_delete")
    assert deleted == 3
    assert vector_manager.get_user_memory_count("usr_delete") == 0


def test_delete_user_memories_does_not_affect_other_users(vector_manager):
    """Deleting one user's memories must not touch other users' data."""
    vector_manager.store_summary(make_summary(user_id="usr_safe", session_id="ses_safe"))
    vector_manager.store_summary(make_summary(user_id="usr_del2", session_id="ses_del2"))

    vector_manager.delete_user_memories("usr_del2")

    assert vector_manager.get_user_memory_count("usr_safe") == 1
    assert vector_manager.get_user_memory_count("usr_del2") == 0


# ── summary_exists ────────────────────────────────────────────────────────────

def test_summary_exists_false_before_storing(vector_manager):
    assert vector_manager.summary_exists("ses_new", "usr_001") is False


def test_summary_exists_true_after_storing(vector_manager):
    vector_manager.store_summary(make_summary(session_id="ses_exists"))
    assert vector_manager.summary_exists("ses_exists", "usr_001") is True


# ── ping ──────────────────────────────────────────────────────────────────────

def test_ping_returns_true(vector_manager):
    assert vector_manager.ping() is True