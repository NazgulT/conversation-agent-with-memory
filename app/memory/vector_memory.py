# app/memory/vector_memory.py

"""
VectorMemoryManager

Manages long-term conversation memory using Chroma as the vector store
and nomic-embed-text (via Ollama) as the embedding model.

All data is scoped by user_id at both write and read time:
  - store_summary()      : attaches user_id to every document's metadata
  - retrieve_relevant()  : filters search results to current user only

This two-layer isolation (metadata filter + application logic) ensures
no user ever receives another user's conversation history.

Storage:
  Location : ./chroma_db/ (configured via CHROMA_PERSIST_DIRECTORY)
  Collection: customer-memory (configured via CHROMA_COLLECTION_NAME)
  One collection holds all users — filtered at query time by user_id.
"""

import logging
import uuid
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.llm.provider import get_embeddings
from app.schemas.memory import MemorySummary

logger = logging.getLogger(__name__)


class VectorMemoryManager:
    """
    Handles all Chroma read/write operations for long-term memory.

    Usage:
        manager = VectorMemoryManager()

        # Store a summary at session end
        summary = manager.store_summary(memory_summary)

        # Retrieve relevant past memories at conversation start
        context_str = manager.retrieve_as_text(
            user_id="usr_001",
            query="I have a problem with my delivery",
        )
    """

    def __init__(
        self,
        chroma_client: Optional[chromadb.ClientAPI] = None,
        embeddings=None,
    ):
        """
        Both dependencies are injectable for testing.

        In production: uses PersistentClient (writes to ./chroma_db/)
        In unit tests: caller passes an EphemeralClient (in-memory only)

        Embedding model defaults to get_embeddings() from provider.py
        (nomic-embed-text via Ollama). Unit tests pass a mock embedder.
        """
        self._settings = get_settings()

        # Build or accept the Chroma client
        if chroma_client is not None:
            self._chroma_client = chroma_client
        else:
            self._chroma_client = chromadb.PersistentClient(
                path=self._settings.chroma_persist_directory
            )

        # Build or accept the embedding function
        self._embeddings = embeddings or get_embeddings()

        # The LangChain Chroma wrapper handles embedding + collection management
        self._store = Chroma(
            client=self._chroma_client,
            collection_name=self._settings.chroma_collection_name,
            embedding_function=self._embeddings,
        )

        logger.debug(
            "VectorMemoryManager initialised: collection=%s",
            self._settings.chroma_collection_name,
        )

    # ── Write operations ──────────────────────────────────────────────────────

    def store_summary(self, summary: MemorySummary) -> MemorySummary:
        """
        Embed a MemorySummary and store it in Chroma.

        The summary_text is embedded by nomic-embed-text into a 768-dim
        vector. The vector and the metadata are stored together in Chroma.

        A unique Chroma document ID is generated using user_id and session_id.
        If a summary for this session is already stored, this method returns
        immediately without writing — idempotent even if Chroma client
        behaviour changes between versions.

        Returns the input MemorySummary with chroma_id populated.
        """
        doc_id = f"summary_{summary.user_id}_{summary.session_id}"

        if self.summary_exists(summary.session_id, summary.user_id):
            summary.chroma_id = doc_id
            logger.debug(
                "Summary already stored for user=%s session=%s — skipping",
                summary.user_id,
                summary.session_id,
            )
            return summary

        document = Document(
            page_content=summary.summary_text,
            metadata=summary.to_chroma_metadata(),
        )

        self._store.add_documents(
            documents=[document],
            ids=[doc_id],
        )

        summary.chroma_id = doc_id

        logger.debug(
            "Stored summary for user=%s session=%s (id=%s)",
            summary.user_id,
            summary.session_id,
            doc_id,
        )

        return summary

    # ── Read operations ───────────────────────────────────────────────────────

    def retrieve_relevant(
        self,
        user_id: str,
        query: str,
        k: Optional[int] = None,
    ) -> list[MemorySummary]:
        """
        Retrieve the K most relevant past conversation summaries for a user.

        The query is embedded by nomic-embed-text into a vector. Chroma
        returns the K stored documents whose vectors are closest to the
        query vector — filtered strictly to the given user_id.

        If fewer than K documents exist for this user, all of them are
        returned. If none exist, returns an empty list (not an error).

        Args:
            user_id : Current user. ONLY their documents are returned.
            query   : The current user message. Used as the search query.
            k       : Number of results to return. Defaults to config value.

        Returns:
            List of MemorySummary objects, most relevant first.
        """
        k = k or self._settings.long_term_retrieval_k

        try:
            results = self._store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter={"user_id": user_id},
            )
        except Exception as e:
            logger.warning(
                "Vector retrieval failed for user=%s: %s", user_id, e
            )
            return []

        if not results:
            return []

        summaries = []
        for document, score in results:
            # Only include results with meaningful similarity
            # Scores below 0.3 are likely irrelevant (Chroma cosine similarity)
            if score < 0.3:
                logger.debug(
                    "Skipping low-relevance result (score=%.2f): %s",
                    score,
                    document.page_content[:60],
                )
                continue

            summary = MemorySummary.from_chroma_result(
                document=document.page_content,
                metadata=document.metadata,
                chroma_id=document.metadata.get("session_id", ""),
            )
            summaries.append(summary)

        logger.debug(
            "Retrieved %d relevant memories for user=%s (query: '%s...')",
            len(summaries),
            user_id,
            query[:40],
        )

        return summaries

    def retrieve_as_text(
        self,
        user_id: str,
        query: str,
        k: Optional[int] = None,
    ) -> str:
        """
        Convenience method: retrieve relevant summaries and format as text.

        This is what the retrieve_long_term node in Phase 4 will call:
            context = vector_manager.retrieve_as_text(user_id, message)

        Returns an empty string if no relevant memories exist.
        Returns a formatted string like:

            Past customer interactions:
            - [2024-01-10] Jane's order #A123 arrived late. Refund issued.
            - [2024-01-05] Billing discrepancy resolved. $15 reversed.
        """
        summaries = self.retrieve_relevant(user_id, query, k=k)

        if not summaries:
            return ""

        lines = ["Past customer interactions:"]
        for s in summaries:
            date = s.created_at[:10] if s.created_at else "unknown date"
            lines.append(f"- [{date}] {s.summary_text}")

        return "\n".join(lines)

    # ── Management operations ─────────────────────────────────────────────────

    def get_user_memory_count(self, user_id: str) -> int:
        """
        Return the number of summaries stored for a user.
        Used by the /health and /sessions endpoints in Phase 5.
        """
        try:
            collection = self._chroma_client.get_collection(
                self._settings.chroma_collection_name
            )
            results = collection.get(where={"user_id": user_id})
            ids = results.get("ids") or []
            return len(ids)
        except Exception:
            return 0

    def delete_user_memories(self, user_id: str) -> int:
        """
        Delete ALL stored summaries for a user.

        This is the GDPR/right-to-erasure operation. It permanently removes
        all long-term memory for a user. Short-term Redis data must be
        cleared separately via RedisMemoryManager.clear_session().

        Returns the number of documents deleted.
        """
        try:
            collection = self._chroma_client.get_collection(
                self._settings.chroma_collection_name
            )
            results = collection.get(
                where={"user_id": user_id},
                include=["documents"],
            )
            ids_to_delete = results.get("ids", [])

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(
                    "Deleted %d memories for user=%s",
                    len(ids_to_delete),
                    user_id,
                )

            return len(ids_to_delete)

        except Exception as e:
            logger.error("Failed to delete memories for user=%s: %s", user_id, e)
            return 0

    def summary_exists(self, session_id: str, user_id: str) -> bool:
        """
        Check if a summary for a given session has already been stored.
        Prevents double-storing the same session if session end is called twice.
        """
        doc_id = f"summary_{user_id}_{session_id}"
        try:
            collection = self._chroma_client.get_collection(
                self._settings.chroma_collection_name
            )
            result = collection.get(ids=[doc_id])
            return len(result.get("ids", [])) > 0
        except Exception:
            return False

    def ping(self) -> bool:
        """
        Check if Chroma is operational.
        Since Chroma is in-process, this checks the collection is accessible.
        """
        try:
            self._chroma_client.list_collections()
            return True
        except Exception:
            return False