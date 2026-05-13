# app/llm/provider.py

"""
LLM Provider abstraction — Ollama (local) and HuggingFace (cloud) backends.

All agent code imports exclusively from this module:
    from app.llm.provider import get_llm, get_classifier_llm, get_embeddings

────────────────────────────────────────────────────────────────────────────────
OLLAMA MODEL DECISIONS (documented here as the authoritative record)
────────────────────────────────────────────────────────────────────────────────

Main chatbot — llama3.2:3b
  Why: Meta's Llama 3.2 3B is the best open-weight conversational model under
  4B params as of late 2024. It was specifically fine-tuned for instruction
  following and multi-turn dialogue — the exact use case of a support agent.
  Its 128K context window means build_context can pass full conversation history
  without truncation in most real sessions. Temperature 0.7 gives natural,
  varied replies without hallucinating.

Classifier — gemma2:2b
  Why: Google's Gemma 2 2B outperforms Phi-3-mini on structured output tasks
  despite being smaller (2B vs 3.8B). At temperature=0 it reliably returns
  exactly one word with no preamble. This is the critical property for the
  guardrail node — a classifier that adds "Certainly! The category is ORDER."
  breaks the downstream routing logic. Gemma 2 2B does not do this.
  It also uses ~800MB less RAM than phi3:mini, relevant on 8GB Macs.

Embeddings — nomic-embed-text
  Why: Produced by Nomic AI, this model tops the MTEB embedding leaderboard
  for open-source models at its size. Its 768-dimensional output gives Chroma
  richer semantic representations than MiniLM's 384-dim, improving long-term
  memory retrieval quality in Phases 2–6.

────────────────────────────────────────────────────────────────────────────────
"""

from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from app.config import get_settings


# ── Main Chatbot LLM ──────────────────────────────────────────────────────────

@lru_cache
def get_llm() -> BaseChatModel:
    """
    Returns the main chat LLM.

    Ollama: llama3.2:3b
      - 128K context window
      - temperature=0.7 for natural, varied support replies
      - 30s timeout: conservative for Intel CPU (8-15 tok/s)
        A 100-token response takes ~8s on Intel; 30s gives safe headroom.

    HuggingFace: Mistral-7B-Instruct-v0.3
      - Cloud inference, no local GPU needed
      - Same LangChain interface, different backend
    """
    s = get_settings()

    if s.using_ollama:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=s.ollama_main_model,          # llama3.2:3b
            base_url=s.ollama_base_url,
            temperature=0.7,
            timeout=s.llm_timeout_seconds,      # 30s
            num_ctx=4096,     # context window passed to each request.
                              # 4096 is enough for Phase 0-3; raise to 8192
                              # in Phase 4 once full conversation + memory
                              # context is assembled by build_context.
        )

    if s.using_huggingface:
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        endpoint = HuggingFaceEndpoint(
            repo_id=s.hf_main_model,
            huggingfacehub_api_token=s.hf_api_token,
            task="text-generation",
            temperature=0.7,
            max_new_tokens=512,
            timeout=s.llm_timeout_seconds,
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider}")


# ── Classifier LLM ────────────────────────────────────────────────────────────

@lru_cache
def get_classifier_llm() -> BaseChatModel:
    """
    Returns the fast classification LLM.

    Ollama: gemma2:2b
      - temperature=0: fully deterministic, no randomness in labels
      - num_predict=5: hard cap at 5 output tokens. Classification needs
        one word (e.g. "ORDER"). Capping tokens prevents the model from
        ever generating an explanation even if it wants to.
      - 8s timeout: a 2B model returning 1 token should respond in < 2s
        on Apple Silicon and < 5s on Intel. 8s is a hard ceiling — if
        classification takes longer than this, something is wrong.
      - repeat_penalty=1.0: disabled. With a 1-token output, repetition
        penalties are irrelevant and disabling them removes a processing step.

    HuggingFace: google/flan-t5-large
      - Encoder-decoder architecture, excellent at classification
      - Uses text2text-generation task type (different from chat models)
    """
    s = get_settings()

    if s.using_ollama:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=s.ollama_classifier_model,    # gemma2:2b
            base_url=s.ollama_base_url,
            temperature=0,
            timeout=s.classifier_timeout_seconds,   # 8s hard limit
            num_predict=5,      # classifier only ever needs 1-2 tokens
            num_ctx=512,        # classification prompt is short; small
                                # context window = faster inference
            repeat_penalty=1.0, # disabled — irrelevant for 1-token output
        )

    if s.using_huggingface:
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        endpoint = HuggingFaceEndpoint(
            repo_id=s.hf_classifier_model,
            huggingfacehub_api_token=s.hf_api_token,
            task="text2text-generation",
            temperature=0.0,
            max_new_tokens=10,
            timeout=s.classifier_timeout_seconds,
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider}")


# ── Embedding Model ───────────────────────────────────────────────────────────

@lru_cache
def get_embeddings() -> Embeddings:
    """
    Returns the embedding model.

    Ollama: nomic-embed-text
      - 768-dim vectors (vs MiniLM's 384 — richer semantic representations)
      - Tops MTEB open-source embedding leaderboard at this size
      - Runs through Ollama server — no separate service needed

    HuggingFace: sentence-transformers/all-MiniLM-L6-v2
      - Always runs locally even in HF mode — small enough for Mac CPU
      - No API call, no rate limits, works offline
      - Downloads once to ~/.cache/huggingface/
    """
    s = get_settings()

    if s.using_ollama:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=s.ollama_embedding_model,     # nomic-embed-text
            base_url=s.ollama_base_url,
        )

    if s.using_huggingface:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=s.hf_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider}")


# ── Safety / Guard LLM ───────────────────────────────────────────────────────

@lru_cache
def get_guard_llm() -> BaseChatModel:
    """
    Returns the Llama Guard 3 safety classification LLM.

    Ollama: llama-guard3:1b
      - temperature=0: deterministic safe/unsafe decisions
      - num_predict=20: guard output is "safe" or "unsafe\nS{n}" — short
      - 10s timeout: safety gate must be fast; failure closes the message

    HuggingFace: falls back to the classifier model for basic safety heuristics.
    """
    s = get_settings()

    if s.using_ollama:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=s.ollama_guard_model,
            base_url=s.ollama_base_url,
            temperature=0,
            timeout=10,
            num_predict=20,
            num_ctx=512,
        )

    if s.using_huggingface:
        # HuggingFace has no llama-guard equivalent at this size; reuse classifier
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        endpoint = HuggingFaceEndpoint(
            repo_id=s.hf_classifier_model,
            huggingfacehub_api_token=s.hf_api_token,
            task="text2text-generation",
            temperature=0.0,
            max_new_tokens=10,
            timeout=10,
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider}")


# ── Provider Info ─────────────────────────────────────────────────────────────

def get_provider_info() -> dict:
    """Returns active provider config. Used by smoke test and /health endpoint."""
    s = get_settings()
    return {
        "provider": s.llm_provider,
        "main_model": s.active_main_model,
        "classifier_model": s.active_classifier_model,
        "guard_model": s.ollama_guard_model if s.using_ollama else s.hf_classifier_model,
        "embedding_model": s.active_embedding_model,
        "ollama_url": s.ollama_base_url if s.using_ollama else None,
        "hf_token_set": bool(s.hf_api_token) if s.using_huggingface else None,
    }