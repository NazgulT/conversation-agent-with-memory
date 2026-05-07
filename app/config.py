# app/config.py

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    """
    All application config loaded from .env.

    LLM_PROVIDER controls which LLM backend is used globally.
    Set to 'ollama' for fully local offline use.
    Set to 'huggingface' for cloud free-tier fallback.

    When LLM_PROVIDER=ollama:
      - Main chatbot  : llama3.2:3b  (Meta, 3B params, 128K ctx, dialogue-optimised)
      - Classifier    : gemma2:2b    (Google, 2B params, deterministic classification)
      - Embeddings    : nomic-embed-text (768-dim, MTEB top performer)

    When LLM_PROVIDER=huggingface:
      - Main chatbot  : mistralai/Mistral-7B-Instruct-v0.3 (cloud, free tier)
      - Classifier    : google/flan-t5-large (cloud, free tier)
      - Embeddings    : sentence-transformers/all-MiniLM-L6-v2 (local, no API)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Provider Switch ───────────────────────────────────────────
    llm_provider: str = "ollama"

    # ── Ollama ────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_main_model: str = "llama3.2:3b"       
    ollama_classifier_model: str = "phi3:mini"    
    ollama_embedding_model: str = "nomic-embed-text"

    # ── HuggingFace ───────────────────────────────────────────────
    hf_api_token: str = ""
    hf_main_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_classifier_model: str = "google/flan-t5-large"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── LangSmith ─────────────────────────────────────────────────
    langsmith_api_key: str
    langsmith_project: str = "customer-support-agent"
    langchain_tracing_v2: str = "true"

    # ── Redis (Homebrew native) ───────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_ttl_seconds: int = 86400

    # ── Chroma (in-process) ───────────────────────────────────────
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "customer-memory"

    # ── Agent Behaviour ───────────────────────────────────────────
    buffer_memory_size: int = 10
    long_term_retrieval_k: int = 3
    confidence_threshold: float = 0.4
    max_retries: int = 2

    # Two separate timeouts now — classifier must be fast, main LLM can be slow
    llm_timeout_seconds: int = 30           # main chatbot: generous for CPU inference
    classifier_timeout_seconds: int = 8     # classifier: must return < 8s or it fails

    # ── App ───────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── Validators ────────────────────────────────────────────────

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"ollama", "huggingface"}
        if v.lower() not in allowed:
            raise ValueError(
                f"LLM_PROVIDER must be one of {allowed}, got '{v}'"
            )
        return v.lower()

    # ── Convenience properties ────────────────────────────────────

    @property
    def using_ollama(self) -> bool:
        return self.llm_provider == "ollama"

    @property
    def using_huggingface(self) -> bool:
        return self.llm_provider == "huggingface"

    @property
    def active_main_model(self) -> str:
        return self.ollama_main_model if self.using_ollama else self.hf_main_model

    @property
    def active_classifier_model(self) -> str:
        return (
            self.ollama_classifier_model
            if self.using_ollama
            else self.hf_classifier_model
        )

    @property
    def active_embedding_model(self) -> str:
        return (
            self.ollama_embedding_model
            if self.using_ollama
            else self.hf_embedding_model
        )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()