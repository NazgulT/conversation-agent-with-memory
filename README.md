# Customer Support Agent

A multi-turn AI customer support agent with short-term buffer memory,
long-term vector memory, topic guardrails, and LangSmith observability.

---

## Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM framework | LangChain |
| Main LLM | Ollama · `llama3.2:3b` (local) |
| Classifier LLM | Ollama · `gemma2:2b` (local) |
| Embeddings | Ollama · `nomic-embed-text` (local) |
| LLM fallback | HuggingFace free tier (switchable via `.env`) |
| Short-term memory | Redis 7.4 (Homebrew) |
| Long-term memory | Chroma (in-process) |
| API | FastAPI |
| Observability | LangSmith |


---

## Quick Start

### Prerequisites

```bash
python3 --version    # 3.11+
brew --version       # Homebrew
ollama --version     # Ollama
redis-cli ping       # → PONG (brew services start redis)
```

### Setup

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd customer-support-agent

# 2. Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env: set LANGSMITH_API_KEY and HF_API_TOKEN
# Set LLM_PROVIDER=ollama (default) or LLM_PROVIDER=huggingface

# 5. Pull Ollama models (one-time, ~4GB total)
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull nomic-embed-text

# 6. Start Redis
brew services start redis

# 7. Run smoke test — all 7 must pass
python scripts/smoke_test.py
```

### Switching LLM provider

No code changes needed. Edit one line in `.env`:

```env
LLM_PROVIDER=ollama        # fully local, offline, no API keys
LLM_PROVIDER=huggingface   # cloud free tier, requires HF_API_TOKEN
```

---

## Build Progress

### ✅ Phase 0 — Infrastructure Setup

Verified all services communicate correctly before writing any agent logic.

- Native Redis via Homebrew (`redis-cli ping` → PONG)
- Chroma running in-process (no server required)
- Ollama serving `llama3.2:3b`, `gemma2:2b`, `nomic-embed-text`, `llama-guard3:1b`
- LangSmith project created and receiving traces
- HuggingFace token configured as cloud fallback
- Switchable provider via `LLM_PROVIDER` in `.env`
- **7-point smoke test passing** (`python scripts/smoke_test.py`)

---

### ✅ Phase 1 — Short-Term Buffer Memory

The agent remembers the current conversation using a sliding window buffer in Redis.


---

### ✅ Phase 2 — Long-Term Vector Memory

When a session ends, the conversation is compressed into a summary and stored
in Chroma. When a new session starts, semantically relevant past summaries are
retrieved and injected into the prompt — giving the agent memory of returning customers.

---

### ✅  Phase 3 — Topic Guardrails

Intent classifier using `gemma2:2b` — routes off-topic messages to fallback
before reaching the main LLM. Prompt injection detection.


### ✅  Phase 4 — LangGraph Agent Graph

Wire all components into a stateful `StateGraph` with conditional routing:
`classify_intent → retrieve_short_term → retrieve_long_term → build_context
→ generate_response → save_memory`.

### ⏳ Phase 5 — FastAPI Layer

HTTP endpoints: `POST /chat`, `POST /sessions`, `DELETE /sessions/{id}`,
`GET /health`.

### ⏳ Phase 6 — Failure Handling & LangSmith Logging

Typed `FailureReason` enum. Every failure node logs structured metadata to
LangSmith. Error wrapping in all graph nodes.

### ⏳ Phase 7 — Testing & Evaluation

LangSmith evaluation datasets. Load testing with Locust.

### ⏳ Phase 8 — Deployment

Dockerfile, managed Redis (Redis Cloud), production Chroma or Qdrant,
monitoring alerts.

---

## Running Tests

```bash
# All unit tests (no infrastructure needed)
pytest tests/unit/ -v

# All integration tests (Redis + Ollama must be running)
pytest tests/integration/ -v

# Full suite
pytest -v
```

---

## LangSmith Traces

Every LLM call is automatically traced.

```
https://smith.langchain.com → Projects → customer-support-agent
```

Set `LANGCHAIN_TRACING_V2=false` in `.env` to disable tracing locally.

---


## Architecture
See /docs/architecture.md (added in Phase 4)

## Development Phases
- Phase 0: Infrastructure setup (current)
- Phase 1: Short-term Redis memory
- Phase 2: Long-term vector memory
- Phase 3: Topic guardrails
- Phase 4: LangGraph agent graph
- Phase 5: FastAPI layer
- Phase 6: Failure handling + LangSmith logging
- Phase 7: Testing and evaluation
- Phase 8: Deployment