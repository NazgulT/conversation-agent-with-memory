# Customer Support Agent

Multi-turn AI customer support agent with short-term buffer memory,
long-term vector memory, topic guardrails, and LangSmith observability.

Built with: LangChain · LangGraph · LangSmith · FastAPI · Redis

## Quick Start

### Prerequisites
- Python 3.11+
- Ollama

### Setup

1. Clone the repo
2. Create and activate a virtual environment:
   python -m venv .venv && source .venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
4. Copy and fill in environment variables:
   cp .env.example .env
   # Edit .env with your API keys
5. Start infrastructure:
   ollama pull gemma2:2b (or phi3:mini)
   ollama pull llama3.2:3b
6. Run smoke test:
   python scripts/smoke_test.py

All six tests must pass before beginning Phase 1.

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