# scripts/smoke_test.py
"""
Phase 0 Smoke Test — macOS Monterey, No Docker, Switchable LLM Provider

Checks (7 total):
  1. Config     — settings load, provider detected correctly
  2. Redis      — Homebrew native service: ping, write, read, list, TTL
  3. Chroma     — In-process: embed, upsert, query, user-ID filter
  4. Embeddings — Active provider's embedding model
  5. Main LLM   — Active provider's main chat model
  6. Classifier — Active provider's fast classification model
  7. LangSmith  — Auth, project existence, trace arrival

Run from project root (venv must be active):
    python scripts/smoke_test.py

To test the other provider, change LLM_PROVIDER in .env and re-run.
"""

import os
import sys
import time
import json

# Add project root to path so we can import from app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings

settings = get_settings()

# Set LangSmith env vars before importing LangChain
os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

# ── Terminal colours ───────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {BLUE}→{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def section(title):
    print(f"\n{BOLD}{title}{RESET}")
    print("─" * 54)


# ── Test 1: Config & Provider Detection ───────────────────────────────────────

def test_config() -> bool:
    section("Test 1: Config & Provider Detection")
    try:
        from app.llm.provider import get_provider_info
        info_dict = get_provider_info()

        ok(f"LLM_PROVIDER    : {info_dict['provider'].upper()}")
        ok(f"Main model      : {info_dict['main_model']}")
        ok(f"Classifier model: {info_dict['classifier_model']}")
        ok(f"Embedding model : {info_dict['embedding_model']}")

        if settings.using_ollama:
            ok(f"Ollama URL      : {info_dict['ollama_url']}")
        else:
            token_status = "set ✓" if info_dict["hf_token_set"] else "MISSING ✗"
            ok(f"HF token        : {token_status}")
            if not info_dict["hf_token_set"]:
                fail("HF_API_TOKEN is required when LLM_PROVIDER=huggingface")
                return False

        ok(f"LangSmith project: {settings.langsmith_project}")
        return True
    except Exception as e:
        fail(f"Config failed: {e}")
        return False


# ── Test 2: Redis ─────────────────────────────────────────────────────────────

def test_redis() -> bool:
    section("Test 2: Redis (Homebrew native)")
    try:
        import redis as redis_lib

        info(f"Connecting to {settings.redis_url} ...")
        client = redis_lib.from_url(
            settings.redis_url,
            db=settings.redis_db,
            decode_responses=True,
            socket_connect_timeout=5,
        )

        # Ping
        assert client.ping(), "Ping failed"
        ok("Ping → PONG")

        # Write & read
        key = "smoke_test:phase0"
        val = json.dumps({"ok": True, "ts": time.time()})
        client.set(key, val, ex=60)
        back = json.loads(client.get(key))
        assert back["ok"] is True
        ok("Write → Read round-trip ✓")

        # List ops (used by buffer memory in Phase 1)
        lkey = "smoke_test:list"
        client.delete(lkey)
        client.rpush(lkey, "turn_1", "turn_2", "turn_3")
        items = client.lrange(lkey, 0, -1)
        assert len(items) == 3
        ok(f"List ops (rpush/lrange): {items}")

        # TTL
        client.set("smoke_test:ttl", "x", ex=5)
        ttl = client.ttl("smoke_test:ttl")
        assert 1 <= ttl <= 5
        ok(f"TTL set and readable: expires in {ttl}s")

        # Clean up
        client.delete(key, lkey, "smoke_test:ttl")
        ok("Test keys cleaned up")
        return True

    except redis_lib.ConnectionError:
        fail(f"Cannot connect to Redis at {settings.redis_url}")
        fail("  Fix: brew services start redis")
        fail("  Then verify: redis-cli ping")
        return False
    except Exception as e:
        fail(f"Redis error: {e}")
        return False


# ── Test 3: Embeddings ────────────────────────────────────────────────────────

def test_embeddings():
    """Returns (success: bool, sample_vector: list | None)"""
    section(f"Test 3: Embeddings ({settings.llm_provider.upper()})")

    try:
        from app.llm.provider import get_embeddings

        info(f"Loading embedding model: {settings.active_embedding_model}")
        info("(First run may download the model — this is normal)")

        embedder = get_embeddings()

        texts = [
            "My order has not arrived.",
            "I need help with a refund.",
        ]
        info(f"Embedding {len(texts)} test strings ...")
        start = time.time()
        vectors = embedder.embed_documents(texts)
        elapsed = round((time.time() - start) * 1000)

        assert len(vectors) == len(texts), f"Expected {len(texts)} vectors"
        dim = len(vectors[0])
        ok(f"Embedded {len(texts)} strings → {dim}-dim vectors in {elapsed}ms")

        # Dimension sanity check
        # nomic-embed-text → 768, all-MiniLM-L6-v2 → 384
        if dim < 100:
            fail(f"Vector dim {dim} is suspiciously small — check model name")
            return False, None

        ok(f"Dimension {dim} ✓ (expected 384 or 768 depending on model)")

        # embed_query (different code path from embed_documents)
        qvec = embedder.embed_query("Where is my package?")
        assert len(qvec) == dim
        ok(f"embed_query → {len(qvec)}-dim vector ✓")

        return True, vectors[0]

    except Exception as e:
        fail(f"Embeddings failed: {e}")
        if settings.using_ollama:
            fail("  Fix: ollama pull nomic-embed-text")
            fail("  Then verify: ollama list")
        return False, None


# ── Test 4: Chroma In-Process ─────────────────────────────────────────────────

def test_chroma() -> bool:
    section("Test 4: Chroma (in-process, no server)")
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from app.llm.provider import get_embeddings

        embedder = get_embeddings()

        info("Creating temporary in-memory Chroma collection ...")
        # Using in-memory client for the smoke test — no disk writes
        import chromadb
        raw_client = chromadb.EphemeralClient()  # in-memory only, deleted after test

        store = Chroma(
            client=raw_client,
            collection_name="smoke_test",
            embedding_function=embedder,
        )

        # Upsert 3 test docs — 2 for user_001, 1 for user_002
        docs = [
            Document(
                page_content="Jane's order #A123 arrived late due to a warehouse delay. Refund of $12 issued.",
                metadata={"user_id": "user_001", "topics": "order,refund"},
            ),
            Document(
                page_content="Jane called again about a billing error. $15 charge reversed.",
                metadata={"user_id": "user_001", "topics": "billing"},
            ),
            Document(
                page_content="John had a login issue. Password reset completed.",
                metadata={"user_id": "user_002", "topics": "account"},
            ),
        ]

        info(f"Upserting {len(docs)} documents ...")
        ids = store.add_documents(docs)
        ok(f"Upserted {len(ids)} docs")

        # Semantic search — should find order/shipping docs
        results = store.similarity_search("problem with my delivery", k=2)
        assert len(results) > 0, "Query returned nothing"
        ok(f"Semantic search returned {len(results)} results")
        ok(f"  Top result: '{results[0].page_content[:70]}...'")

        # CRITICAL: user-ID filter — must never return another user's docs
        info("Testing user_id filter (privacy check) ...")
        user_results = store.similarity_search(
            "my previous contact",
            k=3,
            filter={"user_id": "user_001"},
        )
        for doc in user_results:
            assert doc.metadata["user_id"] == "user_001", (
                f"PRIVACY LEAK: got doc from {doc.metadata['user_id']}"
            )
        ok(f"user_id filter → {len(user_results)} docs, all from user_001 ✓")
        ok("No cross-user data leakage ✓")

        return True

    except Exception as e:
        fail(f"Chroma test failed: {e}")
        return False


# ── Test 5: Main LLM ──────────────────────────────────────────────────────────

def test_main_llm() -> bool:
    section(f"Test 5: Main LLM ({settings.llm_provider.upper()} · {settings.active_main_model})")
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from app.llm.provider import get_llm

        info(f"Loading model: {settings.active_main_model}")
        if settings.using_ollama:
            info("(Ollama must be running: ollama serve)")
        else:
            info("(HuggingFace — first call may have cold-start delay of ~30s)")

        llm = get_llm()

        messages = [
            SystemMessage(content="You are a concise assistant. Answer in one sentence only."),
            HumanMessage(content="Say exactly: 'Phase 0 LLM test passed.' Nothing else."),
        ]

        info("Invoking LLM (this call is traced in LangSmith) ...")
        start = time.time()
        response = llm.invoke(messages)
        elapsed = round((time.time() - start) * 1000)

        content = response.content.strip()
        ok(f"Response received in {elapsed}ms")
        ok(f"Content: '{content}'")

        if elapsed > 20000:
            warn(f"Response took {elapsed}ms — slow but acceptable for local models")

        return True

    except Exception as e:
        fail(f"Main LLM failed: {e}")
        if settings.using_ollama:
            fail("  Is Ollama running? Run: ollama serve")
            fail(f"  Is the model pulled? Run: ollama pull {settings.ollama_main_model}")
            fail("  Check: curl http://localhost:11434/api/tags")
        else:
            fail("  HuggingFace error — check HF_API_TOKEN and model access")
            fail(f"  Confirm you accepted the license for {settings.hf_main_model}")
        return False


# ── Test 6: Classifier LLM ────────────────────────────────────────────────────

def test_classifier_llm() -> bool:
    section(f"Test 6: Classifier LLM ({settings.active_classifier_model})")
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from app.llm.provider import get_classifier_llm

        llm = get_classifier_llm()

        # gemma2:2b at temperature=0 with num_predict=5 should return
        # exactly one of these category words with no surrounding text.
        # The .split()[0] call handles the rare case where whitespace
        # or a newline precedes the label.
        CLASSIFIER_PROMPT = (
            "Classify this customer message into exactly ONE of these categories: "
            "ORDER, BILLING, TECHNICAL, ACCOUNT, GENERAL, OFF_TOPIC.\n"
            "Rules:\n"
            "- Return ONLY the category name in UPPERCASE.\n"
            "- Do not add any explanation, punctuation, or extra words.\n"
            "- If unsure, return OFF_TOPIC.\n\n"
            "Message: {message}\n"
            "Category:"
        )

        test_cases = [
            # (input message, expected label, must_match_exactly)
            ("Where is my order #12345?",               "ORDER",     True),
            ("I want a refund for last month's charge", "BILLING",   True),
            ("I can't log into my account",             "ACCOUNT",   True),
            ("What is the capital of France?",          "OFF_TOPIC", True),
            ("My app keeps crashing on startup",        "TECHNICAL", False),
            # False = warn but don't fail — small models sometimes say GENERAL
        ]

        all_required_passed = True
        for message, expected, required in test_cases:
            msgs = [
                SystemMessage(content=CLASSIFIER_PROMPT.format(message=message)),
                HumanMessage(content=message),
            ]

            start = time.time()
            resp = llm.invoke(msgs)
            elapsed = round((time.time() - start) * 1000)

            # gemma2:2b with num_predict=5 may include trailing whitespace
            # Take the first word only and uppercase it
            raw = resp.content.strip()
            result = raw.split()[0].upper().rstrip(".,!?") if raw.split() else "EMPTY"

            if result == expected:
                ok(f"'{message[:42]}' → {result} ✓ ({elapsed}ms)")
            elif not required:
                warn(f"'{message[:42]}' → {result} (expected {expected}) — acceptable")
            else:
                fail(f"'{message[:42]}' → {result} (expected {expected})")
                all_required_passed = False

            # Latency gate — classifier must respond fast
            if elapsed > 8000:
                fail(f"  Classifier too slow: {elapsed}ms (limit: 8000ms)")
                fail("  On Intel Mac this may indicate Ollama is under load")
                all_required_passed = False
            elif elapsed > 4000:
                warn(f"  Slow response: {elapsed}ms — acceptable but monitor")

        if all_required_passed:
            ok("Classifier LLM: all required classifications correct ✓")
        return all_required_passed

    except Exception as e:
        fail(f"Classifier LLM failed: {e}")
        if settings.using_ollama:
            fail(f"  Is the model pulled? Run: ollama pull {settings.ollama_classifier_model}")
            fail("  Check model list: ollama list")
        return False

# ── Test 7: LangSmith ─────────────────────────────────────────────────────────

def test_langsmith() -> bool:
    section("Test 7: LangSmith Tracing")
    try:
        from langsmith import Client

        info("Connecting to LangSmith ...")
        client = Client(api_key=settings.langsmith_api_key)

        projects = list(client.list_projects())
        names = [p.name for p in projects]
        ok(f"Authenticated — {len(projects)} project(s) found")

        if settings.langsmith_project in names:
            ok(f"Project '{settings.langsmith_project}' exists ✓")
        else:
            info(f"Project '{settings.langsmith_project}' will be created on first trace")

        # List recent runs
        try:
            runs = list(client.list_runs(
                project_name=settings.langsmith_project,
                limit=3,
            ))
            if runs:
                ok(f"Found {len(runs)} recent trace(s)")
                ok(f"Latest: '{runs[0].name}' at {runs[0].start_time}")
            else:
                info("No traces yet — LLM tests above will have created some")
                info("Wait 30s and check the LangSmith dashboard")
        except Exception:
            info("Could not list runs yet — normal on first run")

        ok(f"Dashboard: https://smith.langchain.com")
        ok(f"Project: {settings.langsmith_project}")
        return True

    except Exception as e:
        fail(f"LangSmith failed: {e}")
        if "403" in str(e) or "auth" in str(e).lower():
            fail("  Invalid LANGSMITH_API_KEY — check your .env")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*54}{RESET}")
    print(f"{BOLD}  Phase 0 Smoke Test{RESET}")
    print(f"{BOLD}{'='*54}{RESET}")
    print(f"  Provider    : {BOLD}{settings.llm_provider.upper()}{RESET}")
    print(f"  Main model  : {settings.active_main_model}")
    print(f"  Embeddings  : {settings.active_embedding_model}")
    print(f"  Redis       : {settings.redis_url}")
    print(f"  Chroma      : in-process (no server)")
    print(f"  LangSmith   : {settings.langsmith_project}")
    print(f"  {DIM}Tip: change LLM_PROVIDER in .env to switch providers{RESET}")

    results = {}
    results["config"]         = test_config()
    results["redis"]          = test_redis()
    embed_ok, _               = test_embeddings()
    results["embeddings"]     = embed_ok
    results["chroma"]         = test_chroma()
    results["main_llm"]       = test_main_llm()
    results["classifier_llm"] = test_classifier_llm()
    results["langsmith"]      = test_langsmith()

    section("Results Summary")
    all_passed = True
    for name, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}  ✓ All 7 tests passed.{RESET}")
        print(f"{GREEN}  Provider: {settings.llm_provider.upper()} · Phase 0 complete.{RESET}")
        print(f"{DIM}  Switch providers: change LLM_PROVIDER in .env and re-run.{RESET}")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"{RED}{BOLD}  ✗ {len(failed)} test(s) failed: {', '.join(failed)}{RESET}")
        print(f"{YELLOW}  Fix failures above before starting Phase 1.{RESET}")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()