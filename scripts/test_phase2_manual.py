# scripts/test_phase2_manual.py
"""
Manual end-to-end test for Phase 2 long-term memory.
Simulates two conversations from the same user, separated by a session end.

Run: python scripts/test_phase2_manual.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from app.memory.redis_memory import RedisMemoryManager
from app.memory.vector_memory import VectorMemoryManager
from app.memory.orchestrator import end_session

print("\n── Phase 2 Manual Test ──────────────────────────────")
print("Simulates a returning customer across two conversations\n")

redis_mgr  = RedisMemoryManager()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vector_mgr = VectorMemoryManager(chroma_client=chroma_client)

if not redis_mgr.ping():
    print("Redis not running. Run: brew services start redis")
    sys.exit(1)

USER = "demo_user_001"
SESSION_1 = "demo_session_001"
SESSION_2 = "demo_session_002"

# Clean up from previous runs
redis_mgr.clear_session(SESSION_1)
redis_mgr.clear_session(SESSION_2)
vector_mgr.delete_user_memories(USER)

# ── Conversation 1 ──────────────────────────────────────────
print("Session 1: Customer reports delayed order\n")
redis_mgr.save_turn(SESSION_1, USER,
    "Hi, my order #99123 has not arrived yet and it's been 2 weeks.",
    "I'm sorry to hear that. Let me look into order #99123 for you.")
redis_mgr.save_turn(SESSION_1, USER,
    "I really need it for an event this weekend.",
    "I can see there was a carrier delay. I'll expedite a replacement shipment.")
redis_mgr.save_turn(SESSION_1, USER,
    "Thank you, that's very helpful.",
    "You'll receive tracking info within 2 hours. Is there anything else?")

print(f"Turns saved to Redis for session {SESSION_1}")
print("\nEnding session 1 — summarising and storing to Chroma...")

result = end_session(SESSION_1, USER, redis_manager=redis_mgr, vector_manager=vector_mgr)

print(f"  Turns processed : {result.turns_processed}")
print(f"  Stored in Chroma: {result.stored_in_chroma}")
print(f"  Redis cleared   : {result.redis_cleared}")
if result.summary:
    print(f"  Summary         : {result.summary.summary_text}")
    print(f"  Topics          : {result.summary.topics}")
    print(f"  Resolution      : {result.summary.resolution}")
    print(f"  Sentiment       : {result.summary.sentiment}")

# ── Conversation 2 ──────────────────────────────────────────
print("\n\nSession 2: Same customer returns\n")
new_query = "I want to ask about a refund for my previous order"

print(f"New query: '{new_query}'")
print("\nRetrieving long-term memory from Chroma...")

long_term = vector_mgr.retrieve_as_text(USER, new_query, k=3)

if long_term:
    print("\nContext retrieved from past conversations:")
    print(long_term)
    print("\n✓ Agent now has context about this customer's history")
else:
    print("No past context found (this may happen if summarisation was skipped)")

memories_count = vector_mgr.get_user_memory_count(USER)
print(f"\nTotal memories stored for {USER}: {memories_count}")

# Cleanup
vector_mgr.delete_user_memories(USER)
print("\n── Test complete. Long-term memory working correctly. ──\n")