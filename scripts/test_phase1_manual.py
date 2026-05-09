# scripts/test_phase1_manual.py
"""
Manual end-to-end test for Phase 1.
Run: python scripts/test_phase1_manual.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.memory.redis_memory import RedisMemoryManager

manager = RedisMemoryManager()

if not manager.ping():
    print("Redis not running. Run: brew services start redis")
    sys.exit(1)

SESSION = "manual_test_session"
USER    = "user_demo_001"

# Clean up any previous run
manager.clear_session(SESSION)
print(f"\n── Phase 1 Manual Test ──────────────────────")
print(f"Session : {SESSION}")
print(f"Buffer  : {manager._settings.buffer_memory_size} turns\n")

# Simulate a 5-turn conversation
exchanges = [
    ("Hi, I need help with my order", "Of course! What's your order number?"),
    ("It's order #98765",             "Found it. Your order is currently in transit."),
    ("When will it arrive?",          "Expected delivery is tomorrow by 6pm."),
    ("Can I change the address?",     "I can try — what's the new address?"),
    ("123 New Street, London",        "Done! Address updated successfully."),
]

for i, (human, assistant) in enumerate(exchanges):
    turn = manager.save_turn(SESSION, USER, human, assistant)
    print(f"Turn {i}: saved (index {turn.turn_index})")

print(f"\n── Conversation history (last {manager._settings.buffer_memory_size} turns) ──")
print(manager.get_formatted_history(SESSION))

print(f"\n── Session metadata ──")
meta = manager.get_session_meta(SESSION)
print(f"Turn count : {meta.turn_count}")
print(f"Created    : {meta.created_at}")
print(f"Last active: {meta.last_active}")

print(f"\n── Buffer window demonstration ──")
print(f"Saving 8 more turns to overflow the buffer...")
for i in range(8):
    manager.save_turn(SESSION, USER, f"Overflow message {i}", f"Overflow reply {i}")

turns = manager.get_history(SESSION)
print(f"Turns in Redis after overflow: {len(turns)} (max: {manager._settings.buffer_memory_size})")
print(f"Oldest turn now: '{turns[0].role_human}'")
print(f"Latest turn now: '{turns[-1].role_human}'")

# Clean up
manager.clear_session(SESSION)
print(f"\n── Session cleared ──")
print("Phase 1 manual test complete.\n")