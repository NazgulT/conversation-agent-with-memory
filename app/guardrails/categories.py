# app/guardrails/categories.py

import re
from enum import Enum


class Intent(str, Enum):
    ORDER = "ORDER"
    BILLING = "BILLING"
    RETURNS = "RETURNS"
    SHIPPING = "SHIPPING"
    ACCOUNT = "ACCOUNT"
    PRODUCT_INFO = "PRODUCT_INFO"
    COMPLAINT = "COMPLAINT"
    GENERAL = "GENERAL"
    OFF_TOPIC = "OFF_TOPIC"
    UNSAFE = "UNSAFE"


# All intents that represent a legitimate in-scope customer request.
# OFF_TOPIC and UNSAFE are control values, not routable support categories.
VALID_INTENTS: set["Intent"] = {
    Intent.ORDER,
    Intent.BILLING,
    Intent.RETURNS,
    Intent.SHIPPING,
    Intent.ACCOUNT,
    Intent.PRODUCT_INFO,
    Intent.COMPLAINT,
    Intent.GENERAL,
}

# Human-readable descriptions used in the classifier system prompt.
INTENT_DESCRIPTIONS: dict["Intent", str] = {
    Intent.ORDER:        "order status, tracking, cancellation, or modification",
    Intent.BILLING:      "payments, invoices, charges, or refunds",
    Intent.RETURNS:      "returning items or questions about return policy",
    Intent.SHIPPING:     "delivery, shipping address, or carrier questions",
    Intent.ACCOUNT:      "account settings, login, password, or personal details",
    Intent.PRODUCT_INFO: "product features, availability, sizing, or compatibility",
    Intent.COMPLAINT:    "expressions of dissatisfaction or formal complaints",
    Intent.GENERAL:      "general customer service questions not fitting other categories",
}

# ── Pre-compiled regex patterns (module-load time, never inside a call) ───────

# Prompt-injection attack patterns.
# These fire before any LLM call — fast and cheap first gate.
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(?:(?:previous|all|above|prior)\s+)+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(?:(?:previous|all|above|prior)\s+)+instructions?", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
    # "act as X" — but NOT "act as a helpful/customer/support agent"
    re.compile(r"\bact\s+as\s+(?!a\s+(?:helpful|customer|support\s+agent))", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bDAN\b"),   # "Do Anything Now" jailbreak token
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"override\s+(your|the)\s+(instructions?|guidelines?|rules?)", re.IGNORECASE),
    re.compile(r"disregard\s+(your|all|previous|these)\s+", re.IGNORECASE),
    # "you are now X" — but NOT "you are now a helpful/customer/support agent"
    re.compile(r"you\s+are\s+now\s+(?!a\s+(?:helpful|customer|support))", re.IGNORECASE),
]

# Off-topic keyword patterns.
# Catches clearly out-of-scope topics before spending an LLM call on them.
_OFF_TOPIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(politics|politician|elections?|voting|president|congress)\b", re.IGNORECASE),
    re.compile(r"\b(stock\s+market|cryptocurrency|bitcoin|ethereum|invest(?:ing|ments?))\b", re.IGNORECASE),
    re.compile(r"\b(homework|assignment|essay|thesis|dissertation)\b", re.IGNORECASE),
    re.compile(r"\b(recipe|cooking|baking|chef|restaurant)\b", re.IGNORECASE),
    re.compile(r"\b(sports?|football|basketball|soccer|baseball|tennis|cricket)\b", re.IGNORECASE),
]
