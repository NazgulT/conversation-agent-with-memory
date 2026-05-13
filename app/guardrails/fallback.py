# app/guardrails/fallback.py

from typing import Optional

# Keyed by blocked_reason string. The INJECTION and UNSAFE responses are
# intentionally identical — telling a user *why* their message was blocked
# gives adversaries information about what patterns to avoid next time.
_RESPONSES: dict[str, str] = {
    "OFF_TOPIC": (
        "I can only help with e-commerce support topics such as orders, billing, "
        "returns, shipping, and account questions. How can I assist you today?"
    ),
    "UNSAFE": (
        "I'm not able to help with that request. If you have questions about "
        "your orders, billing, or account, I'm happy to assist."
    ),
    "INJECTION": (
        "I'm not able to help with that request. If you have questions about "
        "your orders, billing, or account, I'm happy to assist."
    ),
    "LOW_CONFIDENCE": (
        "I'm not confident I understood your question correctly. Could you rephrase it, "
        "or contact our support team directly for further help?"
    ),
    "DEFAULT": (
        "I can only assist with customer support questions. Please ask about your "
        "orders, billing, returns, shipping, or account and I'll be happy to help."
    ),
}


class FallbackHandler:
    """
    Returns canned responses for blocked or low-confidence messages.

    All responses are pre-written strings — no LLM call, no latency.
    This guarantees handle_fallback always produces output even when the
    LLM stack is completely unavailable.
    """

    def get_response(self, reason: Optional[str] = None) -> str:
        return _RESPONSES.get(reason or "DEFAULT", _RESPONSES["DEFAULT"])
