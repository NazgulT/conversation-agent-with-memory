# app/schemas/guardrails.py

from typing import Optional
from pydantic import BaseModel


class ClassificationResult(BaseModel):
    """
    Output of GuardrailPipeline.run(). This is the only object Phase 4 reads
    from the guardrails module.

    Fields:
        intent          : Classified intent label (e.g. "ORDER", "BILLING").
                          "UNSAFE" or "OFF_TOPIC" when the message is blocked.
        is_safe         : False if Llama Guard 3 or a regex injection check
                          blocked the message.
        is_on_topic     : False if the message is off-topic or unsafe.
        blocked_reason  : "INJECTION", "OFF_TOPIC", or "UNSAFE" when blocked;
                          None when the message passes all gates.
        fallback_response: Pre-generated canned response for blocked messages.
                           None for messages that pass. Phase 4's classify_intent
                           node stores this in state so handle_fallback can read
                           it without re-importing guardrails internals.
        raw_safety_output: Raw text from the Llama Guard 3 model, e.g. "safe"
                           or "unsafe\\nS1". None for regex-blocked messages or
                           when the guard errored.
    """

    intent: str
    is_safe: bool
    is_on_topic: bool
    blocked_reason: Optional[str] = None
    fallback_response: Optional[str] = None
    raw_safety_output: Optional[str] = None
