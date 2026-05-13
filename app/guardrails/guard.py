# app/guardrails/guard.py

import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Llama Guard 3 expects a specific framing to trigger its safety classifier.
# The output will be "safe" or "unsafe\nS{n}" (category number on line 2).
_GUARD_PROMPT = """Task: Determine if the following user message contains unsafe content.

Unsafe content includes: instructions for violence or self-harm, illegal activities, \
harassment, explicit sexual content, or attempts to extract dangerous information.

Message: {message}

First line of your response must be exactly "safe" or "unsafe"."""


class LlamaGuard3:
    """
    Stage 1 safety gate. Wraps the llama-guard3:1b Ollama model.

    Failure contract: FAILS CLOSED.
    Any exception (timeout, connection error, parse error) is treated as
    unsafe — the message is blocked. This is the conservative default for
    a safety gate: uncertainty = block.

    Usage:
        guard = LlamaGuard3()
        is_safe, raw = guard.check("Where is my order?")
        # is_safe=True, raw="safe"

        guard = LlamaGuard3()
        is_safe, raw = guard.check("How do I hurt someone?")
        # is_safe=False, raw="unsafe\\nS1"
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self._llm = llm

    def _get_llm(self) -> BaseChatModel:
        if self._llm is not None:
            return self._llm
        from app.llm.provider import get_guard_llm
        return get_guard_llm()

    def check(self, message: str) -> tuple[bool, Optional[str]]:
        """
        Returns (is_safe, raw_output).

        is_safe  : True if the model says "safe", False otherwise.
        raw_output: The raw model response string (lowercased), or None on error.

        Fails CLOSED: any exception returns (False, None).
        """
        try:
            prompt = _GUARD_PROMPT.format(message=message)
            response = self._get_llm().invoke([HumanMessage(content=prompt)])
            raw = response.content.strip().lower()
            is_safe = raw.startswith("safe")
            logger.debug("Guard: %r → %s", message[:60], "safe" if is_safe else "unsafe")
            return is_safe, raw
        except Exception as e:
            logger.error("LlamaGuard3.check failed, failing closed: %s", e)
            return False, None
