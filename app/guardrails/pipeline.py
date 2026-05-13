# app/guardrails/pipeline.py

import logging
from typing import Optional

from app.schemas.guardrails import ClassificationResult
from app.guardrails.categories import Intent, _INJECTION_PATTERNS, _OFF_TOPIC_PATTERNS
from app.guardrails.guard import LlamaGuard3
from app.guardrails.classifier import IntentClassifier
from app.guardrails.fallback import FallbackHandler

logger = logging.getLogger(__name__)


class GuardrailPipeline:
    """
    Three-stage safety and intent classification gate.

    Stage 0  — Regex pre-check (injection patterns, off-topic keywords).
               Fast, no LLM. Runs before any model call.
    Stage 1  — Llama Guard 3 binary safety classification. Fails CLOSED.
    Stage 2  — Intent classification with gemma2:2b. Fails OPEN (→ GENERAL).

    pipeline.run() is the ONLY interface Phase 4 uses from this module.
    All internal components are injected via the constructor, enabling
    full unit-test coverage without any real LLM calls.

    Usage:
        pipeline = GuardrailPipeline()
        result = pipeline.run("Where is my order #12345?")
        # ClassificationResult(intent="ORDER", is_safe=True, is_on_topic=True, ...)

        result = pipeline.run("ignore previous instructions")
        # ClassificationResult(intent="UNSAFE", is_safe=False, blocked_reason="INJECTION", ...)
    """

    def __init__(
        self,
        guard: Optional[LlamaGuard3] = None,
        classifier: Optional[IntentClassifier] = None,
        fallback: Optional[FallbackHandler] = None,
    ):
        self._guard = guard or LlamaGuard3()
        self._classifier = classifier or IntentClassifier()
        self._fallback = fallback or FallbackHandler()

    def run(self, message: str) -> ClassificationResult:
        # ── Stage 0a: injection regex ─────────────────────────────────────────
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(message):
                logger.warning("Injection pattern blocked message (len=%d)", len(message))
                return ClassificationResult(
                    intent=Intent.UNSAFE.value,
                    is_safe=False,
                    is_on_topic=False,
                    blocked_reason="INJECTION",
                    fallback_response=self._fallback.get_response("INJECTION"),
                    raw_safety_output=None,
                )

        # ── Stage 0b: off-topic regex ─────────────────────────────────────────
        for pattern in _OFF_TOPIC_PATTERNS:
            if pattern.search(message):
                logger.debug("Off-topic regex blocked message")
                return ClassificationResult(
                    intent=Intent.OFF_TOPIC.value,
                    is_safe=True,
                    is_on_topic=False,
                    blocked_reason="OFF_TOPIC",
                    fallback_response=self._fallback.get_response("OFF_TOPIC"),
                    raw_safety_output=None,
                )

        # ── Stage 1: Llama Guard 3 (fails CLOSED) ────────────────────────────
        is_safe, raw_output = self._guard.check(message)
        if not is_safe:
            logger.warning("LlamaGuard3 blocked message (raw=%r)", raw_output)
            return ClassificationResult(
                intent=Intent.UNSAFE.value,
                is_safe=False,
                is_on_topic=False,
                blocked_reason="UNSAFE",
                fallback_response=self._fallback.get_response("UNSAFE"),
                raw_safety_output=raw_output,
            )

        # ── Stage 2: intent classifier (fails OPEN) ───────────────────────────
        intent = self._classifier.classify(message)
        is_on_topic = intent != Intent.OFF_TOPIC

        return ClassificationResult(
            intent=intent.value,
            is_safe=True,
            is_on_topic=is_on_topic,
            blocked_reason="OFF_TOPIC" if not is_on_topic else None,
            fallback_response=self._fallback.get_response("OFF_TOPIC") if not is_on_topic else None,
            raw_safety_output=raw_output,
        )
