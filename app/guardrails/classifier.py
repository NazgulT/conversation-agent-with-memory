# app/guardrails/classifier.py

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from app.guardrails.categories import Intent, VALID_INTENTS, INTENT_DESCRIPTIONS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an e-commerce customer support intent classifier.

Classify the customer message into exactly ONE of these categories:
{categories}

Or classify as OFF_TOPIC if the message is not related to e-commerce customer support.

Rules:
- Reply with ONLY the category name in UPPERCASE. No explanation. No punctuation.
- If the message fits more than one category, choose the most specific one.
- If genuinely unsure between support categories, use GENERAL."""

_USER_TEMPLATE = "Customer message: {message}\n\nCategory:"


class IntentClassifier:
    """
    Stage 2 intent classifier. Uses gemma2:2b via Ollama.

    Failure contract: FAILS OPEN.
    Any exception or unrecognised model output defaults to GENERAL, not a block.
    The message has already cleared the safety gate — an unknown intent should
    still receive a generic support response rather than be silently dropped.

    Usage:
        classifier = IntentClassifier()
        intent = classifier.classify("Where is my order?")
        # Intent.ORDER

        intent = classifier.classify("Tell me about the weather")
        # Intent.OFF_TOPIC (or Intent.GENERAL on model error)
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self._llm = llm
        self._system_prompt: Optional[str] = None

    def _get_llm(self) -> BaseChatModel:
        if self._llm is not None:
            return self._llm
        from app.llm.provider import get_classifier_llm
        return get_classifier_llm()

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            lines = "\n".join(
                f"- {intent.value}: {desc}"
                for intent, desc in INTENT_DESCRIPTIONS.items()
            )
            self._system_prompt = _SYSTEM_PROMPT.format(categories=lines)
        return self._system_prompt

    def classify(self, message: str) -> Intent:
        """
        Returns the classified Intent. Fails OPEN: returns GENERAL on any error.
        """
        try:
            response = self._get_llm().invoke([
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=_USER_TEMPLATE.format(message=message)),
            ])

            raw = response.content.strip()
            if not raw:
                return Intent.GENERAL

            # Take only the first word, strip punctuation, uppercase
            label = raw.split()[0].rstrip(".:,;!").upper()

            if label == Intent.OFF_TOPIC.value:
                return Intent.OFF_TOPIC

            for intent in VALID_INTENTS:
                if intent.value == label:
                    logger.debug("Classifier: %r → %s", message[:60], label)
                    return intent

            logger.warning("Classifier returned unknown label %r, defaulting to GENERAL", label)
            return Intent.GENERAL

        except Exception as e:
            logger.error("IntentClassifier.classify failed, defaulting to GENERAL: %s", e)
            return Intent.GENERAL
