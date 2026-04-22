"""
Intent Detection for AutoStream Agent
Classifies user messages into one of three intent categories.
"""

from enum import Enum


class Intent(str, Enum):
    CASUAL_GREETING = "casual_greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    HIGH_INTENT_LEAD = "high_intent_lead"


# Keywords used for lightweight pre-classification (LLM is the primary classifier)
GREETING_KEYWORDS = [
    "hi", "hello", "hey", "good morning", "good evening",
    "howdy", "sup", "what's up", "greetings", "yo"
]

HIGH_INTENT_KEYWORDS = [
    "sign up", "signup", "subscribe", "buy", "purchase",
    "want to try", "ready to start", "let's go", "i'm in",
    "i want the", "get started", "onboard", "upgrade",
    "take the pro", "take the basic", "enroll", "checkout",
    "how do i join", "how to start"
]

PRODUCT_KEYWORDS = [
    "price", "pricing", "plan", "cost", "how much",
    "feature", "resolution", "caption", "4k", "720p",
    "support", "refund", "cancel", "policy", "trial",
    "what does", "tell me about", "explain", "difference"
]


def keyword_classify(message: str) -> Intent:
    """
    Fast keyword-based classifier as a fallback / pre-filter.
    The LLM-based classification in the agent takes precedence.
    """
    msg = message.lower().strip()

    for kw in HIGH_INTENT_KEYWORDS:
        if kw in msg:
            return Intent.HIGH_INTENT_LEAD

    for kw in PRODUCT_KEYWORDS:
        if kw in msg:
            return Intent.PRODUCT_INQUIRY

    # Check if the message is very short and greeting-like
    words = msg.split()
    if len(words) <= 4:
        for kw in GREETING_KEYWORDS:
            if kw in msg:
                return Intent.CASUAL_GREETING

    return Intent.PRODUCT_INQUIRY  # Default fallback
