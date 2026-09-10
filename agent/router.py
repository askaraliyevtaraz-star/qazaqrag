import re

from agent.state import RouteName

IDENTIFIER_PATTERN = re.compile(
    r"\b[A-Z]{2,}-\d+\b",
    re.IGNORECASE,
)


SOURCE_QUERIES = (
    "list sources",
    "show sources",
    "what sources",
    "какие документы",
    "покажи документы",
    "источники",
    "какие источники",
    "құжаттар",
)


def classify_route(
    question: str,
) -> RouteName:
    normalized = question.strip().casefold()

    if any(phrase in normalized for phrase in SOURCE_QUERIES):
        return "sources"

    if IDENTIFIER_PATTERN.search(question):
        return "identifier"

    return "rag"
