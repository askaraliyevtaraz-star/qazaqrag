from agent.router import (
    classify_route,
)


def test_identifier_route() -> None:
    assert classify_route("What is REG-4721?") == "identifier"


def test_identifier_apl_route() -> None:
    assert classify_route("Explain APL-19") == "identifier"


def test_sources_route() -> None:
    assert classify_route("Show sources") == "sources"


def test_normal_question_routes_to_rag() -> None:
    assert classify_route("When does the library close?") == "rag"
