from app.core.config import (
    Settings,
)
from app.services.agent_service import (
    AgentRAGService,
)
from tests.fakes import (
    FakeRAGService,
)


def make_settings(
    threshold: float | None,
) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="stub",
        min_reranker_score=(threshold),
        agent_max_retrieval_attempts=2,
    )


def test_agent_uses_identifier_route() -> None:
    agent = AgentRAGService(
        rag_service=(FakeRAGService()),
        settings=(make_settings(threshold=0.0)),
    )

    state, _ = agent.query("What is REG-4721?")

    assert state["route"] == "identifier"

    assert "retrieve_identifier" in state["trace"]

    assert "generate" in state["trace"]


def test_agent_retries_weak_retrieval() -> None:
    agent = AgentRAGService(
        rag_service=(FakeRAGService()),
        settings=(make_settings(threshold=10.0)),
    )

    state, _ = agent.query("Unknown question")

    assert state["retrieval_attempts"] == 2

    assert "no_answer" in state["trace"]
