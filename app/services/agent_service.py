import time

from agent.graph import (
    build_agent_graph,
)
from app.core.config import Settings
from app.services.rag_service import (
    RAGService,
)
from rag.query_rewriting import (
    QueryRewriter,
)


class AgentRAGService:
    def __init__(
        self,
        rag_service: RAGService,
        settings: Settings,
    ) -> None:
        self.rag_service = rag_service
        self.settings = settings

        self.rewriter = QueryRewriter(
            provider=(settings.llm_provider),
            api_key=(settings.openai_api_key),
            model=(settings.openai_model),
        )

        self.graph = build_agent_graph(
            rag_service=rag_service,
            settings=settings,
            rewriter=self.rewriter,
        )

    def query(
        self,
        question: str,
    ) -> tuple[dict, float]:
        start = time.perf_counter()

        initial_state = {
            "question": question,
            "effective_query": (question),
            "retrieval_attempts": 0,
            "max_retrieval_attempts": (self.settings.agent_max_retrieval_attempts),
            "trace": [],
        }

        result = self.graph.invoke(initial_state)

        latency_ms = (time.perf_counter() - start) * 1000

        return result, latency_ms
