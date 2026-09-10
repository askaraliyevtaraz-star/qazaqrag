from dataclasses import asdict
from typing import Literal

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from agent.router import (
    classify_route,
)
from agent.state import (
    AgentState,
    RouteName,
)
from app.core.config import Settings
from app.services.rag_service import (
    RAGService,
)
from rag.context import build_context
from rag.hybrid import (
    HybridSearchResult,
)
from rag.query_rewriting import (
    QueryRewriter,
)


def build_agent_graph(
    rag_service: RAGService,
    settings: Settings,
    rewriter: QueryRewriter,
):
    def route_query(
        state: AgentState,
    ) -> dict:
        question = state["question"]

        route = classify_route(question)

        return {
            "route": route,
            "effective_query": (question),
            "trace": [f"route:{route}"],
        }

    def route_after_router(
        state: AgentState,
    ) -> RouteName:
        return state["route"]

    def list_sources(
        state: AgentState,
    ) -> dict:
        del state

        counts = rag_service.source_counts()

        lines = [(f"- {source}: {count} chunks") for source, count in sorted(counts.items())]

        answer = "Indexed sources:\n" + "\n".join(lines)

        return {
            "answer": answer,
            "sources": [],
            "trace": ["list_sources"],
        }

    def retrieve_rag(
        state: AgentState,
    ) -> dict:
        query = state["effective_query"]

        results = rag_service.retrieve(query)

        context, sources = build_context(results)

        top_score = None

        if results:
            top_score = float(results[0].rerank_score)

        attempts = (
            state.get(
                "retrieval_attempts",
                0,
            )
            + 1
        )

        return {
            "context": context,
            "sources": [asdict(source) for source in sources],
            "top_score": top_score,
            "retrieval_attempts": (attempts),
            "trace": [(f"retrieve_rag:attempt_{attempts}")],
        }

    def retrieve_identifier(
        state: AgentState,
    ) -> dict:
        query = state["effective_query"]

        bm25_results = rag_service.bm25.search(
            query=query,
            top_k=(settings.retrieve_k),
        )

        candidates = []

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            candidates.append(
                HybridSearchResult(
                    chunk=(result.chunk),
                    rrf_score=0.0,
                    dense_rank=None,
                    bm25_rank=rank,
                    dense_score=None,
                    bm25_score=(result.score),
                )
            )

        reranked = rag_service.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=(settings.final_k),
        )

        context, sources = build_context(reranked)

        top_score = None

        if reranked:
            top_score = float(reranked[0].rerank_score)

        return {
            "context": context,
            "sources": [asdict(source) for source in sources],
            "top_score": top_score,
            "retrieval_attempts": 1,
            "trace": ["retrieve_identifier"],
        }

    def grade_evidence(
        state: AgentState,
    ) -> dict:
        top_score = state.get("top_score")

        return {"trace": [(f"grade_evidence:score={top_score}")]}

    def decide_after_evidence(
        state: AgentState,
    ) -> Literal[
        "generate",
        "rewrite_query",
        "no_answer",
    ]:
        sources = state.get(
            "sources",
            [],
        )

        threshold = settings.min_reranker_score

        top_score = state.get("top_score")

        if not sources:
            enough = False

        elif threshold is None:
            enough = True

        elif top_score is None:
            enough = False

        else:
            enough = top_score >= threshold

        if enough:
            return "generate"

        if state["route"] == "identifier":
            return "no_answer"

        attempts = state.get(
            "retrieval_attempts",
            0,
        )

        max_attempts = state.get(
            "max_retrieval_attempts",
            2,
        )

        if attempts < max_attempts:
            return "rewrite_query"

        return "no_answer"

    def rewrite_query(
        state: AgentState,
    ) -> dict:
        rewritten = rewriter.rewrite(
            original_question=(state["question"]),
            current_query=(state["effective_query"]),
        )

        return {
            "effective_query": (rewritten),
            "trace": [(f"rewrite_query:{rewritten}")],
        }

    def generate(
        state: AgentState,
    ) -> dict:
        answer = rag_service.generator.generate(
            question=(state["question"]),
            context=(state["context"]),
        )

        return {
            "answer": answer,
            "trace": ["generate"],
        }

    def no_answer(
        state: AgentState,
    ) -> dict:
        del state

        return {
            "answer": (
                "The indexed documents "
                "do not contain enough "
                "evidence to answer this "
                "question reliably."
            ),
            "trace": ["no_answer"],
        }

    builder = StateGraph(AgentState)

    builder.add_node(
        "route_query",
        route_query,
    )

    builder.add_node(
        "list_sources",
        list_sources,
    )

    builder.add_node(
        "retrieve_rag",
        retrieve_rag,
    )

    builder.add_node(
        "retrieve_identifier",
        retrieve_identifier,
    )

    builder.add_node(
        "grade_evidence",
        grade_evidence,
    )

    builder.add_node(
        "rewrite_query",
        rewrite_query,
    )

    builder.add_node(
        "generate",
        generate,
    )

    builder.add_node(
        "no_answer",
        no_answer,
    )

    builder.add_edge(
        START,
        "route_query",
    )

    builder.add_conditional_edges(
        "route_query",
        route_after_router,
        {
            "rag": "retrieve_rag",
            "identifier": ("retrieve_identifier"),
            "sources": ("list_sources"),
        },
    )

    builder.add_edge(
        "list_sources",
        END,
    )

    builder.add_edge(
        "retrieve_rag",
        "grade_evidence",
    )

    builder.add_edge(
        "retrieve_identifier",
        "grade_evidence",
    )

    builder.add_conditional_edges(
        "grade_evidence",
        decide_after_evidence,
        {
            "generate": "generate",
            "rewrite_query": ("rewrite_query"),
            "no_answer": ("no_answer"),
        },
    )

    builder.add_edge(
        "rewrite_query",
        "retrieve_rag",
    )

    builder.add_edge(
        "generate",
        END,
    )

    builder.add_edge(
        "no_answer",
        END,
    )

    return builder.compile()
