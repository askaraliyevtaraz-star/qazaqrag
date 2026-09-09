from pathlib import Path

from rag.chunking import (
    Chunk,
    load_and_chunk_directory,
)
from rag.evaluation import (
    GoldenQuery,
    RankedChunk,
    load_golden_queries,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_grade,
)


def test_every_answerable_query_has_relevant_chunk() -> None:
    chunks = load_and_chunk_directory(
        Path("data/demo"),
        chunk_size=60,
        overlap=15,
    )

    queries = load_golden_queries(Path("evaluation/golden_queries.jsonl"))

    for query in queries:
        if not query.answerable:
            continue

        answer_chunks = [
            chunk
            for chunk in chunks
            if relevance_grade(
                chunk,
                query,
            )
            == 2
        ]

        assert answer_chunks, f"No answer chunk for {query.query_id}: {query.query}"


def test_retrieval_metrics_reward_correct_ranking() -> None:
    correct = Chunk(
        chunk_id="correct",
        source="correct.md",
        text="The answer is 42.",
        start_word=0,
        end_word=5,
    )

    wrong = Chunk(
        chunk_id="wrong",
        source="wrong.md",
        text="Unrelated content.",
        start_word=0,
        end_word=2,
    )

    query = GoldenQuery(
        query_id="test",
        query="What is the answer?",
        query_type="semantic",
        query_language="en",
        answerable=True,
        expected_source="correct.md",
        expected_contains="answer is 42",
    )

    results = [
        RankedChunk(
            chunk=wrong,
            score=0.9,
        ),
        RankedChunk(
            chunk=correct,
            score=0.8,
        ),
    ]

    corpus = [
        correct,
        wrong,
    ]

    assert (
        recall_at_k(
            results,
            corpus,
            query,
            k=1,
        )
        == 0.0
    )

    assert (
        recall_at_k(
            results,
            corpus,
            query,
            k=2,
        )
        == 1.0
    )

    assert (
        reciprocal_rank(
            results,
            query,
            k=2,
        )
        == 0.5
    )

    assert (
        0
        < ndcg_at_k(
            results,
            corpus,
            query,
            k=2,
        )
        < 1
    )
