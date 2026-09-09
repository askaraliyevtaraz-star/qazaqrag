import json
import math
from dataclasses import dataclass
from pathlib import Path

from rag.chunking import Chunk


@dataclass
class GoldenQuery:
    query_id: str
    query: str
    query_type: str
    query_language: str

    answerable: bool

    expected_source: str | None
    expected_contains: str | None


@dataclass
class RankedChunk:
    chunk: Chunk
    score: float


def load_golden_queries(
    path: Path,
) -> list[GoldenQuery]:
    queries = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        queries.append(GoldenQuery(**json.loads(line)))

    return queries


def relevance_grade(
    chunk: Chunk,
    query: GoldenQuery,
) -> int:
    if not query.answerable:
        return 0

    if chunk.source != query.expected_source:
        return 0

    if query.expected_contains and query.expected_contains.casefold() in chunk.text.casefold():
        return 2

    return 1


def recall_at_k(
    results: list[RankedChunk],
    corpus: list[Chunk],
    query: GoldenQuery,
    k: int,
) -> float | None:
    if not query.answerable:
        return None

    relevant_ids = {
        chunk.chunk_id
        for chunk in corpus
        if relevance_grade(
            chunk,
            query,
        )
        == 2
    }

    if not relevant_ids:
        raise ValueError(f"Golden query has no answer-bearing chunk: {query.query_id}")

    retrieved_ids = {result.chunk.chunk_id for result in results[:k]}

    return len(relevant_ids & retrieved_ids) / len(relevant_ids)


def reciprocal_rank(
    results: list[RankedChunk],
    query: GoldenQuery,
    k: int,
) -> float | None:
    if not query.answerable:
        return None

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):
        if (
            relevance_grade(
                result.chunk,
                query,
            )
            == 2
        ):
            return 1.0 / rank

    return 0.0


def _dcg(
    grades: list[int],
) -> float:
    total = 0.0

    for rank, grade in enumerate(
        grades,
        start=1,
    ):
        gain = (2**grade) - 1

        discount = math.log2(rank + 1)

        total += gain / discount

    return total


def ndcg_at_k(
    results: list[RankedChunk],
    corpus: list[Chunk],
    query: GoldenQuery,
    k: int,
) -> float | None:
    if not query.answerable:
        return None

    actual_grades = [
        relevance_grade(
            result.chunk,
            query,
        )
        for result in results[:k]
    ]

    all_grades = sorted(
        [
            relevance_grade(
                chunk,
                query,
            )
            for chunk in corpus
        ],
        reverse=True,
    )

    ideal_grades = all_grades[:k]

    ideal_dcg = _dcg(ideal_grades)

    if ideal_dcg == 0:
        return 0.0

    return _dcg(actual_grades) / ideal_dcg
