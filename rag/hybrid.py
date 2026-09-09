from dataclasses import dataclass

from rag.chunking import Chunk
from rag.lexical import (
    LexicalSearchResult,
)
from rag.retrieval import SearchResult


@dataclass
class HybridSearchResult:
    chunk: Chunk

    rrf_score: float

    dense_rank: int | None
    bm25_rank: int | None

    dense_score: float | None
    bm25_score: float | None


def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    bm25_results: list[LexicalSearchResult],
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[HybridSearchResult]:
    records: dict[
        str,
        dict,
    ] = {}

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        chunk_id = result.chunk.chunk_id

        records.setdefault(
            chunk_id,
            {
                "chunk": result.chunk,
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
            },
        )

        record = records[chunk_id]

        record["dense_rank"] = rank
        record["dense_score"] = result.score

        record["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        chunk_id = result.chunk.chunk_id

        records.setdefault(
            chunk_id,
            {
                "chunk": result.chunk,
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
            },
        )

        record = records[chunk_id]

        record["bm25_rank"] = rank
        record["bm25_score"] = result.score

        record["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(
        records.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return [
        HybridSearchResult(
            chunk=item["chunk"],
            rrf_score=float(item["rrf_score"]),
            dense_rank=(item["dense_rank"]),
            bm25_rank=(item["bm25_rank"]),
            dense_score=(item["dense_score"]),
            bm25_score=(item["bm25_score"]),
        )
        for item in ranked[:top_k]
    ]
