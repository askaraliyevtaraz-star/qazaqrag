from dataclasses import dataclass

import numpy as np

from rag.chunking import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def search_embeddings(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[SearchResult]:
    if len(corpus_embeddings) != len(chunks):
        raise ValueError("Number of embeddings must match chunks")

    scores = corpus_embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        SearchResult(
            chunk=chunks[index],
            score=float(scores[index]),
        )
        for index in top_indices
    ]
