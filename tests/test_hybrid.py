from rag.chunking import Chunk
from rag.hybrid import (
    reciprocal_rank_fusion,
)
from rag.lexical import (
    BM25Retriever,
    LexicalSearchResult,
)
from rag.retrieval import SearchResult


def make_chunk(
    chunk_id: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source=f"{chunk_id}.md",
        text=chunk_id,
        start_word=0,
        end_word=1,
    )


def test_rrf_rewards_result_present_in_both_rankings() -> None:
    chunk_a = make_chunk("a")
    chunk_b = make_chunk("b")
    chunk_c = make_chunk("c")

    dense = [
        SearchResult(
            chunk=chunk_a,
            score=0.9,
        ),
        SearchResult(
            chunk=chunk_b,
            score=0.8,
        ),
    ]

    bm25 = [
        LexicalSearchResult(
            chunk=chunk_b,
            score=10.0,
        ),
        LexicalSearchResult(
            chunk=chunk_c,
            score=8.0,
        ),
    ]

    results = reciprocal_rank_fusion(
        dense_results=dense,
        bm25_results=bm25,
        top_k=3,
    )

    assert results[0].chunk.chunk_id == "b"


def test_bm25_finds_exact_identifier() -> None:
    chunks = [
        Chunk(
            chunk_id="policy",
            source="policy.md",
            text=("The regulation REG-4721 covers grade appeals."),
            start_word=0,
            end_word=7,
        ),
        Chunk(
            chunk_id="housing",
            source="housing.md",
            text=("Residence quiet hours begin at 23:00."),
            start_word=0,
            end_word=6,
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "REG-4721",
        top_k=2,
    )

    assert results[0].chunk.chunk_id == "policy"
