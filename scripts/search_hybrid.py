import argparse
from pathlib import Path

from rag.chunking import (
    load_and_chunk_directory,
)
from rag.embeddings import (
    EmbeddingModel,
)
from rag.hybrid import (
    reciprocal_rank_fusion,
)
from rag.lexical import BM25Retriever
from rag.vector_store import (
    QdrantVectorStore,
)

DATA_DIR = Path("data/demo")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        type=str,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    chunks = load_and_chunk_directory(
        DATA_DIR,
        chunk_size=60,
        overlap=15,
    )

    embedding_model = EmbeddingModel()

    query_embedding = embedding_model.embed_query(args.query)

    vector_store = QdrantVectorStore()

    dense_results = vector_store.search(
        query_embedding=(query_embedding),
        top_k=args.candidate_k,
    )

    bm25 = BM25Retriever(chunks)

    bm25_results = bm25.search(
        query=args.query,
        top_k=args.candidate_k,
    )

    hybrid_results = reciprocal_rank_fusion(
        dense_results=(dense_results),
        bm25_results=(bm25_results),
        top_k=args.top_k,
    )

    print()
    print("QUERY:", args.query)

    print()
    print("=" * 80)
    print("DENSE")

    for rank, result in enumerate(
        dense_results[:5],
        start=1,
    ):
        print(
            rank,
            result.chunk.source,
            result.chunk.chunk_id,
            f"{result.score:.4f}",
        )

    print()
    print("=" * 80)
    print("BM25")

    for rank, result in enumerate(
        bm25_results[:5],
        start=1,
    ):
        print(
            rank,
            result.chunk.source,
            result.chunk.chunk_id,
            f"{result.score:.4f}",
        )

    print()
    print("=" * 80)
    print("HYBRID RRF")

    for rank, result in enumerate(
        hybrid_results,
        start=1,
    ):
        print()
        print(
            f"#{rank}",
            result.chunk.source,
        )

        print(f"RRF={result.rrf_score:.6f}")

        print(
            "dense_rank=",
            result.dense_rank,
            "bm25_rank=",
            result.bm25_rank,
        )

        print(result.chunk.text)


if __name__ == "__main__":
    main()
