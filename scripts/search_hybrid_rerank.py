import argparse
from pathlib import Path

from rag.chunking import (
    load_and_chunk_directory,
)
from rag.embeddings import EmbeddingModel
from rag.hybrid import (
    reciprocal_rank_fusion,
)
from rag.lexical import BM25Retriever
from rag.reranking import (
    MultilingualReranker,
)
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
        "--retrieve-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--rerank-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--final-k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    chunks = load_and_chunk_directory(
        DATA_DIR,
        chunk_size=60,
        overlap=15,
    )

    embedding_model = EmbeddingModel()

    query_embedding = embedding_model.embed_query(args.query)

    dense = QdrantVectorStore().search(
        query_embedding=(query_embedding),
        top_k=args.retrieve_k,
    )

    bm25 = BM25Retriever(chunks).search(
        query=args.query,
        top_k=args.retrieve_k,
    )

    hybrid = reciprocal_rank_fusion(
        dense_results=dense,
        bm25_results=bm25,
        top_k=args.rerank_k,
    )

    reranker = MultilingualReranker()

    final_results = reranker.rerank(
        query=args.query,
        candidates=hybrid,
        top_k=args.final_k,
    )

    print()
    print("=" * 80)
    print("FINAL RERANKED RESULTS")

    for rank, result in enumerate(
        final_results,
        start=1,
    ):
        candidate = result.candidate

        print()
        print(f"#{rank}")

        print(
            "source:",
            candidate.chunk.source,
        )

        print(
            "rerank_score:",
            f"{result.rerank_score:.4f}",
        )

        print(
            "RRF:",
            f"{candidate.rrf_score:.6f}",
        )

        print(
            "dense_rank:",
            candidate.dense_rank,
        )

        print(
            "bm25_rank:",
            candidate.bm25_rank,
        )

        print()

        print(candidate.chunk.text)


if __name__ == "__main__":
    main()
