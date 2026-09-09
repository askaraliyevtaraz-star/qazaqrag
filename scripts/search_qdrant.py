import argparse

from rag.embeddings import EmbeddingModel
from rag.vector_store import (
    QdrantVectorStore,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--exact",
        action="store_true",
    )

    args = parser.parse_args()

    model = EmbeddingModel()

    query_embedding = model.embed_query(args.query)

    store = QdrantVectorStore()

    results = store.search(
        query_embedding=query_embedding,
        top_k=args.top_k,
        language=args.language,
        source=args.source,
        score_threshold=(args.score_threshold),
        exact=args.exact,
    )

    print()
    print(
        "QUERY:",
        args.query,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print()
        print("=" * 70)

        print(f"#{rank} score={result.score:.4f}")

        print(
            "source:",
            result.chunk.source,
        )

        print()

        print(result.chunk.text)


if __name__ == "__main__":
    main()
