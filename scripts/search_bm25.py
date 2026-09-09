import argparse
from pathlib import Path

from rag.chunking import (
    load_and_chunk_directory,
)
from rag.lexical import BM25Retriever

DATA_DIR = Path("data/demo")


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

    args = parser.parse_args()

    chunks = load_and_chunk_directory(
        DATA_DIR,
        chunk_size=60,
        overlap=15,
    )

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
    )

    print()
    print("QUERY:", args.query)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print()
        print("=" * 70)

        print(f"#{rank} BM25={result.score:.4f}")

        print(
            "source:",
            result.chunk.source,
        )

        print()

        print(result.chunk.text)


if __name__ == "__main__":
    main()
