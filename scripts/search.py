import argparse
import json
from pathlib import Path

import numpy as np

from rag.chunking import Chunk
from rag.embeddings import EmbeddingModel
from rag.retrieval import (
    search_embeddings,
)

ARTIFACT_DIR = Path("artifacts")


def load_chunks(
    path: Path,
) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))

    return [Chunk(**item) for item in data]


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

    embeddings = np.load(ARTIFACT_DIR / "embeddings.npy")

    chunks = load_chunks(ARTIFACT_DIR / "chunks.json")

    model = EmbeddingModel()

    query_embedding = model.embed_query(args.query)

    results = search_embeddings(
        query_embedding=query_embedding,
        corpus_embeddings=embeddings,
        chunks=chunks,
        top_k=args.top_k,
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
