import argparse
import json
from pathlib import Path

import numpy as np

from rag.chunking import Chunk
from rag.embeddings import EmbeddingModel
from rag.prompting import (
    build_grounded_prompt,
)
from rag.retrieval import (
    search_embeddings,
)

ARTIFACT_DIR = Path("artifacts")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    raw_chunks = json.loads((ARTIFACT_DIR / "chunks.json").read_text(encoding="utf-8"))

    chunks = [Chunk(**item) for item in raw_chunks]

    embeddings = np.load(ARTIFACT_DIR / "embeddings.npy")

    embedding_model = EmbeddingModel()

    query_embedding = embedding_model.embed_query(args.question)

    results = search_embeddings(
        query_embedding=query_embedding,
        corpus_embeddings=embeddings,
        chunks=chunks,
        top_k=args.top_k,
    )

    prompt = build_grounded_prompt(
        question=args.question,
        results=results,
    )

    print(prompt)


if __name__ == "__main__":
    main()
