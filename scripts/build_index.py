import json
from pathlib import Path

import numpy as np

from rag.chunking import (
    load_and_chunk_directory,
)
from rag.embeddings import EmbeddingModel

DATA_DIR = Path("data/demo")

ARTIFACT_DIR = Path("artifacts")

EMBEDDINGS_PATH = ARTIFACT_DIR / "embeddings.npy"

CHUNKS_PATH = ARTIFACT_DIR / "chunks.json"


def main() -> None:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = load_and_chunk_directory(
        DATA_DIR,
        chunk_size=60,
        overlap=15,
    )

    print(f"Loaded {len(chunks)} chunks")

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])

    print(
        "Embeddings shape:",
        embeddings.shape,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    CHUNKS_PATH.write_text(
        json.dumps(
            [chunk.to_dict() for chunk in chunks],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Saved:",
        EMBEDDINGS_PATH,
    )

    print(
        "Saved:",
        CHUNKS_PATH,
    )


if __name__ == "__main__":
    main()
