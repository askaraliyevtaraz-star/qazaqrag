import argparse
from pathlib import Path

from rag.chunking import (
    load_and_chunk_directory,
)
from rag.embeddings import EmbeddingModel
from rag.vector_store import (
    QdrantVectorStore,
)

DATA_DIR = Path("data/demo")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--recreate",
        action="store_true",
    )

    args = parser.parse_args()

    chunks = load_and_chunk_directory(
        DATA_DIR,
        chunk_size=60,
        overlap=15,
    )

    print(f"Loaded {len(chunks)} chunks")

    model = EmbeddingModel()

    embeddings = model.embed_documents([chunk.text for chunk in chunks])

    print(
        "Embeddings shape:",
        embeddings.shape,
    )

    vector_size = int(embeddings.shape[1])

    store = QdrantVectorStore()

    store.ensure_collection(
        vector_size=vector_size,
        recreate=args.recreate,
    )

    store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(
        "Qdrant points:",
        store.count(),
    )


if __name__ == "__main__":
    main()
