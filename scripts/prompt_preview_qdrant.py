import argparse

from rag.embeddings import EmbeddingModel
from rag.prompting import (
    build_grounded_prompt,
)
from rag.vector_store import (
    QdrantVectorStore,
)


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

    parser.add_argument(
        "--language",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    model = EmbeddingModel()

    query_embedding = model.embed_query(args.question)

    store = QdrantVectorStore()

    results = store.search(
        query_embedding=query_embedding,
        top_k=args.top_k,
        language=args.language,
    )

    prompt = build_grounded_prompt(
        question=args.question,
        results=results,
    )

    print(prompt)


if __name__ == "__main__":
    main()
