from rag.retrieval import SearchResult


def build_context(
    results: list[SearchResult],
) -> str:
    parts = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        parts.append(
            "\n".join(
                [
                    f"[SOURCE {index}]",
                    (f"file={result.chunk.source}"),
                    (f"score={result.score:.4f}"),
                    result.chunk.text,
                ]
            )
        )

    return "\n\n".join(parts)


def build_grounded_prompt(
    question: str,
    results: list[SearchResult],
) -> str:
    context = build_context(results)

    return f"""
You are a knowledge-base assistant.

Answer the user's question using only
the provided context.

If the context does not contain enough
information, say that the available
documents do not provide the answer.

Cite the relevant source filenames.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
""".strip()
