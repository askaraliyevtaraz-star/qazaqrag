from dataclasses import dataclass

from rag.reranking import RerankedResult


@dataclass
class ContextSource:
    citation_id: str
    source: str
    chunk_id: str
    score: float
    text: str


def build_context(
    results: list[RerankedResult],
) -> tuple[
    str,
    list[ContextSource],
]:
    blocks = []
    sources = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        candidate = result.candidate

        citation_id = f"S{index}"

        source = ContextSource(
            citation_id=citation_id,
            source=candidate.chunk.source,
            chunk_id=(candidate.chunk.chunk_id),
            score=float(result.rerank_score),
            text=candidate.chunk.text,
        )

        sources.append(source)

        blocks.append(
            "\n".join(
                [
                    f"[{citation_id}]",
                    (f"source={source.source}"),
                    (f"chunk_id={source.chunk_id}"),
                    source.text,
                ]
            )
        )

    return "\n\n".join(blocks), sources
