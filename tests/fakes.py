from types import SimpleNamespace

from rag.chunking import Chunk
from rag.context import ContextSource
from rag.hybrid import (
    HybridSearchResult,
)
from rag.lexical import (
    LexicalSearchResult,
)
from rag.reranking import (
    RerankedResult,
)


class FakeGenerator:
    model_name = "fake-model"

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        del question
        del context

        return "At least five students are required [S1]."


class FakeBM25:
    def __init__(
        self,
        chunk: Chunk,
    ) -> None:
        self.chunk = chunk

    def search(
        self,
        query: str,
        top_k: int,
    ):
        del query
        del top_k

        return [
            LexicalSearchResult(
                chunk=self.chunk,
                score=10.0,
            )
        ]


class FakeReranker:
    def rerank(
        self,
        query: str,
        candidates,
        top_k: int,
    ):
        del query

        return [
            RerankedResult(
                candidate=candidate,
                rerank_score=5.0,
            )
            for candidate in candidates[:top_k]
        ]


class FakeRAGService:
    def __init__(self) -> None:
        self.is_ready = True

        self.qdrant_points = 12

        self.chunk = Chunk(
            chunk_id=("clubs_kk-0000"),
            source="clubs_kk.md",
            text=("Жаңа клубты тіркеу үшін кемінде бес студент керек."),
            start_word=0,
            end_word=9,
        )

        self.chunks = [self.chunk]

        self.settings = SimpleNamespace(llm_provider="stub")

        self.generator = FakeGenerator()

        self.bm25 = FakeBM25(self.chunk)

        self.reranker = FakeReranker()

    def retrieve(
        self,
        question: str,
    ):
        del question

        candidate = HybridSearchResult(
            chunk=self.chunk,
            rrf_score=0.03,
            dense_rank=1,
            bm25_rank=1,
            dense_score=0.9,
            bm25_score=10.0,
        )

        return [
            RerankedResult(
                candidate=candidate,
                rerank_score=5.0,
            )
        ]

    def query(
        self,
        question: str,
    ):
        del question

        return (
            "At least five students are required [S1].",
            [
                ContextSource(
                    citation_id="S1",
                    source=("clubs_kk.md"),
                    chunk_id=("clubs_kk-0000"),
                    score=5.0,
                    text=(self.chunk.text),
                )
            ],
            12.5,
        )

    def source_counts(
        self,
    ) -> dict[str, int]:
        return {
            "clubs_kk.md": 1,
        }
