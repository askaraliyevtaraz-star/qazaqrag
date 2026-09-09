from types import SimpleNamespace

from rag.context import ContextSource


class FakeRAGService:
    def __init__(self) -> None:
        self.is_ready = True

        self.qdrant_points = 12

        self.chunks = [
            SimpleNamespace(source="clubs_kk.md"),
            SimpleNamespace(source="library_ru.md"),
        ]

        self.settings = SimpleNamespace(llm_provider="stub")

        self.generator = SimpleNamespace(model_name="fake-model")

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
                    score=8.5,
                    text=("кемінде бес студент..."),
                )
            ],
            12.5,
        )

    def source_counts(
        self,
    ) -> dict[str, int]:
        return {
            "clubs_kk.md": 1,
            "library_ru.md": 1,
        }
