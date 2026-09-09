from rag.chunking import Chunk
from rag.vector_store import (
    infer_language,
    point_id_from_chunk,
)


def test_language_from_filename() -> None:
    assert infer_language("clubs_kk.md") == "kk"

    assert infer_language("library_ru.md") == "ru"

    assert infer_language("housing_en.md") == "en"


def test_point_id_is_deterministic() -> None:
    chunk = Chunk(
        chunk_id="library-0001",
        source="library_ru.md",
        text="Example",
        start_word=0,
        end_word=1,
    )

    first = point_id_from_chunk(chunk)

    second = point_id_from_chunk(chunk)

    assert first == second
