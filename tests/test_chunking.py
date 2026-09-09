import pytest

from rag.chunking import chunk_text


def test_chunking_creates_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(100))

    chunks = chunk_text(
        text=text,
        source="test.md",
        chunk_size=40,
        overlap=10,
    )

    assert len(chunks) > 1

    first_words = chunks[0].text.split()

    second_words = chunks[1].text.split()

    assert first_words[-10:] == second_words[:10]


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValueError):
        chunk_text(
            text="test",
            source="test.md",
            chunk_size=10,
            overlap=10,
        )
