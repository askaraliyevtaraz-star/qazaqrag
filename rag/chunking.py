from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    start_word: int
    end_word: int

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 60,
    overlap: int = 15,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()

    chunks: list[Chunk] = []

    start = 0
    chunk_number = 0

    while start < len(words):
        end = min(
            start + chunk_size,
            len(words),
        )

        chunk_words = words[start:end]

        chunk = Chunk(
            chunk_id=(f"{Path(source).stem}-{chunk_number:04d}"),
            source=source,
            text=" ".join(chunk_words),
            start_word=start,
            end_word=end,
        )

        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap
        chunk_number += 1

    return chunks


def load_and_chunk_directory(
    directory: Path,
    chunk_size: int = 60,
    overlap: int = 15,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []

    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")

        chunks = chunk_text(
            text=text,
            source=path.name,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        all_chunks.extend(chunks)

    return all_chunks
