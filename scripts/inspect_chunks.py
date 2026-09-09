from pathlib import Path

from rag.chunking import (
    load_and_chunk_directory,
)

chunks = load_and_chunk_directory(
    Path("data/demo"),
    chunk_size=60,
    overlap=15,
)


print(
    "Total chunks:",
    len(chunks),
)


for chunk in chunks:
    print()
    print("=" * 70)

    print(
        chunk.chunk_id,
        "|",
        chunk.source,
    )

    print(f"words: {chunk.start_word}-{chunk.end_word}")

    print()

    print(chunk.text)
