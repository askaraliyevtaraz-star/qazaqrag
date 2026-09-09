from dataclasses import dataclass

import bm25s
from bm25s.tokenization import Tokenizer

from rag.chunking import Chunk


@dataclass
class LexicalSearchResult:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
    ) -> None:
        if not chunks:
            raise ValueError("BM25 corpus cannot be empty")

        self.chunks = chunks

        self.tokenizer = Tokenizer()

        corpus = [chunk.text.lower() for chunk in chunks]

        corpus_tokens = self.tokenizer.tokenize(corpus)

        self.retriever = bm25s.BM25(method="lucene")

        self.retriever.index(corpus_tokens)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[LexicalSearchResult]:
        k = min(
            top_k,
            len(self.chunks),
        )

        query_tokens = self.tokenizer.tokenize(
            [query.lower()],
            update_vocab=False,
        )

        indices, scores = self.retriever.retrieve(
            query_tokens,
            k=k,
        )

        results = []

        for index, score in zip(
            indices[0],
            scores[0],
            strict=True,
        ):
            chunk_index = int(index)

            results.append(
                LexicalSearchResult(
                    chunk=self.chunks[chunk_index],
                    score=float(score),
                )
            )

        return results
