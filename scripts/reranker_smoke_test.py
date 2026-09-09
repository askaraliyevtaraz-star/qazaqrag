from rag.chunking import Chunk
from rag.hybrid import (
    HybridSearchResult,
)
from rag.reranking import (
    MultilingualReranker,
)

query = "How many students are needed to create a club?"


texts = [
    ("Для сохранения стипендии GPA должен быть не ниже 3.0."),
    ("Жаңа студенттік клубты тіркеу үшін кемінде бес студент құрылтайшы болуы керек."),
    ("Residence hall quiet hours start at 23:00."),
]


candidates = []

for index, text in enumerate(texts):
    chunk = Chunk(
        chunk_id=f"test-{index}",
        source="test.md",
        text=text,
        start_word=0,
        end_word=len(text.split()),
    )

    candidates.append(
        HybridSearchResult(
            chunk=chunk,
            rrf_score=0.1,
            dense_rank=index + 1,
            bm25_rank=None,
            dense_score=None,
            bm25_score=None,
        )
    )


reranker = MultilingualReranker()


results = reranker.rerank(
    query=query,
    candidates=candidates,
    top_k=3,
)


for result in results:
    print()
    print(
        result.rerank_score,
        result.candidate.chunk.text,
    )
