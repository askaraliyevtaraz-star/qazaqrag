import logging
import time
from collections import Counter

from app.core.config import Settings
from rag.chunking import (
    Chunk,
    load_and_chunk_directory,
)
from rag.context import (
    ContextSource,
    build_context,
)
from rag.embeddings import EmbeddingModel
from rag.generation import (
    LLMGenerator,
    OpenAIGenerator,
    StubGenerator,
)
from rag.hybrid import (
    reciprocal_rank_fusion,
)
from rag.lexical import BM25Retriever
from rag.reranking import (
    MultilingualReranker,
)
from rag.vector_store import (
    QdrantVectorStore,
)

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

        self.chunks: list[Chunk] = []

        self.embedding_model = None
        self.bm25 = None
        self.vector_store = None
        self.reranker = None
        self.generator: LLMGenerator | None = None

        self.qdrant_points = 0

    def load(self) -> None:
        logger.info(
            "Loading corpus from %s",
            self.settings.data_dir,
        )

        self.chunks = load_and_chunk_directory(
            self.settings.data_dir,
            chunk_size=60,
            overlap=15,
        )

        logger.info(
            "Loaded %d chunks",
            len(self.chunks),
        )

        logger.info("Loading embedding model")

        self.embedding_model = EmbeddingModel()

        logger.info("Building BM25 index")

        self.bm25 = BM25Retriever(self.chunks)

        logger.info(
            "Connecting to Qdrant: %s",
            self.settings.qdrant_url,
        )

        self.vector_store = QdrantVectorStore(
            url=(self.settings.qdrant_url),
            collection_name=(self.settings.qdrant_collection),
        )

        self.qdrant_points = self.vector_store.count()

        if self.qdrant_points == 0:
            raise RuntimeError("Qdrant collection is empty. Run the indexing pipeline first.")

        logger.info(
            "Qdrant contains %d points",
            self.qdrant_points,
        )

        logger.info("Loading multilingual reranker")

        self.reranker = MultilingualReranker()

        self.generator = self._build_generator()

        logger.info("RAG service ready")

    def _build_generator(
        self,
    ) -> LLMGenerator:
        if self.settings.llm_provider == "stub":
            logger.info("Using stub LLM provider")

            return StubGenerator()

        if self.settings.llm_provider == "openai":
            if not (self.settings.openai_api_key):
                raise RuntimeError("OpenAI provider selected but API key is missing.")

            logger.info(
                "Using OpenAI model %s",
                self.settings.openai_model,
            )

            return OpenAIGenerator(
                api_key=(self.settings.openai_api_key),
                model=(self.settings.openai_model),
            )

        raise RuntimeError("Unknown LLM provider")

    @property
    def is_ready(self) -> bool:
        return (
            bool(self.chunks)
            and self.embedding_model is not None
            and self.bm25 is not None
            and self.vector_store is not None
            and self.reranker is not None
            and self.generator is not None
            and self.qdrant_points > 0
        )

    def retrieve(
        self,
        question: str,
    ):
        if not self.is_ready:
            raise RuntimeError("RAG service is not ready")

        query_embedding = self.embedding_model.embed_query(question)

        dense = self.vector_store.search(
            query_embedding=(query_embedding),
            top_k=(self.settings.retrieve_k),
        )

        lexical = self.bm25.search(
            query=question,
            top_k=(self.settings.retrieve_k),
        )

        fused = reciprocal_rank_fusion(
            dense_results=dense,
            bm25_results=lexical,
            top_k=(self.settings.rerank_k),
        )

        reranked = self.reranker.rerank(
            query=question,
            candidates=fused,
            top_k=(self.settings.final_k),
        )

        return reranked

    def query(
        self,
        question: str,
    ) -> tuple[
        str,
        list[ContextSource],
        float,
    ]:
        start = time.perf_counter()

        results = self.retrieve(question)

        context, sources = build_context(results)

        answer = self.generator.generate(
            question=question,
            context=context,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return (
            answer,
            sources,
            latency_ms,
        )

    def source_counts(
        self,
    ) -> dict[str, int]:
        return dict(Counter(chunk.source for chunk in self.chunks))
