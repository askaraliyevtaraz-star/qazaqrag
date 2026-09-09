import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from rag.chunking import Chunk
from rag.retrieval import SearchResult

# DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

DEFAULT_COLLECTION_NAME = "qazaqrag_chunks"


def infer_language(
    source: str,
) -> str:
    stem = Path(source).stem

    suffix = stem.rsplit(
        "_",
        maxsplit=1,
    )[-1]

    if suffix in {
        "ru",
        "kk",
        "en",
    }:
        return suffix

    return "unknown"


def point_id_from_chunk(
    chunk: Chunk,
) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"qazaqrag:{chunk.chunk_id}",
        )
    )


class QdrantVectorStore:
    def __init__(
        self,
        url: str = DEFAULT_QDRANT_URL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.client = QdrantClient(url=url)

        self.collection_name = collection_name

    def ensure_collection(
        self,
        vector_size: int,
        recreate: bool = False,
    ) -> None:
        exists = self.client.collection_exists(self.collection_name)

        if exists and recreate:
            self.client.delete_collection(self.collection_name)

            exists = False

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="source",
            field_schema=(models.PayloadSchemaType.KEYWORD),
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="language",
            field_schema=(models.PayloadSchemaType.KEYWORD),
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have equal length")

        points = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            payload = {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "language": infer_language(chunk.source),
                "text": chunk.text,
                "start_word": (chunk.start_word),
                "end_word": (chunk.end_word),
            }

            points.append(
                models.PointStruct(
                    id=point_id_from_chunk(chunk),
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        language: str | None = None,
        source: str | None = None,
        score_threshold: float | None = None,
        exact: bool = False,
    ) -> list[SearchResult]:
        conditions = []

        if language is not None:
            conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=language),
                )
            )

        if source is not None:
            conditions.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source),
                )
            )

        query_filter = None

        if conditions:
            query_filter = models.Filter(must=conditions)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            search_params=models.SearchParams(exact=exact),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            chunk = Chunk(
                chunk_id=str(payload["chunk_id"]),
                source=str(payload["source"]),
                text=str(payload["text"]),
                start_word=int(payload["start_word"]),
                end_word=int(payload["end_word"]),
            )

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(point.score),
                )
            )

        return results

    def count(self) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count
