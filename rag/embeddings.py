import numpy as np
from sentence_transformers import (
    SentenceTransformer,
)

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.model_name = model_name

        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        prepared = [f"passage: {text}" for text in texts]

        embeddings = self.model.encode(
            prepared,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        prepared = f"query: {query}"

        embedding = self.model.encode(
            prepared,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )
