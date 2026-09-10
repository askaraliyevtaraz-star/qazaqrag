from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "QazaqRAG API"
    app_version: str = "0.12.0"

    data_dir: Path = Path("data/demo")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "qazaqrag_chunks"

    agent_max_retrieval_attempts: int = 2

    min_reranker_score: float | None = None

    retrieve_k: int = 10
    rerank_k: int = 8
    final_k: int = 3

    llm_provider: Literal[
        "stub",
        "openai",
    ] = "stub"

    openai_api_key: str | None = None
    openai_model: str = "gpt-5.6-luna"

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="QAZAQRAG_",
        env_file=".env",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
