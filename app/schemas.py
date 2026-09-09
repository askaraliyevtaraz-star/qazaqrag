from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=1000,
    )


class SourceResponse(BaseModel):
    citation_id: str
    source: str
    chunk_id: str

    score: float

    text: str


class QueryResponse(BaseModel):
    answer: str

    sources: list[SourceResponse]

    retrieval_method: str

    llm_model: str

    latency_ms: float


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool

    qdrant_ready: bool

    chunks_loaded: int

    qdrant_points: int

    llm_provider: str


class SourceFileResponse(BaseModel):
    source: str
    chunks: int
