from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)

from app.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    SourceFileResponse,
    SourceResponse,
)
from app.services.rag_service import (
    RAGService,
)

router = APIRouter()


def get_service(
    request: Request,
) -> RAGService:
    return request.app.state.rag_service


@router.get("/")
def root() -> dict[str, str]:
    return {"message": ("QazaqRAG API is running")}


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
)
def ready(
    request: Request,
) -> ReadyResponse:
    service = get_service(request)

    return ReadyResponse(
        ready=service.is_ready,
        qdrant_ready=(service.qdrant_points > 0),
        chunks_loaded=len(service.chunks),
        qdrant_points=(service.qdrant_points),
        llm_provider=(service.settings.llm_provider),
    )


@router.get(
    "/sources",
    response_model=list[SourceFileResponse],
)
def sources(
    request: Request,
) -> list[SourceFileResponse]:
    service = get_service(request)

    counts = service.source_counts()

    return [
        SourceFileResponse(
            source=source,
            chunks=count,
        )
        for source, count in sorted(counts.items())
    ]


@router.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:
    service = get_service(request)

    if not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail=("RAG service is not ready"),
        )

    try:
        (
            answer,
            sources,
            latency_ms,
        ) = service.query(payload.question)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=("Failed to process query"),
        ) from exc

    return QueryResponse(
        answer=answer,
        sources=[
            SourceResponse(
                citation_id=(source.citation_id),
                source=source.source,
                chunk_id=(source.chunk_id),
                score=source.score,
                text=source.text,
            )
            for source in sources
        ],
        retrieval_method=("dense+bm25+rrf+reranker"),
        llm_model=(service.generator.model_name),
        latency_ms=latency_ms,
    )
