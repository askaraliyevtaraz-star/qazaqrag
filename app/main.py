import logging
from contextlib import (
    asynccontextmanager,
)

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import (
    Settings,
    get_settings,
)
from app.core.logging import (
    configure_logging,
)
from app.services.rag_service import (
    RAGService,
)


def create_app(
    settings: Settings | None = None,
    service_override=None,
) -> FastAPI:
    settings = settings or get_settings()

    configure_logging(settings.log_level)

    logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        logger.info("Starting QazaqRAG")

        if service_override is not None:
            service = service_override

        else:
            service = RAGService(settings)

            service.load()

        app.state.rag_service = service

        yield

        logger.info("Stopping QazaqRAG")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.include_router(router)

    return app


app = create_app()
