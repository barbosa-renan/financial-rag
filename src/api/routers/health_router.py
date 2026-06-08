import logging
from fastapi import APIRouter
from src.api.schemas import HealthResponse
from src.api.dependencies import get_vector_store, get_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Infra"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Healthcheck da aplicação",
    description="Verifica disponibilidade do banco vetorial e do cache.",
)
async def health():
    
    results = {"vector_store": "ok", "cache": "ok"}

    # Verifica Chroma
    try:
        store = get_vector_store()
        count = store.count()
        results["vector_store"] = f"ok ({count} chunks indexados)"
    except Exception as exc:
        logger.error("Healthcheck: Chroma com problema | %s", exc)
        results["vector_store"] = "error"

    # Verifica Redis
    try:
        cache = get_cache()
        results["cache"] = "ok" if cache.is_available() else "unavailable"
    except Exception as exc:
        logger.warning("Healthcheck: Redis indisponível | %s", exc)
        results["cache"] = "unavailable"

    # Status geral: degraded se Redis fora, error se Chroma fora
    overall = "healthy"
    if results["cache"] == "unavailable":
        overall = "degraded"
    if results["vector_store"] == "error":
        overall = "error"

    return HealthResponse(
        status=overall,
        vector_store=results["vector_store"],
        cache=results["cache"],
        version="1.0.0",
    )