import logging
from fastapi import APIRouter, HTTPException, Depends

from src.api.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from src.api.dependencies import get_rag_service, get_cache
from src.adapters.cache_adapter import RedisCacheAdapter
from src.application.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["RAG"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Faz uma pergunta aos documentos indexados",
    description="Executa o pipeline RAG completo: retrieval semântico e geração com LLM.",
)
async def ask(
    request: AskRequest,
    rag_service: RAGService = Depends(get_rag_service),
    cache: RedisCacheAdapter = Depends(get_cache),
):
    """
    Fluxo:
      1. Se use_cache=True, busca no Redis pela pergunta normalizada
      2. Se cache hit → retorna imediatamente (cached=True)
      3. Se cache miss → executa RAG completo
      4. Salva resultado no Redis com TTL configurado
      5. Retorna resposta (cached=False)
    """
    logger.info("POST /v1/ask | question='%s...'", request.question[:50])

    # ── Cache lookup ───────────────────────────────────────────────────
    if request.use_cache:
        cached_response = cache.get(request.question)
        if cached_response:
            logger.info("Cache HIT | question='%s...'", request.question[:40])
            return AskResponse(
                answer=cached_response.answer,
                sources=cached_response.sources,
                confidence=cached_response.confidence,
                cached=True,
            )

    # ── RAG pipeline ───────────────────────────────────────────────────
    try:
        response = rag_service.answer(request.question)
    except Exception as exc:
        logger.error("Erro no pipeline RAG | erro=%s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno no pipeline RAG: {str(exc)}",
        )

    # ── Persist no cache ───────────────────────────────────────────────
    if request.use_cache:
        cache.set(request.question, response)

    logger.info(
        "Cache MISS → RAG executado | confidence=%.2f | sources=%d",
        response.confidence, len(response.sources),
    )

    return AskResponse(
        answer=response.answer,
        sources=response.sources,
        confidence=response.confidence,
        cached=False,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Indexa um PDF no banco vetorial",
    description="Carrega, divide em chunks e indexa um arquivo PDF no Chroma.",
)
async def ingest(
    request: IngestRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    logger.info("POST /v1/ingest | file=%s", request.file_path)

    result = rag_service.ingest(request.file_path)

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=result.error or "Falha na ingestão do documento",
        )

    return IngestResponse(
        file_path=result.file_path,
        chunks_created=result.chunks_created,
        success=result.success,
    )