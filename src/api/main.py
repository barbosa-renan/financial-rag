import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.logging_config import setup_logging
from src.api.routers import rag_router, health_router
from src.api.dependencies import get_vector_store, get_llm, get_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
   
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Iniciando Financial RAG API...")

    try:
        get_vector_store()
        get_llm()
        get_cache()
        logger.info("Dependências inicializadas com sucesso")
    except Exception as exc:
        logger.error("Falha ao inicializar dependências | %s", exc)

    yield  

    logger.info("Encerrando Financial RAG API...")


app = FastAPI(
    title="Financial RAG API",
    description="Assistente de documentos financeiros com LLM e busca semântica.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(rag_router.router)
app.include_router(health_router.router)