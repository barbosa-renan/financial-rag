import logging
from functools import lru_cache
from src.adapters.cache_adapter import RedisCacheAdapter
from src.adapters.chroma_vector_store import ChromaVectorStore
from src.adapters.llm_adapter import create_llm_adapter
from src.application.rag_service import RAGService

logger = logging.getLogger(__name__)


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    """
    Singleton do ChromaVectorStore.
    lru_cache garante uma única instância por processo
    evita reabrir a conexão com o Chroma a cada request.
    """
    logger.info("Inicializando ChromaVectorStore...")
    return ChromaVectorStore()


@lru_cache
def get_llm():
    """
    Singleton do LLM adapter.
    Decide entre OpenAI e Anthropic baseado no .env (LLM_PROVIDER).
    """
    logger.info("Inicializando LLM adapter...")
    return create_llm_adapter()


@lru_cache
def get_cache() -> RedisCacheAdapter:
    """
    Singleton do cache Redis.
    Se Redis estiver fora do ar, retorna adapter em modo degradado
    (sem cache) em vez de falhar na inicialização.
    """
    logger.info("Inicializando RedisCacheAdapter...")
    return RedisCacheAdapter()


def get_rag_service() -> RAGService:
    """
    RAGService com dependências injetadas.
    Não usa lru_cache — é stateless, pode ser recriado a cada request.
    As dependências pesadas (vector_store, llm) já são singletons acima.
    """
    return RAGService(
        vector_store=get_vector_store(),
        llm=get_llm(),
    )