from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_provider: str = "openai"   # "openai" ou "anthropic"

    # Vector store
    chroma_persist_dir: str = "./data/chroma_db"
    database_url: str = ""           # pgvector quando em produção
    collection_name: str = "financial_docs"

    # Cache
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 100
    retriever_k: int = 4


@lru_cache
def get_settings() -> Settings:
    """
    Singleton via lru_cache — instancia Settings uma única vez.
    Em testes, basta chamar get_settings.cache_clear() para resetar.
    """
    return Settings()
