import logging
from typing import List

from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LCDocument

from src.domain.entities import Document
from src.domain.ports import VectorStorePort
from src.config import get_settings

logger = logging.getLogger(__name__)


class PGVectorStore(VectorStorePort):
    """
    Adapter do banco vetorial usando PostgreSQL + extensão pgvector.
    """

    def __init__(self):
        settings = get_settings()

        if not settings.database_url:
            raise ValueError(
                "DATABASE_URL não configurada. "
                "Adicione no .env: postgresql+psycopg2://user:pass@host:5432/db"
            )

        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

        self._store = PGVector(
            connection_string=settings.database_url,
            embedding_function=self._embeddings,
            collection_name="financial_docs",
        )

        logger.info("PGVectorStore inicializado | db=%s", settings.database_url.split("@")[-1])

    # ── VectorStorePort ────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            return

        lc_docs = [
            LCDocument(page_content=doc.content, metadata=doc.metadata)
            for doc in docs
        ]
        self._store.add_documents(lc_docs)
        logger.info("Indexados %d chunks no pgvector", len(docs))

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if not query.strip():
            return []

        results = self._store.similarity_search_with_score(query, k=k)

        return [
            Document(
                content=lc_doc.page_content,
                source=lc_doc.metadata.get("source", ""),
                metadata={**lc_doc.metadata, "similarity_score": round(float(score), 4)},
            )
            for lc_doc, score in results
        ]

    def delete_collection(self) -> None:
        self._store.delete_collection()
        logger.warning("Coleção pgvector deletada")