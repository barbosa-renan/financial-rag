import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
from typing import List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LCDocument

from src.domain.entities import Document
from src.domain.ports import VectorStorePort
from src.config import get_settings

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStorePort):
    """
    Adapter do banco vetorial usando Chroma.
    """

    def __init__(self):
        settings = get_settings()

        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

        self._store = Chroma(
            collection_name="financial_docs",
            embedding_function=self._embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

        logger.info(
            "ChromaVectorStore inicializado | dir=%s | docs_indexados=%d",
            settings.chroma_persist_dir,
            self._store._collection.count(),
        )

    # ── VectorStorePort ────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> None:
        """
        Indexa chunks no Chroma.
        Cada chunk é convertido em embedding pela OpenAI antes de ser persistido.
        """
        if not docs:
            logger.warning("add_documents chamado com lista vazia")
            return

        # Converte entidades do domínio para o formato que o LangChain espera
        lc_docs = [
            LCDocument(
                page_content=doc.content,
                metadata=doc.metadata,
            )
            for doc in docs
        ]

        self._store.add_documents(lc_docs)
        logger.info("Indexados %d chunks no Chroma", len(docs))

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Busca semântica: converte query em embedding e retorna os k chunks mais próximos.
        O score retornado é distância L2 — quanto MENOR, mais similar.
        Convertemos para confidence: 1 - score (normalizado).
        """
        if not query.strip():
            return []

        results = self._store.similarity_search_with_score(query, k=k)

        documents = []
        for lc_doc, score in results:
            doc = Document(
                content=lc_doc.page_content,
                source=lc_doc.metadata.get("source", ""),
                metadata={
                    **lc_doc.metadata,
                    "similarity_score": round(float(score), 4),
                },
            )
            documents.append(doc)

        logger.debug(
            "similarity_search | query='%s...' | k=%d | scores=%s",
            query[:40],
            k,
            [d.metadata["similarity_score"] for d in documents],
        )

        return documents

    def delete_collection(self) -> None:
        """
        Remove todos os documentos da coleção.
        Use quando mudar chunk_size ou embedding_model — o índice precisa ser
        recriado do zero porque os vetores são incompatíveis entre modelos.
        """
        self._store.delete_collection()
        logger.warning("Coleção 'financial_docs' deletada — reindexação necessária")

    def count(self) -> int:
        """Retorna quantos chunks estão indexados. Útil para monitorar crescimento e validar delete_collection."""
        return self._store._collection.count()