import logging
from typing import List
from src.domain.entities import Document, RAGResponse, IngestResult
from src.domain.ports import VectorStorePort, LLMPort, RAGServicePort
from src.config import get_settings

logger = logging.getLogger(__name__)


class RAGService(RAGServicePort):
    """
    Caso de uso principal — orquestra o pipeline RAG completo.
    """

    def __init__(self, vector_store: VectorStorePort, llm: LLMPort):
        self._vector_store = vector_store
        self._llm = llm
        self._settings = get_settings()
        logger.info("RAGService inicializado")

    # ── RAGServicePort ─────────────────────────────────────────────────

    def answer(self, question: str) -> RAGResponse:
        """
        Pipeline RAG completo
        """
        logger.info("RAG answer | question='%s...'", question[:60])

        # ── Etapa 1: Retrieve ──────────────────────────────────────────
        relevant_docs = self._vector_store.similarity_search(
            query=question,
            k=self._settings.retriever_k,
        )

        if not relevant_docs:
            logger.warning("Nenhum chunk relevante encontrado para: '%s'", question)
            return RAGResponse(
                answer="Não encontrei informações relevantes nos documentos disponíveis.",
                sources=[],
                confidence=0.0,
            )

        # ── Etapa 2: Augment ───────────────────────────────────────────
        context = self._build_context(relevant_docs)

        # ── Etapa 3: Generate ──────────────────────────────────────────
        answer = self._llm.generate(question=question, context=context)

        # ── Montar resposta com rastreabilidade ────────────────────────
        sources = list(dict.fromkeys(doc.source for doc in relevant_docs))  # preserva ordem, sem duplicatas
        confidence = self._calculate_confidence(relevant_docs)

        logger.info(
            "RAG concluído | sources=%d | confidence=%.2f | answer_chars=%d",
            len(sources), confidence, len(answer),
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
        )

    def ingest(self, file_path: str) -> IngestResult:
        """
        Ingere um PDF no banco vetorial.
        Combina o document_loader com o vector_store.
        """
        from src.adapters.document_loader import PDFDocumentLoader

        loader = PDFDocumentLoader()
        result = loader.load_pdf(file_path)

        if not result.success:
            return result

        docs = loader.get_last_documents()
        self._vector_store.add_documents(docs)

        logger.info(
            "Ingestão concluída | arquivo=%s | chunks=%d",
            file_path, result.chunks_created,
        )
        return result

    # ── Helpers privados ───────────────────────────────────────────────

    def _build_context(self, docs: List[Document]) -> str:
        """
        Monta o contexto que será injetado no prompt.
        """
        parts = []
        for doc in docs:
            source = doc.metadata.get("file_name", doc.source)
            page = doc.metadata.get("page", "?")
            header = f"[Fonte: {source} | Página: {page}]"
            parts.append(f"{header}\n{doc.content}")

        return "\n\n---\n\n".join(parts)

    def _calculate_confidence(self, docs: List[Document]) -> float:
        """
        Converte o similarity_score (distância L2, menor = melhor)
        em um valor de confiança entre 0.0 e 1.0.

        Usa apenas o chunk mais relevante (primeiro da lista)
        como referência — se o melhor match já está distante,
        a resposta como um todo tem baixa confiança.
        """
        scores = [
            doc.metadata.get("similarity_score", 1.0)
            for doc in docs
            if "similarity_score" in doc.metadata
        ]

        if not scores:
            return 0.5  # confiança neutra quando score não disponível

        best_score = min(scores)  # menor distância = mais similar
        # Normalização simples: score 0.0 → confidence 1.0, score 1.0+ → próximo de 0
        confidence = max(0.0, 1.0 - best_score)
        return round(confidence, 2)