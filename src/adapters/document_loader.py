import logging
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.domain.entities import Document, IngestResult
from src.config import get_settings

logger = logging.getLogger(__name__)


class PDFDocumentLoader:
    """
    Adapter de entrada responsável por carregar e dividir PDFs em chunks.

    Por que RecursiveCharacterTextSplitter?
    ─────────────────────────────────────────
    Tenta dividir nesta ordem de prioridade:
      1. parágrafos  (\n\n)  — preserva unidades semânticas completas
      2. linhas      (\n)
      3. frases      (". ")
      4. palavras    (" ")

    Só avança para o próximo separador se o chunk ainda estiver
    acima de chunk_size. Resultado: chunks coesos, raramente cortados
    no meio de uma ideia.

    Para documentos financeiros/regulatórios, isso é crítico:
    um artigo cortado no meio muda completamente seu significado legal.
    """

    def __init__(self):
        settings = get_settings()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,          # usa caracteres, não tokens
            is_separator_regex=False,
        )

    def load_pdf(self, path: str) -> IngestResult:
        """
        Carrega um único PDF e retorna IngestResult com os chunks.
        Erros são capturados e devolvidos como IngestResult(success=False)
        em vez de explodir — importante num pipeline de ingestão em lote.
        """
        if not os.path.exists(path):
            return IngestResult(
                file_path=path,
                chunks_created=0,
                success=False,
                error=f"Arquivo não encontrado: {path}",
            )

        try:
            raw_pages = PyPDFLoader(path).load()
            chunks = self.splitter.split_documents(raw_pages)
            documents = [self._to_domain(chunk, path) for chunk in chunks]

            logger.info(
                "PDF carregado | arquivo=%s | páginas=%d | chunks=%d | "
                "chunk_size_médio=%.0f chars",
                os.path.basename(path),
                len(raw_pages),
                len(chunks),
                sum(len(d.content) for d in documents) / max(len(documents), 1),
            )

            self._last_documents = documents

            return IngestResult(
                file_path=path,
                chunks_created=len(documents),
                success=True,
            )

        except Exception as exc:
            logger.error("Falha ao carregar PDF | arquivo=%s | erro=%s", path, exc)
            return IngestResult(
                file_path=path,
                chunks_created=0,
                success=False,
                error=str(exc),
            )

    def load_directory(self, dir_path: str) -> tuple[List[Document], List[IngestResult]]:
        """
        Carrega todos os PDFs de um diretório.
        Retorna (documentos_agregados, resultados_por_arquivo).
        Continua mesmo que um arquivo falhe.
        """
        pdf_files = [
            os.path.join(dir_path, f)
            for f in sorted(os.listdir(dir_path))
            if f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            logger.warning("Nenhum PDF encontrado em: %s", dir_path)
            return [], []

        all_docs: List[Document] = []
        results: List[IngestResult] = []

        for path in pdf_files:
            result = self.load_pdf(path)
            results.append(result)
            if result.success:
                all_docs.extend(self._last_documents)

        total_chunks = sum(r.chunks_created for r in results)
        failures = [r for r in results if not r.success]

        logger.info(
            "Ingestão concluída | arquivos=%d | falhas=%d | total_chunks=%d",
            len(pdf_files),
            len(failures),
            total_chunks,
        )

        return all_docs, results

    def get_last_documents(self) -> List[Document]:
        """Retorna os chunks do último load_pdf() chamado."""
        return getattr(self, "_last_documents", [])

    def _to_domain(self, lc_doc, source_path: str) -> Document:
        """Converte Document do LangChain para nossa entidade de domínio."""
        return Document(
            content=lc_doc.page_content,
            source=source_path,
            metadata={
                "page": lc_doc.metadata.get("page", 0),
                "source": source_path,
                "file_name": os.path.basename(source_path),
                # total_pages vem do PyPDFLoader quando disponível
                "total_pages": lc_doc.metadata.get("total_pages", "?"),
            },
        )
