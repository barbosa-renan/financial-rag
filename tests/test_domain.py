"""
Testes do domain — sem dependência externa nenhuma.
Rodam em milissegundos, sem precisar de OpenAI, Redis ou Chroma.
Isso é o hexagonal funcionando: o domínio é testável isoladamente.
"""
import pytest
from src.domain.entities import Document, RAGResponse, IngestResult


class TestDocument:
    def test_summary_truncates_content(self):
        doc = Document(
            content="A" * 200,
            source="regulamento_pix.pdf",
            metadata={"page": 3}
        )
        summary = doc.summary()
        assert "regulamento_pix.pdf" in summary
        assert "pág. 3" in summary
        assert len(summary) < 150

    def test_summary_without_page_metadata(self):
        doc = Document(content="texto qualquer", source="doc.pdf")
        assert "pág. ?" in doc.summary()


class TestRAGResponse:
    def test_is_grounded_with_sources(self):
        resp = RAGResponse(
            answer="O prazo é 10 segundos.",
            sources=["regulamento_pix.pdf"],
            confidence=0.92
        )
        assert resp.is_grounded() is True

    def test_is_not_grounded_without_sources(self):
        resp = RAGResponse(
            answer="Não sei.",
            sources=[],
            confidence=0.1
        )
        assert resp.is_grounded() is False

    def test_cached_defaults_to_false(self):
        resp = RAGResponse(answer="x", sources=[], confidence=0.5)
        assert resp.cached is False


class TestIngestResult:
    def test_success_result(self):
        result = IngestResult(
            file_path="doc.pdf",
            chunks_created=42,
            success=True
        )
        assert result.error is None
        assert result.chunks_created == 42

    def test_failure_result(self):
        result = IngestResult(
            file_path="corrompido.pdf",
            chunks_created=0,
            success=False,
            error="PDF inválido ou corrompido"
        )
        assert result.success is False
        assert result.error is not None
