"""
Testes do adapter PDFDocumentLoader.

Estratégia: não mockar o splitter — testar o comportamento real de chunking
usando texto em memória. Mocks de I/O apenas onde necessário (arquivo inexistente).
"""
import os
import tempfile
import pytest

from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.domain.entities import Document
from src.config import get_settings


# ── Helpers ────────────────────────────────────────────────────────────────────

REGULAMENTO_TEXTO = """
Resolução Conjunta nº 1 — Open Finance

Art. 1º O presente regulamento estabelece as diretrizes para o compartilhamento
de dados no âmbito do Sistema Financeiro Aberto.

Art. 2º Para os fins desta resolução, consideram-se dados cadastrais as
informações de identificação do cliente, incluindo nome, CPF, endereço e contato.

Art. 3º O prazo máximo de vigência do consentimento é de 12 meses, podendo
ser revogado a qualquer tempo pelo titular dos dados, em conformidade com a LGPD.

Art. 4º A instituição transmissora é responsável por validar a autenticidade
do consentimento antes de qualquer compartilhamento de informações financeiras.

Art. 5º O compartilhamento de dados de pagamento via Pix deve respeitar os
limites estabelecidos pelo Banco Central e ocorrer em no máximo 10 segundos.
""".strip()


def make_splitter(chunk_size=800, overlap=100):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def text_to_domain_docs(text: str, source: str = "test.pdf", **splitter_kwargs) -> list[Document]:
    """Converte texto diretamente em Document sem passar por PyPDFLoader."""
    splitter = make_splitter(**splitter_kwargs)
    chunks = splitter.split_text(text)
    return [
        Document(
            content=chunk,
            source=source,
            metadata={"page": i, "source": source, "file_name": os.path.basename(source)},
        )
        for i, chunk in enumerate(chunks)
    ]


# ── Testes de chunking puro ────────────────────────────────────────────────────

class TestChunkingStrategy:

    def test_chunks_respeitam_chunk_size(self):
        """Nenhum chunk deve ultrapassar chunk_size (com tolerância de separadores)."""
        docs = text_to_domain_docs(REGULAMENTO_TEXTO, chunk_size=400, overlap=50)
        oversized = [d for d in docs if len(d.content) > 450]  # 12% de tolerância
        assert oversized == [], (
            f"{len(oversized)} chunks acima do limite: "
            f"{[len(d.content) for d in oversized]}"
        )

    def test_overlap_preserva_contexto_entre_chunks(self):
        """
        Com overlap > 0, uma frase próxima à borda de um chunk
        deve aparecer também no início do próximo.
        """
        # Texto longo o suficiente para gerar múltiplos chunks pequenos
        long_text = "\n\n".join([REGULAMENTO_TEXTO] * 3)
        docs = text_to_domain_docs(long_text, chunk_size=300, overlap=80)

        if len(docs) < 2:
            pytest.skip("Texto não gerou chunks suficientes para testar overlap")

        # Pega as últimas palavras do chunk N e verifica se aparecem no chunk N+1
        for i in range(len(docs) - 1):
            tail = docs[i].content[-60:].strip()
            next_content = docs[i + 1].content

            # Pelo menos parte do final do chunk anterior deve estar no próximo
            tail_words = tail.split()[-5:]  # últimas 5 palavras
            overlap_found = any(w in next_content for w in tail_words if len(w) > 3)
            if overlap_found:
                return  # basta um par confirmar o comportamento

        pytest.fail("Nenhum par de chunks consecutivos apresentou overlap de conteúdo")

    def test_chunks_nao_vazios(self):
        docs = text_to_domain_docs(REGULAMENTO_TEXTO)
        assert all(len(d.content.strip()) > 0 for d in docs), "Chunks vazios encontrados"

    def test_conteudo_total_preservado(self):
        """A soma de todos os chunks deve conter todas as palavras-chave do texto."""
        docs = text_to_domain_docs(REGULAMENTO_TEXTO)
        all_content = " ".join(d.content for d in docs)

        keywords = ["Open Finance", "consentimento", "LGPD", "Pix", "Banco Central"]
        for kw in keywords:
            assert kw in all_content, f"Palavra-chave '{kw}' perdida no chunking"

    def test_chunk_size_menor_gera_mais_chunks(self):
        docs_small = text_to_domain_docs(REGULAMENTO_TEXTO, chunk_size=200, overlap=20)
        docs_large = text_to_domain_docs(REGULAMENTO_TEXTO, chunk_size=800, overlap=100)
        assert len(docs_small) > len(docs_large), (
            "Chunks menores devem gerar mais fragmentos"
        )

    def test_metadata_preservada(self):
        docs = text_to_domain_docs(REGULAMENTO_TEXTO, source="regulamento_openfinance.pdf")
        for doc in docs:
            assert doc.source == "regulamento_openfinance.pdf"
            assert "file_name" in doc.metadata
            assert doc.metadata["file_name"] == "regulamento_openfinance.pdf"


# ── Testes de Document entity ──────────────────────────────────────────────────

class TestDocumentEntity:

    def test_summary_inclui_fonte_e_pagina(self):
        doc = Document(
            content="O prazo de consentimento é de 12 meses conforme LGPD.",
            source="resolucao_001.pdf",
            metadata={"page": 7},
        )
        summary = doc.summary()
        assert "resolucao_001.pdf" in summary
        assert "pág. 7" in summary

    def test_summary_preview_truncado(self):
        doc = Document(content="x" * 300, source="doc.pdf", metadata={})
        assert len(doc.summary()) < 200

    def test_chunks_financeiros_tem_tamanho_adequado(self):
        """
        Para compliance, chunks muito pequenos (<100 chars) são ruído;
        chunks muito grandes (>1000 chars) perdem precisão no retrieval.
        """
        docs = text_to_domain_docs(REGULAMENTO_TEXTO, chunk_size=800, overlap=100)
        for doc in docs:
            # Ignora chunks finais que naturalmente são menores
            if doc == docs[-1]:
                continue
            assert len(doc.content) >= 80, f"Chunk suspeito (muito pequeno): '{doc.content[:50]}'"


# ── Testes de IngestResult ─────────────────────────────────────────────────────

class TestIngestResult:

    def test_arquivo_inexistente_retorna_falha(self):
        from src.adapters.document_loader import PDFDocumentLoader
        loader = PDFDocumentLoader()
        result = loader.load_pdf("/caminho/inexistente/arquivo.pdf")

        assert result.success is False
        assert result.chunks_created == 0
        assert result.error is not None
        assert "não encontrado" in result.error.lower()

    def test_load_directory_vazio(self, tmp_path):
        from src.adapters.document_loader import PDFDocumentLoader
        loader = PDFDocumentLoader()
        docs, results = loader.load_directory(str(tmp_path))

        assert docs == []
        assert results == []
