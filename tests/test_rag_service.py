"""
Testes do RAGService usando mocks.
Não fazem chamadas reais à OpenAI nem ao Chroma.
Testam a lógica de orquestração: montagem de contexto,
cálculo de confidence, deduplicação de sources, fallback para índice vazio.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.domain.entities import Document, RAGResponse
from src.application.rag_service import RAGService


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_document(content: str, source: str, page: int, score: float = 0.2) -> Document:
    """Helper: cria Document com similarity_score no metadata."""
    return Document(
        content=content,
        source=source,
        metadata={
            "source": source,
            "file_name": source.split("/")[-1],
            "page": page,
            "similarity_score": score,
        },
    )


@pytest.fixture
def mock_vector_store():
    """VectorStore simulado — controla o que similarity_search retorna."""
    store = MagicMock()
    store.similarity_search.return_value = [
        make_document(
            content="O prazo de consentimento no Open Finance é de 12 meses.",
            source="resolucao_001.pdf",
            page=3,
            score=0.12,
        ),
        make_document(
            content="O consentimento pode ser revogado a qualquer tempo pelo titular.",
            source="resolucao_001.pdf",
            page=4,
            score=0.18,
        ),
        make_document(
            content="A LGPD exige consentimento específico e granular do titular.",
            source="lgpd_resumo.pdf",
            page=11,
            score=0.25,
        ),
    ]
    return store


@pytest.fixture
def mock_llm():
    """LLM simulado — retorna resposta fixa sem chamar OpenAI."""
    llm = MagicMock()
    llm.generate.return_value = (
        "O prazo de consentimento no Open Finance é de 12 meses, "
        "conforme Resolução Conjunta nº 1 (pág. 3), podendo ser revogado "
        "a qualquer tempo pelo titular, em conformidade com a LGPD."
    )
    return llm


@pytest.fixture
def rag_service(mock_vector_store, mock_llm):
    """RAGService com dependências mockadas."""
    return RAGService(vector_store=mock_vector_store, llm=mock_llm)


# ── Testes de answer() ─────────────────────────────────────────────────────────

class TestRAGServiceAnswer:

    def test_retorna_rag_response(self, rag_service):
        """answer() deve retornar instância de RAGResponse."""
        result = rag_service.answer("Qual o prazo de consentimento no Open Finance?")
        assert isinstance(result, RAGResponse)

    def test_chama_similarity_search_com_a_pergunta(self, rag_service, mock_vector_store):
        """O vector_store deve ser chamado com a pergunta exata do usuário."""
        question = "Qual o prazo de consentimento no Open Finance?"
        rag_service.answer(question)

        mock_vector_store.similarity_search.assert_called_once()
        call_args = mock_vector_store.similarity_search.call_args
        assert call_args.kwargs["query"] == question

    def test_chama_llm_com_contexto_e_pergunta(self, rag_service, mock_llm):
        """O LLM deve receber question e context preenchidos."""
        question = "Qual o prazo de consentimento?"
        rag_service.answer(question)

        mock_llm.generate.assert_called_once()
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert call_kwargs["question"] == question
        assert len(call_kwargs["context"]) > 0

    def test_contexto_contem_fonte_e_pagina(self, rag_service, mock_llm):
        """O contexto enviado ao LLM deve identificar fonte e página de cada chunk."""
        rag_service.answer("Qual o prazo de consentimento?")

        context = mock_llm.generate.call_args.kwargs["context"]
        assert "resolucao_001.pdf" in context
        assert "Página: 3" in context
        assert "Página: 4" in context

    def test_sources_sem_duplicata_e_em_ordem(self, rag_service):
        """Sources devem aparecer uma única vez, na ordem em que foram recuperadas."""
        result = rag_service.answer("Qual o prazo de consentimento?")

        # resolucao_001.pdf aparece 2x nos docs, mas deve aparecer 1x nas sources
        assert result.sources.count("resolucao_001.pdf") == 1
        assert result.sources.count("lgpd_resumo.pdf") == 1
        # Ordem preservada: resolucao_001 veio primeiro
        assert result.sources[0] == "resolucao_001.pdf"

    def test_confidence_calculada_pelo_melhor_score(self, rag_service):
        """Confidence deve ser baseada no chunk mais similar (menor score)."""
        result = rag_service.answer("Qual o prazo de consentimento?")

        # Melhor score é 0.12 → confidence = 1.0 - 0.12 = 0.88
        assert result.confidence == pytest.approx(0.88, abs=0.01)

    def test_answer_contem_resposta_do_llm(self, rag_service, mock_llm):
        """O campo answer deve conter exatamente o que o LLM retornou."""
        result = rag_service.answer("Qual o prazo?")
        assert result.answer == mock_llm.generate.return_value


class TestRAGServiceFallback:

    def test_indice_vazio_retorna_resposta_padrao(self, mock_llm):
        """Quando o vector_store não encontra chunks, RAGService não deve chamar o LLM."""
        empty_store = MagicMock()
        empty_store.similarity_search.return_value = []

        service = RAGService(vector_store=empty_store, llm=mock_llm)
        result = service.answer("Qual o prazo de consentimento?")

        assert result.confidence == 0.0
        assert result.sources == []
        assert "não encontrei" in result.answer.lower()
        mock_llm.generate.assert_not_called()  # LLM não deve ser chamado sem contexto

    def test_confidence_sem_score_retorna_neutro(self):
        """Docs sem similarity_score no metadata → confidence neutro (0.5)."""
        store = MagicMock()
        store.similarity_search.return_value = [
            Document(content="texto", source="doc.pdf", metadata={"page": 1})
            # sem similarity_score
        ]
        llm = MagicMock()
        llm.generate.return_value = "resposta"

        service = RAGService(vector_store=store, llm=llm)
        result = service.answer("pergunta qualquer")

        assert result.confidence == 0.5