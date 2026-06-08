import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.api.main import app
from src.api.dependencies import get_rag_service, get_cache, get_vector_store
from src.domain.entities import RAGResponse


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_rag_service():
    service = MagicMock()
    service.answer.return_value = RAGResponse(
        answer="O prazo de consentimento é de 12 meses conforme Resolução Conjunta nº 1.",
        sources=["resolucao_conjunta_001.pdf"],
        confidence=0.91,
        cached=False,
    )
    service.ingest.return_value = MagicMock(
        success=True,
        file_path="data/pdfs/resolucao_conjunta_001.pdf",
        chunks_created=38,
        error=None,
    )
    return service


@pytest.fixture
def mock_cache():
    """Cache que sempre retorna miss por padrão."""
    cache = MagicMock()
    cache.get.return_value = None          # cache miss
    cache.is_available.return_value = True
    return cache


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.count.return_value = 42
    return store


@pytest.fixture
def client(mock_rag_service, mock_cache, mock_vector_store):
    """
    TestClient com dependências mockadas via dependency_overrides.
    Cada teste começa com overrides limpos — sem efeito colateral entre testes.
    """
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    app.dependency_overrides[get_cache] = lambda: mock_cache
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    yield TestClient(app)
    app.dependency_overrides.clear()


# ── Testes do POST /v1/ask ─────────────────────────────────────────────────────

class TestAskEndpoint:

    def test_retorna_200_com_resposta_valida(self, client):
        response = client.post("/v1/ask", json={
            "question": "Qual o prazo de consentimento no Open Finance?"
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "cached" in data

    def test_cache_miss_chama_rag_service(self, client, mock_rag_service, mock_cache):
        """Se cache retorna None, o RAGService deve ser chamado."""
        mock_cache.get.return_value = None

        client.post("/v1/ask", json={
            "question": "Qual o prazo de consentimento no Open Finance?"
        })

        mock_rag_service.answer.assert_called_once()

    def test_cache_hit_nao_chama_rag_service(self, client, mock_rag_service, mock_cache):
        """Se cache tem a resposta, o RAGService NÃO deve ser chamado."""
        mock_cache.get.return_value = RAGResponse(
            answer="Resposta do cache.",
            sources=["doc.pdf"],
            confidence=0.88,
            cached=True,
        )

        response = client.post("/v1/ask", json={
            "question": "Qual o prazo de consentimento no Open Finance?"
        })

        assert response.json()["cached"] is True
        mock_rag_service.answer.assert_not_called()

    def test_use_cache_false_ignora_cache(self, client, mock_rag_service, mock_cache):
        """use_cache=False deve pular o cache e sempre chamar o RAGService."""
        mock_cache.get.return_value = RAGResponse(
            answer="Cache seria retornado aqui.",
            sources=[],
            confidence=0.9,
            cached=True,
        )

        client.post("/v1/ask", json={
            "question": "Qual o prazo de consentimento?",
            "use_cache": False,
        })

        mock_rag_service.answer.assert_called_once()

    def test_question_muito_curta_retorna_422(self, client):
        """Pergunta com menos de 5 caracteres deve falhar na validação Pydantic."""
        response = client.post("/v1/ask", json={"question": "Oi"})
        assert response.status_code == 422

    def test_question_ausente_retorna_422(self, client):
        """Body sem o campo question deve retornar erro de validação."""
        response = client.post("/v1/ask", json={})
        assert response.status_code == 422


# ── Testes do POST /v1/ingest ──────────────────────────────────────────────────

class TestIngestEndpoint:

    def test_ingest_com_sucesso_retorna_200(self, client):
        response = client.post("/v1/ingest", json={
            "file_path": "data/pdfs/resolucao_conjunta_001.pdf"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["chunks_created"] == 38

    def test_ingest_com_falha_retorna_422(self, client, mock_rag_service):
        mock_rag_service.ingest.return_value = MagicMock(
            success=False,
            file_path="inexistente.pdf",
            chunks_created=0,
            error="Arquivo não encontrado",
        )

        response = client.post("/v1/ingest", json={"file_path": "inexistente.pdf"})
        assert response.status_code == 422


# ── Testes do GET /health ──────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_retorna_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_retorna_estrutura_correta(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "vector_store" in data
        assert "cache" in data
        assert "version" in data

    def test_health_healthy_quando_tudo_ok(self, client, mock_cache):
        mock_cache.is_available.return_value = True
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_health_degraded_quando_redis_indisponivel(self, client, mock_cache):
        mock_cache.is_available.return_value = False
        response = client.get("/health")
        assert response.json()["status"] == "degraded"