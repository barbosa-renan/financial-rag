"""
Testes do ChromaVectorStore usando FakeEmbeddings.
Não precisam de OPENAI_API_KEY — rodam completamente offline.
FakeEmbeddings gera vetores aleatórios: testa o comportamento do adapter,
não a qualidade semântica dos embeddings (isso é responsabilidade da OpenAI).
"""
import pytest
from unittest.mock import patch
from langchain_community.embeddings import FakeEmbeddings
from langchain_chroma import Chroma

from src.domain.entities import Document
from src.adapters.chroma_vector_store import ChromaVectorStore


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def vector_store(tmp_path):
    """
    Cria um ChromaVectorStore em memória para cada teste.
    tmp_path é uma fixture nativa do pytest — cria uma pasta temporária
    única por teste e apaga ao final. Cada teste começa com índice vazio.
    """
    with patch.object(ChromaVectorStore, "__init__", lambda self: None):
        store = ChromaVectorStore.__new__(ChromaVectorStore)
        store._embeddings = FakeEmbeddings(size=1536)
        store._store = Chroma(
            collection_name="test_collection",
            embedding_function=FakeEmbeddings(size=1536),
            persist_directory=str(tmp_path),
        )
    return store


def make_docs(n: int, source: str = "regulamento.pdf") -> list[Document]:
    """Helper: cria n documentos de teste com conteúdo variado."""
    topics = [
        "O prazo de consentimento no Open Finance é de 12 meses conforme regulação Bacen.",
        "A instituição transmissora deve validar autenticidade antes do compartilhamento.",
        "O Pix deve liquidar transações em até 10 segundos em condições normais.",
        "Dados cadastrais incluem nome, CPF, endereço e informações de contato do cliente.",
        "A LGPD exige consentimento específico, granular e inequívoco do titular dos dados.",
    ]
    return [
        Document(
            content=topics[i % len(topics)],
            source=source,
            metadata={"page": i, "source": source},
        )
        for i in range(n)
    ]


# ── Testes ─────────────────────────────────────────────────────────────────────

class TestChromaVectorStore:

    def test_add_e_count_documentos(self, vector_store):
        """Após indexar N docs, count() deve retornar N."""
        docs = make_docs(3)
        vector_store.add_documents(docs)
        assert vector_store.count() == 3

    def test_add_lista_vazia_nao_quebra(self, vector_store):
        """Indexar lista vazia não deve lançar exceção."""
        vector_store.add_documents([])
        assert vector_store.count() == 0

    def test_similarity_search_retorna_documentos(self, vector_store):
        """Busca semântica deve retornar instâncias de Document."""
        vector_store.add_documents(make_docs(5))
        results = vector_store.similarity_search("consentimento Open Finance", k=2)

        assert len(results) == 2
        assert all(isinstance(r, Document) for r in results)

    def test_similarity_search_respeita_k(self, vector_store):
        """O parâmetro k deve limitar o número de resultados."""
        vector_store.add_documents(make_docs(10))

        for k in [1, 3, 5]:
            results = vector_store.similarity_search("Pix pagamento", k=k)
            assert len(results) == k, f"Esperado {k} resultados, obtido {len(results)}"

    def test_similarity_search_popula_similarity_score(self, vector_store):
        """Cada resultado deve ter similarity_score no metadata."""
        vector_store.add_documents(make_docs(3))
        results = vector_store.similarity_search("LGPD dados pessoais", k=2)

        for doc in results:
            assert "similarity_score" in doc.metadata
            assert isinstance(doc.metadata["similarity_score"], float)

    def test_similarity_search_query_vazia_retorna_lista_vazia(self, vector_store):
        """Query em branco não deve buscar — retorna lista vazia."""
        vector_store.add_documents(make_docs(3))
        results = vector_store.similarity_search("   ")
        assert results == []

    def test_delete_collection_zera_indice(self, vector_store):
        """Após delete_collection, o índice deve estar vazio."""
        vector_store.add_documents(make_docs(5))
        assert vector_store.count() == 5

        vector_store.delete_collection()

        # Recria a coleção para poder usar count() sem erro
        vector_store._store = Chroma(
            collection_name="test_collection",
            embedding_function=FakeEmbeddings(size=1536),
        )
        assert vector_store.count() == 0

    def test_source_preservada_apos_indexacao(self, vector_store):
        """O campo source do Document deve ser preservado após indexar e recuperar."""
        docs = make_docs(3, source="resolucao_conjunta_001.pdf")
        vector_store.add_documents(docs)

        results = vector_store.similarity_search("consentimento", k=3)
        for result in results:
            assert result.source == "resolucao_conjunta_001.pdf"