from abc import ABC, abstractmethod
from typing import List
from src.domain.entities import Document, RAGResponse, IngestResult


# ─────────────────────────────────────────────
# Ports de SAÍDA (driven ports)
# Implementados pelos adapters de infraestrutura
# ─────────────────────────────────────────────

class VectorStorePort(ABC):
    """
    Port para o banco vetorial.
    O domínio não sabe se é Chroma, pgvector, Pinecone ou qualquer outro.
    """

    @abstractmethod
    def add_documents(self, docs: List[Document]) -> None:
        """Indexa chunks no banco vetorial."""
        ...

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Busca semântica: retorna os k chunks mais relevantes."""
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Remove todos os documentos — útil para reindexação completa."""
        ...


class LLMPort(ABC):
    """
    Port para o modelo de linguagem.
    Permite trocar GPT-4 por Claude ou Llama sem mudar o caso de uso.
    """

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Gera resposta fundamentada no contexto fornecido."""
        ...


class CachePort(ABC):
    """
    Port para cache de respostas.
    Reduz latência e custo em perguntas repetidas.
    """

    @abstractmethod
    def get(self, key: str) -> RAGResponse | None:
        ...

    @abstractmethod
    def set(self, key: str, value: RAGResponse, ttl: int) -> None:
        ...


# ─────────────────────────────────────────────
# Ports de ENTRADA (driving ports)
# Definem o contrato que a API/CLI deve seguir
# ─────────────────────────────────────────────

class RAGServicePort(ABC):
    """Contrato do caso de uso principal."""

    @abstractmethod
    def answer(self, question: str) -> RAGResponse:
        ...

    @abstractmethod
    def ingest(self, file_path: str) -> IngestResult:
        ...
