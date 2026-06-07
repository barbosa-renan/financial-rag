from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """
    Entidade central do domínio, representa um chunk de documento após o processamento.
    """
    content: str
    source: str                          # caminho do arquivo original
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        page = self.metadata.get("page", "?")
        preview = self.content[:80].replace("\n", " ")
        return f"[{self.source} | pág. {page}] {preview}..."


@dataclass
class RAGResponse:
    """
    Resposta produzida pelo pipeline RAG.
    Inclui rastreabilidade de fontes, essencial para compliance.
    """
    answer: str
    sources: list[str]
    confidence: float                    # 0.0 a 1.0
    cached: bool = False

    def is_grounded(self) -> bool:
        """Resposta tem pelo menos uma fonte identificada?"""
        return len(self.sources) > 0


@dataclass
class IngestResult:
    """Resultado de uma operação de ingestão de documentos."""
    file_path: str
    chunks_created: int
    success: bool
    error: Optional[str] = None
