from pydantic import BaseModel, Field
from typing import List


class AskRequest(BaseModel):
    """
    Corpo da requisição POST /v1/ask.
    Pydantic valida automaticamente: question não pode ser vazio,
    use_cache deve ser boolean.
    """
    question: str = Field(
        ...,                            # ... = campo obrigatório
        min_length=5,
        max_length=500,
        description="Pergunta sobre os documentos financeiros indexados",
        examples=["Qual o prazo máximo de vigência do consentimento no Open Finance?"],
    )
    use_cache: bool = Field(
        default=True,
        description="Se False, ignora o cache e sempre chama o pipeline RAG completo",
    )


class AskResponse(BaseModel):
    """
    Corpo da resposta do POST /v1/ask.
    Espelha RAGResponse mas com campos pensados para consumo externo.
    """
    answer: str = Field(description="Resposta gerada pelo LLM")
    sources: List[str] = Field(description="Arquivos que embasaram a resposta")
    confidence: float = Field(description="Confiança da resposta (0.0 a 1.0)")
    cached: bool = Field(description="True se a resposta veio do cache Redis")


class IngestRequest(BaseModel):
    """Corpo da requisição POST /v1/ingest."""
    file_path: str = Field(
        ...,
        description="Caminho absoluto ou relativo ao PDF a ser indexado",
        examples=["data/pdfs/file_name.pdf"],
    )


class IngestResponse(BaseModel):
    """Corpo da resposta do POST /v1/ingest."""
    file_path: str
    chunks_created: int
    success: bool
    error: str | None = None


class HealthResponse(BaseModel):
    """Corpo da resposta do GET /health."""
    status: str                 
    vector_store: str        
    cache: str 
    version: str