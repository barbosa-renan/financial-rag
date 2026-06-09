# Financial RAG API

Assistente de documentos financeiros com busca semântica e LLM.  
Arquitetura hexagonal em Python.

## Arquitetura

PDFs → Chunking → Embeddings → Chroma (dev) / pgvector (prod)
↓
Pergunta → Retriever → Contexto → LLM → Resposta + Fontes
↓
Redis Cache

**Stack:**  
LangChain · OpenAI Embeddings · ChromaDB · FastAPI · Redis · Docker · RAGAS · MLflow

**Padrão arquitetural:** Hexagonal (Ports & Adapters)  
- `domain/` — entidades e ports (zero dependência externa)  
- `application/` — casos de uso (orquestra ports)  
- `adapters/` — implementações concretas (Chroma, OpenAI, Redis)  
- `api/` — adapter de entrada (FastAPI)

## Início rápido

```bash
# 1. Clone e configure
cp .env.example .env
# Edite .env com sua OPENAI_API_KEY

# 2. Sobe infraestrutura
docker-compose up -d redis chromadb

# 3. Instala dependências
pip install -r requirements.txt

# 4. Indexa documentos
python scripts/ingest_batch.py --dir data/pdfs

# 5. Sobe a API
uvicorn src.api.main:app --reload --port 8000
```

## Endpoints

| Método | Endpoint    | Descrição                          |
|--------|-------------|-------------------------------------|
| POST   | /v1/ask     | Pergunta aos documentos indexados   |
| POST   | /v1/ingest  | Indexa um PDF no banco vetorial     |
| GET    | /health     | Healthcheck (Chroma + Redis)        |
| GET    | /docs       | Documentação OpenAPI interativa     |

## Exemplo de uso

```bash
# Ingestão
curl -X POST http://localhost:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/pdfs/resolucao_conjunta_001.pdf"}'

# Pergunta
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual o prazo de vigência do consentimento no Open Finance?"}'
```

## Testes

```bash
# Todos os testes (unitários + API)
python -m pytest tests/ -v

# Avaliação de qualidade RAG
python scripts/evaluate_rag.py
```

## MLOps

A qualidade do RAG é medida automaticamente via RAGAS:

| Métrica           | Threshold | Descrição                              |
|-------------------|-----------|----------------------------------------|
| Faithfulness      | ≥ 0.75    | LLM não alucinou além do contexto      |
| Answer Relevancy  | ≥ 0.70    | Resposta pertinente à pergunta         |
| Context Recall    | —         | Retriever achou os chunks certos       |
| Context Precision | —         | Chunks recuperados eram necessários    |

Visualize experimentos no MLflow: `http://localhost:5000`