import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.adapters.chroma_vector_store import ChromaVectorStore

s = ChromaVectorStore()

queries = [
    "instituicao transmissora de dados",
    "quem compartilha os dados do cliente",
    "prazo de validade do consentimento doze meses",
    "qual a receita de bolo de cenoura",
    "prazo vigencia consentimento", 
    "doze meses consentimento",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"  QUERY: {query}")
    print(f"{'='*60}")
    docs = s.similarity_search(query, k=3)
    for i, d in enumerate(docs, 1):
        score = d.metadata.get("similarity_score", "?")
        page = d.metadata.get("page", "?")
        preview = d.content[:150].replace("\n", " ")
        print(f"  [{i}] Score: {score} | Pág: {page}")
        print(f"       {preview}...")
        print()