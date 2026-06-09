import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def run_batch_ingest(pdf_dir: str, reindex: bool = False):
    from src.adapters.chroma_vector_store import ChromaVectorStore
    from src.adapters.llm_adapter import create_llm_adapter
    from src.application.rag_service import RAGService

    print(f"\n{'='*60}")
    print(f"  Ingestão em Lote — Financial RAG")
    print(f"  Diretório: {pdf_dir}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    vector_store = ChromaVectorStore()

    # Reindexação limpa — deleta tudo e recomeça
    if reindex:
        print("Modo reindexação: deletando coleção existente...")
        vector_store.delete_collection()
        # Recria após deletar
        vector_store = ChromaVectorStore()

    llm = create_llm_adapter()
    service = RAGService(vector_store=vector_store, llm=llm)

    # Lista PDFs disponíveis
    pdf_files = [
        f for f in sorted(os.listdir(pdf_dir))
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"Nenhum PDF encontrado em: {pdf_dir}")
        sys.exit(1)

    print(f"PDFs encontrados: {len(pdf_files)}\n")

    results = []

    for i, filename in enumerate(pdf_files, 1):
        filepath = os.path.join(pdf_dir, filename)
        print(f"  [{i}/{len(pdf_files)}] {filename}...")

        result = service.ingest(filepath)
        results.append(result)

        if result.success:
            print(f"           ✓ {result.chunks_created} chunks indexados")
        else:
            print(f"           ✗ FALHOU: {result.error}")

    # Relatório final
    total_chunks = sum(r.chunks_created for r in results)
    successes = sum(1 for r in results if r.success)
    failures = sum(1 for r in results if not r.success)

    print(f"\n{'='*60}")
    print(f"  RELATÓRIO FINAL")
    print(f"{'='*60}")
    print(f"  Arquivos processados : {len(results)}")
    print(f"  Sucesso              : {successes}")
    print(f"  Falhas               : {failures}")
    print(f"  Total de chunks      : {total_chunks}")
    print(f"  Total no banco       : {vector_store.count()}")
    print(f"{'='*60}\n")

    return failures == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestão em lote de PDFs")
    parser.add_argument("--dir", default="data/pdfs", help="Diretório com os PDFs")
    parser.add_argument("--reindex", action="store_true", help="Deleta e reindexar tudo")
    args = parser.parse_args()

    success = run_batch_ingest(args.dir, args.reindex)
    sys.exit(0 if success else 1)