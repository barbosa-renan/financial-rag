import sys
import os
import argparse
import json
import csv
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Golden dataset ─────────────────────────────────────────────────────────────
# Perguntas sobre os documentos que você indexou, com respostas esperadas.
# Expanda este dataset conforme indexar mais documentos.
# ground_truth: resposta correta que o RAG deveria produzir.
# Usado pelo RAGAS para calcular context_recall.

GOLDEN_DATASET = [
    {
        "question": "Qual o prazo máximo de vigência do consentimento no Open Finance?",
        "ground_truth": "O prazo máximo de vigência do consentimento é de 12 meses.",
    },
    {
        "question": "O consentimento no Open Finance pode ser revogado pelo cliente?",
        "ground_truth": "Sim, o consentimento pode ser revogado a qualquer tempo pelo titular dos dados.",
    },
    {
        "question": "Quais dados são considerados cadastrais no Open Finance?",
        "ground_truth": "Dados cadastrais incluem informações de identificação do cliente como nome, CPF/CNPJ, endereço e dados de contato.",
    },
    {
        "question": "Qual legislação rege o consentimento de dados no Open Finance?",
        "ground_truth": "O consentimento deve estar em conformidade com a Lei Geral de Proteção de Dados Pessoais (LGPD).",
    },
    {
        "question": "Qual a responsabilidade da instituição transmissora no Open Finance?",
        "ground_truth": "A instituição transmissora é responsável por validar a autenticidade e integridade do consentimento antes de qualquer compartilhamento.",
    },
]


def run_evaluation(
    min_faithfulness: float = 0.75,
    min_answer_relevancy: float = 0.70,
    export_mlflow: bool = False,
) -> dict:
    """
    Executa o pipeline de avaliação completo.
    Retorna dicionário com métricas calculadas.
    """

    # ── Imports tardios — evita falha se RAGAS não estiver instalado ──
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
        from datasets import Dataset
    except ImportError:
        print("RAGAS não instalado. Execute: pip install ragas datasets")
        sys.exit(1)

    from src.adapters.chroma_vector_store import ChromaVectorStore
    from src.adapters.llm_adapter import create_llm_adapter
    from src.application.rag_service import RAGService

    print(f"\n{'='*60}")
    print(f"  Financial RAG — Avaliação de Qualidade")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── Inicializa o pipeline ──────────────────────────────────────────
    print("Inicializando pipeline RAG...")
    vector_store = ChromaVectorStore()
    llm = create_llm_adapter()
    service = RAGService(vector_store=vector_store, llm=llm)

    # ── Executa RAG para cada pergunta do golden dataset ──────────────
    print(f"Executando {len(GOLDEN_DATASET)} perguntas do golden dataset...\n")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(GOLDEN_DATASET, 1):
        question = item["question"]
        print(f"  [{i}/{len(GOLDEN_DATASET)}] {question[:60]}...")

        try:
            # Executa o RAG completo
            response = service.answer(question)

            # Coleta os chunks recuperados para o RAGAS avaliar o retriever
            retrieved_docs = vector_store.similarity_search(question, k=4)
            context_texts = [doc.content for doc in retrieved_docs]

            questions.append(question)
            answers.append(response.answer)
            contexts.append(context_texts)
            ground_truths.append(item["ground_truth"])

            print(f"       confidence={response.confidence:.2f} | chunks={len(context_texts)}")

        except Exception as exc:
            print(f"       ERRO: {exc}")
            # Inclui com resposta vazia para não distorcer o dataset
            questions.append(question)
            answers.append("")
            contexts.append([])
            ground_truths.append(item["ground_truth"])

    # ── Calcula métricas com RAGAS ─────────────────────────────────────
    print("\nCalculando métricas RAGAS...")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    metrics = {
        "faithfulness": round(float(results["faithfulness"]), 4),
        "answer_relevancy": round(float(results["answer_relevancy"]), 4),
        "context_recall": round(float(results["context_recall"]), 4),
        "context_precision": round(float(results["context_precision"]), 4),
        "evaluated_at": datetime.now().isoformat(),
        "n_questions": len(GOLDEN_DATASET),
    }

    # ── Exibe resultados ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"{'='*60}")
    print(f"  Faithfulness      : {metrics['faithfulness']:.4f}  (threshold: {min_faithfulness})")
    print(f"  Answer Relevancy  : {metrics['answer_relevancy']:.4f}  (threshold: {min_answer_relevancy})")
    print(f"  Context Recall    : {metrics['context_recall']:.4f}")
    print(f"  Context Precision : {metrics['context_precision']:.4f}")
    print(f"{'='*60}\n")

    # ── Salva histórico em CSV ─────────────────────────────────────────
    history_path = "data/evaluation_history.csv"
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(history_path)

    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"Histórico salvo em: {history_path}")

    # ── Exporta para MLflow ────────────────────────────────
    if export_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("financial-rag-quality")
            with mlflow.start_run():
                mlflow.log_metrics({
                    "faithfulness": metrics["faithfulness"],
                    "answer_relevancy": metrics["answer_relevancy"],
                    "context_recall": metrics["context_recall"],
                    "context_precision": metrics["context_precision"],
                })
                mlflow.log_params({
                    "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                    "chunk_size": os.getenv("CHUNK_SIZE", "800"),
                    "chunk_overlap": os.getenv("CHUNK_OVERLAP", "100"),
                    "retriever_k": os.getenv("RETRIEVER_K", "4"),
                })
            print("Métricas exportadas para MLflow")
        except Exception as exc:
            print(f"MLflow indisponível: {exc}")

    return metrics


def check_thresholds(metrics: dict, min_faithfulness: float, min_answer_relevancy: float) -> bool:
    """
    Verifica se as métricas estão acima dos thresholds mínimos.
    Retorna True se passou, False se falhou.
    Usado pelo CI/CD como gate de qualidade.
    """
    passed = True
    failures = []

    if metrics["faithfulness"] < min_faithfulness:
        failures.append(
            f"Faithfulness {metrics['faithfulness']:.4f} < {min_faithfulness} (threshold)"
        )
        passed = False

    if metrics["answer_relevancy"] < min_answer_relevancy:
        failures.append(
            f"Answer Relevancy {metrics['answer_relevancy']:.4f} < {min_answer_relevancy} (threshold)"
        )
        passed = False

    if failures:
        print("QUALIDADE INSUFICIENTE — deploy bloqueado:")
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("Qualidade aprovada — deploy liberado ✓")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação de qualidade do RAG")
    parser.add_argument("--min-faithfulness", type=float, default=0.75)
    parser.add_argument("--min-answer-relevancy", type=float, default=0.70)
    parser.add_argument("--export-mlflow", action="store_true")
    args = parser.parse_args()

    metrics = run_evaluation(
        min_faithfulness=args.min_faithfulness,
        min_answer_relevancy=args.min_answer_relevancy,
        export_mlflow=args.export_mlflow,
    )

    passed = check_thresholds(metrics, args.min_faithfulness, args.min_answer_relevancy)
    sys.exit(0 if passed else 1)