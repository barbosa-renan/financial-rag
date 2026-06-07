"""
scripts/experiment_chunking.py
──────────────────────────────
Script de exploração para entender o impacto dos parâmetros de chunking
ANTES de indexar no banco vetorial. Roda sem precisar de OpenAI ou Redis.

Uso:
    python scripts/experiment_chunking.py --pdf data/pdfs/seu_arquivo.pdf
    python scripts/experiment_chunking.py --pdf data/pdfs/seu_arquivo.pdf --chunk-size 500 --overlap 80
    python scripts/experiment_chunking.py --text  # usa texto de exemplo embutido
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain.text_splitter import RecursiveCharacterTextSplitter


# Texto de exemplo: trecho fictício de regulamento Open Finance
SAMPLE_TEXT = """
Resolução Conjunta nº 1 — Regulamentação do Open Finance

Capítulo I — Das Disposições Gerais

Art. 1º O presente regulamento estabelece as diretrizes para o compartilhamento
de dados e serviços no âmbito do Sistema Financeiro Aberto (Open Finance).

Art. 2º Para os fins desta resolução, consideram-se:
I - dados cadastrais: informações de identificação do cliente, incluindo nome,
    CPF/CNPJ, endereço e dados de contato;
II - dados transacionais: registros de operações financeiras realizadas pelo
    cliente, incluindo data, valor e contraparte;
III - consentimento: autorização expressa concedida pelo cliente titular dos
    dados para o compartilhamento com instituições receptoras.

Parágrafo único. O consentimento de que trata o inciso III deverá ser obtido
de forma específica, granular, livre, informada e inequívoca, em conformidade
com a Lei Geral de Proteção de Dados Pessoais (LGPD).

Capítulo II — Do Prazo de Vigência do Consentimento

Art. 3º O prazo máximo de vigência do consentimento para compartilhamento de
dados é de 12 (doze) meses, podendo ser revogado a qualquer tempo pelo titular.

Art. 4º A instituição transmissora de dados é responsável por validar a
autenticidade e a integridade do consentimento antes de qualquer
compartilhamento de informações.
"""


def analyze_chunks(text: str, chunk_size: int, overlap: int, label: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    sizes = [len(c) for c in chunks]
    avg = sum(sizes) / len(sizes) if sizes else 0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  chunk_size={chunk_size} | overlap={overlap}")
    print(f"{'='*60}")
    print(f"  Total de chunks : {len(chunks)}")
    print(f"  Chars por chunk : min={min(sizes)} | max={max(sizes)} | avg={avg:.0f}")
    print()

    for i, chunk in enumerate(chunks):
        preview = chunk[:120].replace("\n", "↵ ")
        print(f"  [chunk {i+1:02d}] ({len(chunk)} chars)")
        print(f"  {preview}{'...' if len(chunk) > 120 else ''}")
        print()

    # Avisa chunks muito pequenos (provavelmente cabeçalho ou rodapé isolado)
    tiny = [c for c in chunks if len(c) < 100]
    if tiny:
        print(f"  ⚠️  {len(tiny)} chunk(s) com menos de 100 chars — candidatos a ruído")

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Experimentos de chunking")
    parser.add_argument("--pdf", help="Caminho para um PDF real")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--text", action="store_true", help="Usa texto de exemplo embutido")
    args = parser.parse_args()

    if args.pdf:
        try:
            from langchain_community.document_loaders import PyPDFLoader
            pages = PyPDFLoader(args.pdf).load()
            # Usa as primeiras 3 páginas para análise
            text = "\n\n".join(p.page_content for p in pages[:3])
            print(f"\nPDF carregado: {args.pdf} ({len(pages)} páginas, usando primeiras 3)")
        except Exception as e:
            print(f"Erro ao carregar PDF: {e}")
            sys.exit(1)
    else:
        text = SAMPLE_TEXT
        print("\nUsando texto de exemplo (regulamento fictício Open Finance)")

    if args.chunk_size:
        # Modo: testa um único config especificado pelo usuário
        analyze_chunks(
            text,
            chunk_size=args.chunk_size,
            overlap=args.overlap or int(args.chunk_size * 0.12),
            label="Configuração customizada",
        )
    else:
        # Modo padrão: compara 3 configurações lado a lado
        configs = [
            (400,  50,  "Pequeno  — bom para FAQ / perguntas curtas"),
            (800,  100, "Médio    — recomendado para docs regulatórios ✓"),
            (1200, 150, "Grande   — bom para contexto amplo, risco de ruído"),
        ]
        for size, overlap, label in configs:
            analyze_chunks(text, size, overlap, label)

        print("\n" + "="*60)
        print("  RECOMENDAÇÃO para documentos financeiros/regulatórios:")
        print("  chunk_size=800, overlap=100")
        print("  Motivo: artigos e parágrafos costumam ter 400-700 chars.")
        print("  800 cabe um artigo completo sem cortar; overlap=100")
        print("  garante que frases na borda não se percam entre chunks.")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
