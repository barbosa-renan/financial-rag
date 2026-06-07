from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Prompt principal do sistema — instrui o LLM sobre seu papel e restrições.
# As restrições são intencionais para contexto financeiro/regulatório:
#   1. Responder SOMENTE com base no contexto
#   2. Admitir quando não sabe → melhor que inventar dado regulatório
#   3. Citar página/seção → rastreabilidade para auditoria
SYSTEM_TEMPLATE = """Você é um assistente especialista em regulamentação financeira brasileira, \
com foco em Open Finance, Pix e normas do Banco Central do Brasil.

Responda SOMENTE com base nas informações presentes no contexto fornecido abaixo.
Não use conhecimento externo, não invente dados, não faça suposições.

Se a informação solicitada não estiver no contexto, responda exatamente:
"Não encontrei essa informação nos documentos disponíveis."

Quando a informação estiver disponível:
- Seja preciso e objetivo
- Cite o documento e número de página quando disponível no contexto
- Use linguagem formal adequada ao contexto regulatório

CONTEXTO:
{context}"""

HUMAN_TEMPLATE = """Pergunta: {question}

Resposta:"""

# Template montado — será reutilizado em cada chamada do RAGService
RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
])