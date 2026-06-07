import logging
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from src.domain.ports import LLMPort
from src.config import get_settings

logger = logging.getLogger(__name__)


class OpenAILLMAdapter(LLMPort):
    """
    Adapter para modelos da OpenAI (gpt-4o-mini, gpt-4o, etc).

    Usando o gpt-4o-mini como padrão por questões de custo e latência menores que gpt-4o
    """

    def __init__(self):
        settings = get_settings()

        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,        # determinístico — crítico para compliance
            openai_api_key=settings.openai_api_key,
            max_tokens=1024,      # suficiente para respostas regulatórias
        )
        self._parser = StrOutputParser()
        logger.info("OpenAILLMAdapter inicializado | model=%s", settings.llm_model)

    def generate(self, question: str, context: str) -> str:
        """
        Gera resposta fundamentada no contexto.
        A chain usa o operador | do LangChain (pipe) — equivalente a:
        parser(llm(prompt.format(context=context, question=question)))
        """
        from src.application.prompt_templates import RAG_PROMPT

        chain = RAG_PROMPT | self._llm | self._parser

        response = chain.invoke({
            "context": context,
            "question": question,
        })

        logger.debug("LLM gerou resposta | chars=%d", len(response))
        return response


class AnthropicLLMAdapter(LLMPort):
    """
    Adapter para modelos da Anthropic (Claude).
    Útil se a empresa já usar Claude e quiser consistência.
    """

    def __init__(self):
        settings = get_settings()

        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "Instale langchain-anthropic: pip install langchain-anthropic"
            )

        self._llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",  # mais rápido e barato para RAG
            temperature=0,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=1024,
        )
        self._parser = StrOutputParser()

    def generate(self, question: str, context: str) -> str:
        from src.application.prompt_templates import RAG_PROMPT
        chain = RAG_PROMPT | self._llm | self._parser
        return chain.invoke({"context": context, "question": question})


def create_llm_adapter() -> LLMPort:
    """
    Factory function: decide qual adapter instanciar baseado no .env.
    O RAGService chama esta função em vez de instanciar diretamente,
    mantendo o desacoplamento.

    Para usar Anthropic, adicione no .env:
      LLM_PROVIDER=anthropic
    """
    settings = get_settings()
    provider = getattr(settings, "llm_provider", "openai")

    if provider == "anthropic":
        logger.info("Usando provider: Anthropic")
        return AnthropicLLMAdapter()

    logger.info("Usando provider: OpenAI")
    return OpenAILLMAdapter()