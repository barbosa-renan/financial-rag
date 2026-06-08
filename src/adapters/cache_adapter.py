import json
import hashlib
import logging
from typing import Optional

import redis

from src.domain.entities import RAGResponse
from src.domain.ports import CachePort
from src.config import get_settings

logger = logging.getLogger(__name__)


class RedisCacheAdapter(CachePort):
    """
    Adapter de cache usando Redis.

    Graceful degradation: se o Redis estiver indisponível,
    get() retorna None e set() loga o erro sem lançar exceção.
    """

    def __init__(self):
        settings = get_settings()
        try:
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,   
                socket_timeout=2,
            )
            self._client.ping()            
            self._ttl = settings.cache_ttl_seconds
            logger.info("RedisCacheAdapter conectado | url=%s", settings.redis_url)
        except redis.exceptions.ConnectionError:
            self._client = None
            logger.warning(
                "Redis indisponível — cache desativado. "
                "O sistema funciona normalmente, porém sem cache."
            )

    # ── CachePort ──────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[RAGResponse]:
        """
        Busca resposta no cache.
        Retorna None se: chave não existe, Redis indisponível, ou dado corrompido.
        """
        if self._client is None:
            return None

        try:
            raw = self._client.get(self._make_key(key))
            if raw is None:
                return None

            data = json.loads(raw)
            return RAGResponse(
                answer=data["answer"],
                sources=data["sources"],
                confidence=data["confidence"],
                cached=True,      
            )
        except Exception as exc:
            logger.warning("Cache get falhou | key=%s | erro=%s", key[:20], exc)
            return None

    def set(self, key: str, value: RAGResponse, ttl: int = None) -> None:
        """
        Persiste resposta no cache com TTL.
        Falhas silenciosas, cache indisponível não deve quebrar o fluxo.
        """
        if self._client is None:
            return

        try:
            data = {
                "answer": value.answer,
                "sources": value.sources,
                "confidence": value.confidence,
            }
            self._client.setex(
                name=self._make_key(key),
                time=ttl or self._ttl,
                value=json.dumps(data, ensure_ascii=False),
            )
            logger.debug("Cache set | key=%s... | ttl=%ds", key[:20], ttl or self._ttl)
        except Exception as exc:
            logger.warning("Cache set falhou | erro=%s", exc)

    def is_available(self) -> bool:
        """Verifica se o Redis está acessível"""
        if self._client is None:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    # ── Helper ─────────────────────────────────────────────────────────

    def _make_key(self, question: str) -> str:
        """
        Gera chave de cache a partir da pergunta.
        MD5 do texto normalizado (lowercase, sem espaços extras)
        garante que variações de capitalização não gerem cache miss.
        """
        normalized = question.strip().lower()
        hash_hex = hashlib.md5(normalized.encode()).hexdigest()
        return f"rag:v1:{hash_hex}"