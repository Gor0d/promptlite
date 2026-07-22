"""
PromptLite — Camada de Providers de LLM
Abstrai OpenAI e Anthropic (Claude) atrás de uma interface única, para que o
otimizador seja agnóstico de provider.

Embeddings: a Anthropic não expõe API de embeddings. Para a similaridade
semântica reutilizamos os embeddings da OpenAI quando `OPENAI_API_KEY` existir;
caso contrário caímos num fallback local baseado em difflib (aproximação léxica,
documentada como tal).
"""

from __future__ import annotations

import os
import json
import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional


# ─────────────────────────────────────────────
# TABELA DE PREÇOS (USD por token de input)
# ─────────────────────────────────────────────
# Mantida em um único lugar para `estimate_cost` e o endpoint /models.
PRICING_PER_INPUT_TOKEN = {
    # OpenAI
    "gpt-4o-mini": 0.15 / 1_000_000,
    "gpt-4o": 2.50 / 1_000_000,
    "gpt-4-turbo": 10.00 / 1_000_000,
    # Anthropic (Claude)
    "claude-opus-4-8": 5.00 / 1_000_000,
    "claude-sonnet-5": 3.00 / 1_000_000,
    "claude-haiku-4-5": 1.00 / 1_000_000,
}

PROVIDER_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
    "anthropic": ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8"],
}

DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}


def _extract_json(text: str) -> dict:
    """Extrai o primeiro objeto JSON de um texto, tolerante a cercas de código."""
    text = text.strip()
    # Remove cercas ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Última tentativa: pega do primeiro { ao último }
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


# ─────────────────────────────────────────────
# EMBEDDINGS (compartilhado entre providers)
# ─────────────────────────────────────────────

def _openai_embedding_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def embed_similarity(text1: str, text2: str) -> float:
    """
    Similaridade semântica entre dois textos (0..1).

    Usa embeddings da OpenAI (`text-embedding-3-small`) quando a chave está
    disponível; caso contrário, um fallback léxico via difflib. O fallback é
    uma aproximação — não substitui embeddings semânticos.
    """
    if _openai_embedding_available():
        try:
            from openai import OpenAI
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            client = OpenAI()
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=[text1[:8000], text2[:8000]],
            )
            emb1 = np.array(resp.data[0].embedding).reshape(1, -1)
            emb2 = np.array(resp.data[1].embedding).reshape(1, -1)
            return float(cosine_similarity(emb1, emb2)[0][0])
        except Exception:
            # Se a chamada de embeddings falhar, degrada para o fallback léxico.
            pass
    return SequenceMatcher(None, text1, text2).ratio()


# ─────────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────────

class LLMProvider(ABC):
    """Interface fina que o otimizador usa, independente do provider."""

    name: str

    @abstractmethod
    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        """Envia uma mensagem e retorna o texto da resposta."""

    @abstractmethod
    def chat_json(self, system: str, user: str) -> dict:
        """Como `chat`, mas parseia a resposta como JSON (com fallback)."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Conta tokens do texto para este modelo."""


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI()  # lê OPENAI_API_KEY do ambiente

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()

    def chat_json(self, system: str, user: str) -> dict:
        return _extract_json(self.chat(system, user, json_mode=True))

    def count_tokens(self, text: str) -> int:
        return _tiktoken_count(text, self.model)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5"):
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        # A Anthropic não tem `response_format`; instruímos via system quando JSON.
        if json_mode:
            system = system + "\n\nReturn ONLY a valid JSON object, no prose, no code fences."
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()

    def chat_json(self, system: str, user: str) -> dict:
        return _extract_json(self.chat(system, user, json_mode=True))

    def count_tokens(self, text: str) -> int:
        try:
            resp = self._client.messages.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": text}],
            )
            return resp.input_tokens
        except Exception:
            # Fallback aproximado se a API de contagem falhar.
            return _tiktoken_count(text, "gpt-4o-mini")


@lru_cache(maxsize=8)
def _encoding_for(model: str):
    import tiktoken
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Modelos novos podem não estar mapeados — usa o encoder mais recente.
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def _tiktoken_count(text: str, model: str) -> int:
    return len(_encoding_for(model).encode(text))


def get_provider(provider: str, model: Optional[str] = None) -> LLMProvider:
    """Fábrica: devolve o provider configurado, validando provider/modelo."""
    provider = (provider or "openai").lower()
    if provider not in PROVIDER_MODELS:
        raise ValueError(f"Provider desconhecido: {provider!r}. Use um de {list(PROVIDER_MODELS)}.")
    model = model or DEFAULT_MODEL[provider]
    if provider == "openai":
        return OpenAIProvider(model=model)
    return AnthropicProvider(model=model)
