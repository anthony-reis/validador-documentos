"""Wrapper sobre o LLM local (Ollama). `temperature=0` por reprodutibilidade
dos experimentos (ver CLAUDE.md > Avaliacao).

`method="json_schema"` no `with_structured_output`, nao o default
("function_calling"): testado empiricamente contra qwen2.5:7b-instruct-q4_K_M
e o default retorna None silenciosamente (o modelo nao produz uma tool
call que o parser reconheca), enquanto json_schema -- que usa o suporte
nativo do Ollama a saida restrita por schema -- funciona de forma
confiavel.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src import config

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def _carregar_llm() -> ChatOllama:
    return ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)


def invocar_estruturado(system: str, prompt: str, schema: type[T]) -> T:
    llm_estruturado = _carregar_llm().with_structured_output(schema, method="json_schema")
    resultado = llm_estruturado.invoke([("system", system), ("human", prompt)])
    if resultado is None:
        raise RuntimeError(
            f"LLM nao produziu saida estruturada valida para o schema {schema.__name__} "
            "-- ver CLAUDE.md > 'sem mocks silenciosos': isso deve propagar como erro, "
            "nao virar um veredito fabricado."
        )
    return resultado
