"""Grafo LangGraph do agente (ver CLAUDE.md > "Arquitetura do agente").

Simplificacao deliberada: o diagrama da especificacao mostra um loop
"para cada requisito normativo aplicavel" dentro do estagio de
julgamento. Implementamos esse loop como uma iteracao Python dentro de
UM no' (`no_processar_assercoes`), em vez de expandir cada assercao em
um no' dinamico via a API `Send` do LangGraph. A maquina de estados real
(parse -> extrair -> processar -> fim) ja' captura a sequencia de
estagios que importa para o TFG (rastreabilidade de decisao por
estagio); usar `Send` so' adicionaria complexidade de execucao paralela
sem mudar o comportamento observavel, ja' que os julgamentos sao
independentes entre si e nao ha' necessidade real de paralelismo aqui.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.agent.nos import extrair_assercoes, julgar_assercao, parse_documento
from src.agent.schemas import JulgamentoAssercao
from src.retrieval.factory import criar_retriever


class EstadoAgente(TypedDict):
    caminho_documento: str
    estrategia_recuperacao: Literal["A", "B", "C", "D"]
    paginas_texto: list[str]
    assercoes: list[str]
    julgamentos: list[JulgamentoAssercao]


def no_parse(estado: EstadoAgente) -> dict:
    return {"paginas_texto": parse_documento(estado["caminho_documento"])}


def no_extrair(estado: EstadoAgente) -> dict:
    return {"assercoes": extrair_assercoes(estado["paginas_texto"])}


def no_processar_assercoes(estado: EstadoAgente) -> dict:
    retriever = criar_retriever(estado["estrategia_recuperacao"])
    julgamentos = [julgar_assercao(retriever, assercao) for assercao in estado["assercoes"]]
    return {"julgamentos": julgamentos}


def compilar_grafo():
    grafo = StateGraph(EstadoAgente)
    grafo.add_node("parse", no_parse)
    grafo.add_node("extrair_assercoes", no_extrair)
    grafo.add_node("processar_assercoes", no_processar_assercoes)

    grafo.set_entry_point("parse")
    grafo.add_edge("parse", "extrair_assercoes")
    grafo.add_edge("extrair_assercoes", "processar_assercoes")
    grafo.add_edge("processar_assercoes", END)

    return grafo.compile()
