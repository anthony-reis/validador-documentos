"""Análise de documento em thread de segundo plano.

Por quê: o Streamlit reexecuta o script a cada interação com um widget e
CANCELA a execução em andamento. Rodando a análise dentro do script, dar
feedback durante os julgamentos abortaria o que faltava. Aqui a análise
roda numa thread própria e só escreve neste objeto; a interface apenas
lê o estado (sem chamar `st.*` de dentro da thread).

Mesma lógica de antes (extração e julgamento intercalados por página,
recuperação em lote) -- só mudou onde ela roda.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from src import config
from src.agent import chat
from src.agent.nos import extrair_assercoes, julgar_assercao_com_resultados, parse_documento
from src.agent.schemas import JulgamentoAssercao
from src.retrieval.factory import criar_retriever


@dataclass
class AnaliseJob:
    documento: str
    estrategia: str
    caminho_pdf: Path
    limitar_paginas: int = 0

    # Estado escrito pela thread e lido pela UI; `_trava` protege as
    # listas e o texto de status para leitura consistente.
    julgamentos: list[JulgamentoAssercao] = field(default_factory=list)
    mensagens_pagina: list[str] = field(default_factory=list)
    fracao: float = 0.0
    status: str = "Lendo documento…"
    total_assercoes: int = 0
    indice_chat: object | None = None
    erro: str | None = None
    concluido: bool = False
    finalizado_na_ui: bool = False
    _cancelar: threading.Event = field(default_factory=threading.Event)
    _trava: threading.Lock = field(default_factory=threading.Lock)
    _thread: threading.Thread | None = None

    def iniciar(self) -> None:
        self._thread = threading.Thread(target=self._executar, daemon=True, name="analise-job")
        self._thread.start()

    def cancelar(self) -> None:
        self._cancelar.set()

    @property
    def cancelado(self) -> bool:
        return self._cancelar.is_set()

    def instantaneo(self) -> dict:
        """Cópia consistente do estado para a UI renderizar."""
        with self._trava:
            return {
                "julgamentos": list(self.julgamentos),
                "mensagens_pagina": list(self.mensagens_pagina),
                "fracao": self.fracao,
                "status": self.status,
                "concluido": self.concluido,
                "erro": self.erro,
            }

    def _atualizar(self, fracao: float | None = None, status: str | None = None) -> None:
        with self._trava:
            if fracao is not None:
                self.fracao = fracao
            if status is not None:
                self.status = status

    def _executar(self) -> None:
        paginas: list[str] = []
        try:
            paginas = parse_documento(self.caminho_pdf)
            if self.limitar_paginas:
                paginas = paginas[: self.limitar_paginas]
            total = len(paginas)
            retriever = None

            for indice_pagina, pagina in enumerate(paginas):
                if self.cancelado:
                    break
                self._atualizar(indice_pagina / max(total, 1), f"Extraindo página {indice_pagina + 1}/{total}…")
                novas = extrair_assercoes([pagina])
                self.total_assercoes += len(novas)
                if not novas:
                    continue
                with self._trava:
                    self.mensagens_pagina.append(
                        f"Página {indice_pagina + 1}/{total}: {len(novas)} asserção(ões) encontrada(s)."
                    )
                if retriever is None:
                    retriever = criar_retriever(self.estrategia)

                # Recuperação em lote por página (ver CLAUDE.md >
                # "Melhorias de performance"), em sub-lotes.
                resultados: list = []
                for inicio in range(0, len(novas), config.RETRIEVAL_BATCH_SIZE):
                    sub_lote = novas[inicio : inicio + config.RETRIEVAL_BATCH_SIZE]
                    if hasattr(retriever, "buscar_lote"):
                        resultados.extend(retriever.buscar_lote(sub_lote, top_k=config.RETRIEVAL_TOP_K))
                    else:
                        resultados.extend(retriever.buscar(a, top_k=config.RETRIEVAL_TOP_K) for a in sub_lote)

                for i, (assercao, resultado) in enumerate(zip(novas, resultados)):
                    if self.cancelado:
                        break
                    self._atualizar(
                        (indice_pagina + (i + 1) / len(novas)) / max(total, 1),
                        f"Julgando… {len(self.julgamentos) + 1}ª asserção (página {indice_pagina + 1}/{total})",
                    )
                    julgamento = julgar_assercao_com_resultados(resultado, assercao)
                    with self._trava:
                        self.julgamentos.append(julgamento)

            # Índice do chat construído uma única vez por documento.
            if paginas and self.julgamentos:
                self._atualizar(status="Indexando o documento para o chat…")
                self.indice_chat = chat.construir_indice(paginas)
            self._atualizar(
                1.0 if not self.cancelado else None,
                f"Cancelado — {len(self.julgamentos)} asserção(ões) julgada(s)."
                if self.cancelado
                else f"100% concluído — {len(self.julgamentos)} asserção(ões) julgada(s).",
            )
        except Exception as erro:  # a UI mostra o erro; a thread não pode morrer calada
            with self._trava:
                self.erro = f"{type(erro).__name__}: {erro}"
        finally:
            self.caminho_pdf.unlink(missing_ok=True)
            with self._trava:
                self.concluido = True
