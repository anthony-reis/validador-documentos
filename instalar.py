#!/usr/bin/env python3
"""Instala e prepara TODO o ambiente do validador-docs a partir de um
clone limpo do repositório -- pensado para rodar uma única vez logo
após `git clone`, antes de qualquer outro comando.

Só usa a biblioteca padrão do Python (nenhuma dependência externa): o
objetivo é funcionar ANTES de qualquer coisa estar instalada.

Uso:
    python3 instalar.py

Requisitos: macOS com Homebrew (https://brew.sh) já instalado -- este
projeto foi construído e validado nesse ambiente (ver README.md >
Requisitos). Outros sistemas operacionais não são suportados por este
script.

O que ele faz, em ordem, com barra de progresso e tempo gasto em cada
etapa (idempotente: rodar de novo pula o que já está pronto):
  1. Confere o Homebrew.
  2. Instala Python 3.11 e Ollama via Homebrew.
  3. Cria o ambiente virtual .venv.
  4. Instala as dependências do projeto dentro do .venv.
  5. Sobe o serviço do Ollama e baixa o modelo de LLM (~4,7 GB).
  6. Baixa o modelo de embeddings bge-m3 (~4,3 GB).
  7. Baixa o reranker bge-reranker-v2-m3 (~2,1 GB).
  8. Copia .env.example -> .env (se ainda não existir) e indexa o
     corpus normativo (já versionado no repositório).
Depois disso, abre a interface Streamlit no navegador padrão.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

# Sem isso, o stdout do PROPRIO script fica bufferizado quando a saida
# nao e' um terminal (ex.: redirecionada pra um arquivo de log) --
# so' a saida dos subprocessos (brew/pip/ollama/hf, que escrevem direto
# no descritor de arquivo) apareceria em tempo real, e as mensagens de
# progresso deste script ficariam invisiveis ate o processo terminar.
# Bug real encontrado testando o script com stdout redirecionado.
sys.stdout.reconfigure(line_buffering=True)

RAIZ = Path(__file__).resolve().parent
VENV_DIR = RAIZ / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
VENV_PIP = VENV_DIR / "bin" / "pip"
VENV_HF = VENV_DIR / "bin" / "hf"
VENV_STREAMLIT = VENV_DIR / "bin" / "streamlit"

TOTAL_ETAPAS = 8
LARGURA_BARRA = 30


def _barra(fracao: float) -> str:
    fracao = max(0.0, min(fracao, 1.0))
    preenchido = int(LARGURA_BARRA * fracao)
    return "#" * preenchido + "-" * (LARGURA_BARRA - preenchido)


def _tempo(inicio: float) -> str:
    segundos = time.time() - inicio
    if segundos < 60:
        return f"{segundos:.0f}s"
    return f"{segundos / 60:.1f}min"


def _rodar(comando: list[str]) -> None:
    """Roda um comando herdando stdout/stderr -- o usuário vê o
    progresso nativo de brew/pip/ollama/hf em tempo real, além da
    barra de etapas deste script."""
    print(f"    $ {' '.join(str(c) for c in comando)}")
    resultado = subprocess.run(comando, cwd=RAIZ)
    if resultado.returncode != 0:
        print(f"\nERRO: comando falhou (código {resultado.returncode}): {' '.join(str(c) for c in comando)}")
        sys.exit(resultado.returncode)


def _executar_etapa(numero: int, titulo: str, funcao) -> None:
    fracao_antes = (numero - 1) / TOTAL_ETAPAS
    print(f"\n[{_barra(fracao_antes)}] {fracao_antes * 100:5.1f}%  Etapa {numero}/{TOTAL_ETAPAS}: {titulo}...")
    inicio = time.time()
    funcao()
    fracao_depois = numero / TOTAL_ETAPAS
    print(f"[{_barra(fracao_depois)}] {fracao_depois * 100:5.1f}%  Etapa {numero}/{TOTAL_ETAPAS} concluída em {_tempo(inicio)}.")


def _localizar_python311() -> str:
    candidatos = [
        "/opt/homebrew/bin/python3.11",  # Apple Silicon
        "/usr/local/bin/python3.11",  # Intel
        shutil.which("python3.11"),
    ]
    for candidato in candidatos:
        if candidato and Path(candidato).exists():
            return candidato
    print("\nERRO: python3.11 não encontrado mesmo após 'brew install python@3.11'.")
    sys.exit(1)


def _modelo_ollama_ja_presente(nome_modelo: str) -> bool:
    resultado = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    return resultado.returncode == 0 and nome_modelo in resultado.stdout


# --- Etapas ---------------------------------------------------------


def etapa_homebrew() -> None:
    if shutil.which("brew") is None:
        print(
            "\nERRO: Homebrew não encontrado. Este script requer macOS com "
            "Homebrew já instalado -- veja https://brew.sh e rode este "
            "script de novo depois."
        )
        sys.exit(1)
    print("    Homebrew encontrado.")


def etapa_python_ollama() -> None:
    _rodar(["brew", "install", "python@3.11", "ollama"])


def etapa_venv() -> None:
    if VENV_PYTHON.exists():
        print("    .venv já existe, pulando.")
        return
    python311 = _localizar_python311()
    _rodar([python311, "-m", "venv", str(VENV_DIR)])


def etapa_dependencias() -> None:
    _rodar([str(VENV_PIP), "install", "--upgrade", "pip"])
    _rodar([str(VENV_PIP), "install", "-e", ".[dev]"])


def etapa_ollama_modelo() -> None:
    modelo = "qwen2.5:7b-instruct-q4_K_M"
    subprocess.run(["brew", "services", "start", "ollama"], cwd=RAIZ)
    time.sleep(3)
    if _modelo_ollama_ja_presente(modelo):
        print(f"    Modelo {modelo} já baixado, pulando.")
        return
    _rodar(["ollama", "pull", modelo])


def etapa_embeddings() -> None:
    destino = RAIZ / "models" / "bge-m3"
    if (destino / "config.json").exists():
        print("    models/bge-m3 já existe, pulando.")
        return
    _rodar([str(VENV_HF), "download", "BAAI/bge-m3", "--local-dir", str(destino)])


def etapa_reranker() -> None:
    destino = RAIZ / "models" / "bge-reranker-v2-m3"
    if (destino / "config.json").exists():
        print("    models/bge-reranker-v2-m3 já existe, pulando.")
        return
    _rodar([str(VENV_HF), "download", "BAAI/bge-reranker-v2-m3", "--local-dir", str(destino)])


def etapa_env_e_indice() -> None:
    env = RAIZ / ".env"
    if not env.exists():
        shutil.copyfile(RAIZ / ".env.example", env)
        print("    .env criado a partir de .env.example.")
    else:
        print("    .env já existe, mantido como está.")
    _rodar([str(VENV_PYTHON), "-m", "src.indexing.build"])


def abrir_streamlit() -> None:
    print("\nAbrindo a interface no navegador padrão (http://localhost:8501)...")
    print("Pressione Ctrl+C nesta janela para encerrar o servidor quando terminar.\n")
    try:
        subprocess.run([str(VENV_STREAMLIT), "run", str(RAIZ / "app" / "streamlit_app.py")], cwd=RAIZ)
    except KeyboardInterrupt:
        print("\nServidor encerrado.")


def main() -> None:
    inicio_total = time.time()
    print("=" * 64)
    print("validador-docs -- instalação completa do ambiente")
    print("=" * 64)
    print(
        "Isso vai baixar ~11 GB no total (LLM + modelos de embedding/\n"
        "reranker) e instalar todas as dependências do projeto. Pode\n"
        "levar bastante tempo dependendo da sua conexão -- acompanhe o\n"
        "progresso abaixo. Rodar este script de novo pula o que já\n"
        "estiver pronto."
    )

    try:
        _executar_etapa(1, "Verificando Homebrew", etapa_homebrew)
        _executar_etapa(2, "Instalando Python 3.11 e Ollama", etapa_python_ollama)
        _executar_etapa(3, "Criando ambiente virtual (.venv)", etapa_venv)
        _executar_etapa(4, "Instalando dependências Python do projeto", etapa_dependencias)
        _executar_etapa(5, "Subindo o Ollama e baixando o modelo de LLM (~4,7 GB)", etapa_ollama_modelo)
        _executar_etapa(6, "Baixando o modelo de embeddings bge-m3 (~4,3 GB)", etapa_embeddings)
        _executar_etapa(7, "Baixando o reranker bge-reranker-v2-m3 (~2,1 GB)", etapa_reranker)
        _executar_etapa(8, "Configurando .env e indexando o corpus normativo", etapa_env_e_indice)
    except KeyboardInterrupt:
        print("\n\nInstalação interrompida pelo usuário. Rode 'python3 instalar.py' de novo para retomar de onde parou.")
        sys.exit(1)

    print(f"\n[{_barra(1.0)}] 100.0%  Instalação concluída em {_tempo(inicio_total)}.")
    abrir_streamlit()


if __name__ == "__main__":
    main()
