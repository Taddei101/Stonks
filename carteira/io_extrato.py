"""Leitura dos extratos de Movimentação exportados da Área do Investidor da B3.

Esta camada é deliberadamente burra: localiza arquivos, valida o layout e
devolve os dados como texto, sem converter nada. A conversão fica em
`normalizar`, para que um erro de parsing apareça lá e não escondido num dtype.
"""

from pathlib import Path

import pandas as pd

# Contrato do export da B3. Se a B3 mexer no layout, queremos falhar aqui,
# alto e claro, em vez de descobrir por uma coluna trocada três meses depois.
COLUNAS_B3 = [
    "Entrada/Saída",
    "Data",
    "Movimentação",
    "Produto",
    "Instituição",
    "Quantidade",
    "Preço unitário",
    "Valor da Operação",
]


class LayoutInesperado(ValueError):
    """O arquivo não tem o layout do extrato de Movimentação da B3."""


def encontrar_extratos(pasta: Path) -> list[Path]:
    """Lista os .xlsx da pasta, em ordem de nome.

    O nome do arquivo da B3 carrega timestamp
    (``movimentacao-2026-11-03-17-17-56.xlsx``), então ordem alfabética é
    ordem cronológica. Arquivos ``~$...`` são locks que o Excel cria ao abrir
    a planilha e precisam ser ignorados.
    """
    pasta = Path(pasta)
    if not pasta.is_dir():
        return []
    return sorted(f for f in pasta.glob("*.xlsx") if not f.name.startswith("~$"))


def validar_layout(df: pd.DataFrame) -> None:
    """Levanta `LayoutInesperado` se as colunas não forem as esperadas."""
    encontradas = list(df.columns)
    if encontradas != COLUNAS_B3:
        raise LayoutInesperado(
            "Layout inesperado no extrato.\n"
            f"  esperado: {COLUNAS_B3}\n"
            f"  recebido: {encontradas}"
        )


def ler_extrato(arquivo: Path) -> pd.DataFrame:
    """Lê um extrato como texto puro e valida o layout.

    `dtype=str` é intencional: a coluna "Preço unitário" contém `-` literal em
    linhas de bonificação e atualização, o que faria o pandas inferir um tipo
    misto na leitura.
    """
    df = pd.read_excel(arquivo, dtype=str)
    validar_layout(df)
    return df


def _com_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Numera linhas idênticas dentro de um mesmo arquivo.

    O extrato não traz ID de transação, e linhas legitimamente repetidas
    existem (dois JCP do mesmo papel, mesma data, mesmo valor). O ordinal
    distingue "a mesma linha vista em dois arquivos" de "duas linhas iguais
    no mesmo arquivo", que é o que permite deduplicar sem perder dado.
    """
    df = df.copy()
    df["_ord"] = df.groupby(COLUNAS_B3, dropna=False).cumcount()
    return df


def carregar_movimentacoes(pasta: Path) -> pd.DataFrame:
    """Lê todos os extratos da pasta e devolve um só DataFrame, deduplicado.

    Downloads sucessivos se sobrepõem — cada export novo repete o histórico
    antigo. A deduplicação usa as oito colunas da B3 mais o ordinal.
    """
    arquivos = encontrar_extratos(pasta)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum extrato .xlsx em {pasta}")

    partes = []
    for arquivo in arquivos:
        parte = _com_ordinal(ler_extrato(arquivo))
        parte["_arquivo"] = arquivo.name
        partes.append(parte)

    todos = pd.concat(partes, ignore_index=True)
    deduplicado = todos.drop_duplicates(subset=COLUNAS_B3 + ["_ord"], keep="first")
    return deduplicado.drop(columns="_ord").reset_index(drop=True)