"""Posição por ativo, derivada do extrato.

`posicao` é tabela derivada: nunca é atualizada incrementalmente, sempre
recalculada a partir do extrato mais os ajustes. Descobriu um evento meses
depois? Acrescenta uma linha em ajustes.csv e roda de novo.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from carteira.normalizar import MOV_SNAPSHOT

COLUNAS_AJUSTES = ["data", "ticker", "delta", "nota"]


def carregar_ajustes(caminho: Path) -> pd.DataFrame:
    """Lê os ajustes manuais de quantidade.

    Existem para os eventos que o extrato registra pela metade. A conversão
    CPLE6 -> CPLE5 -> CPLE3, por exemplo: a B3 credita o papel novo via
    `Atualização` mas nunca debita o antigo, então sem ajuste o CPLE6 fica
    eternamente na carteira.

    Desdobramento e bonificação NÃO entram aqui — o extrato já os traz como
    `Bonificação em Ativos`.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        return pd.DataFrame(columns=COLUNAS_AJUSTES).astype(
            {"data": "datetime64[ns]", "ticker": "object", "delta": "float64"}
        )

    df = pd.read_csv(caminho, dtype=str, comment="#").dropna(how="all")
    faltando = set(COLUNAS_AJUSTES) - set(df.columns)
    if faltando:
        raise ValueError(f"ajustes.csv sem as colunas {sorted(faltando)}")

    df["data"] = pd.to_datetime(df["data"], format="%Y-%m-%d")
    df["delta"] = df["delta"].astype(float)
    df["ticker"] = df["ticker"].str.strip()
    return df[COLUNAS_AJUSTES].sort_values("data").reset_index(drop=True)


def construir_ledger(
    movimentacoes: pd.DataFrame, ajustes: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Monta o livro de deltas de quantidade: data, ticker, delta, origem.

    Só linhas da classe `quantidade` entram. Caixa e snapshot ficam de fora
    por construção, o que torna impossível somar uma `Atualização` por engano.
    """
    movimento = movimentacoes[movimentacoes["classe"] == "quantidade"]

    ledger = pd.DataFrame(
        {
            "data": movimento["data"],
            "ticker": movimento["ticker"],
            "delta": movimento["quantidade"] * movimento["sinal"],
            "origem": "extrato",
        }
    )

    if ajustes is not None and len(ajustes):
        ledger = pd.concat(
            [
                ledger,
                pd.DataFrame(
                    {
                        "data": ajustes["data"],
                        "ticker": ajustes["ticker"],
                        "delta": ajustes["delta"],
                        "origem": "ajuste",
                    }
                ),
            ],
            ignore_index=True,
        )

    return ledger.sort_values(["data", "ticker"]).reset_index(drop=True)


def posicao_em(
    ledger: pd.DataFrame, data=None, incluir_zerados: bool = False
) -> pd.Series:
    """Quantidade por ticker numa data (padrão: a última do ledger)."""
    if data is not None:
        ledger = ledger[ledger["data"] <= pd.Timestamp(data)]

    posicao = ledger.groupby("ticker")["delta"].sum().sort_index()
    if not incluir_zerados:
        posicao = posicao[~np.isclose(posicao, 0)]
    return posicao.rename("quantidade")


def serie_posicao(ledger: pd.DataFrame, dias_uteis: bool = False) -> pd.DataFrame:
    """Quantidade por ticker em cada data com movimento (base para o TWR).

    Com `dias_uteis=True`, reindexa para todos os dias úteis do período,
    preenchendo os dias sem movimento — que é o formato que o cálculo de
    retorno vai querer, casado com a série de preços.
    """
    serie = (
        ledger.pivot_table(
            index="data", columns="ticker", values="delta", aggfunc="sum"
        )
        .fillna(0)
        .sort_index()
        .cumsum()
    )

    if dias_uteis and len(serie):
        calendario = pd.date_range(serie.index.min(), serie.index.max(), freq="B")
        serie = serie.reindex(serie.index.union(calendario)).ffill()

    return serie


def conferir_atualizacoes(
    movimentacoes: pd.DataFrame, ledger: pd.DataFrame, tolerancia: float = 1e-6
) -> pd.DataFrame:
    """Confere o saldo derivado contra as linhas de `Atualização` do extrato.

    Essas linhas são a posição que a B3 diz que você tinha naquela data. Toda
    divergência é um evento que o modelo não capturou — normalmente conversão
    de papel, que pede uma linha em ajustes.csv.

    Devolve uma linha por `Atualização`, com a coluna `ok`.
    """
    snapshots = movimentacoes[movimentacoes["movimentacao"].isin(MOV_SNAPSHOT)]

    linhas = []
    for snapshot in snapshots.itertuples():
        derivado = ledger[
            (ledger["ticker"] == snapshot.ticker) & (ledger["data"] <= snapshot.data)
        ]["delta"].sum()
        linhas.append(
            {
                "data": snapshot.data,
                "ticker": snapshot.ticker,
                "informado": snapshot.quantidade,
                "derivado": derivado,
                "dif": snapshot.quantidade - derivado,
            }
        )

    conferencia = pd.DataFrame(linhas, columns=["data", "ticker", "informado", "derivado", "dif"])
    if len(conferencia):
        conferencia["ok"] = np.isclose(conferencia["dif"], 0, atol=tolerancia)
    else:
        conferencia["ok"] = pd.Series(dtype=bool)
    return conferencia.sort_values(["data", "ticker"]).reset_index(drop=True)