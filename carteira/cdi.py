"""Série do CDI, via SGS do Banco Central.

POR QUE ISSO IMPORTA NA COMPARAÇÃO

"A carteira rendeu 14% ao ano" não significa nada sozinho. Significa coisas
opostas se o CDI do período foi 6% ou 13%. No Brasil, onde a taxa básica
passou de 2% a mais de 14% na última década, comparar retorno de ação sem
descontar o custo de oportunidade é o erro mais fácil de cometer.

O CDI entra como taxa livre de risco no índice de Sharpe e como referência
direta: uma carteira de ação que não bate o CDI entregou volatilidade de
graça.

FONTE: série 12 do SGS, taxa diária do CDI em percentual ao dia. Compõe-se
multiplicativamente dentro do mês — não se soma, e não se divide a taxa
anual por 12.

Sem `requests` de propósito: é uma requisição GET de JSON, e a biblioteca
padrão resolve sem acrescentar dependência direta ao projeto.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

SERIE_CDI = 12
URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"
TIMEOUT = 30


class CDIIndisponivel(RuntimeError):
    """Não foi possível obter a série do CDI."""


def _buscar(inicio: pd.Timestamp, fim: pd.Timestamp) -> list[dict]:
    endereco = (
        f"{URL.format(serie=SERIE_CDI)}?formato=json"
        f"&dataInicial={inicio:%d/%m/%Y}&dataFinal={fim:%d/%m/%Y}"
    )
    try:
        with urllib.request.urlopen(endereco, timeout=TIMEOUT) as resposta:
            return json.loads(resposta.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as erro:
        raise CDIIndisponivel(
            f"Não foi possível baixar o CDI do Banco Central: {erro}.\n"
            "A comparação roda sem ele, mas sem Sharpe nem excesso sobre o CDI."
        ) from erro


def cdi_diario(
    inicio, fim=None, cache: Path | None = None
) -> pd.Series:
    """Taxa diária do CDI, em fração decimal (0.0004 = 0,04% ao dia).

    O SGS devolve percentual; a conversão para fração acontece aqui, uma vez
    só, para que o resto do código nunca precise lembrar da escala.
    """
    inicio = pd.Timestamp(inicio)
    fim = pd.Timestamp(fim) if fim is not None else pd.Timestamp.today().normalize()

    if cache is not None and Path(cache).exists():
        guardado = pd.read_parquet(cache)
        serie = guardado.set_index("data")["cdi"]
        recorte = serie[(serie.index >= inicio) & (serie.index <= fim)]
        if len(recorte) and recorte.index.max() >= fim - pd.Timedelta(days=7):
            return recorte.sort_index()

    bruto = _buscar(inicio, fim)
    if not bruto:
        raise CDIIndisponivel(f"SGS devolveu série vazia para {inicio:%d/%m/%Y}.")

    serie = pd.Series(
        {
            pd.to_datetime(linha["data"], format="%d/%m/%Y"): float(linha["valor"])
            / 100.0
            for linha in bruto
        },
        name="cdi",
    ).sort_index()

    if cache is not None:
        cache = Path(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        serie.rename_axis("data").reset_index().to_parquet(cache, index=False)

    return serie


def compor_mensal(diario: pd.Series) -> pd.Series:
    """Compõe a taxa diária dentro de cada mês.

    Multiplicativo, não aditivo: 21 dias a 0,05% dão 1,0558% no mês, não
    1,05%. A diferença é pequena no mês e relevante no acumulado de anos.
    """
    if not len(diario):
        return pd.Series(dtype=float, name="cdi")

    return (
        (1.0 + diario)
        .resample("ME")
        .prod()
        .sub(1.0)
        .rename("cdi")
    )


def cdi_mensal(inicio, fim=None, cache: Path | None = None) -> pd.Series:
    """Atalho: baixa a série diária e devolve o retorno mensal do CDI."""
    return compor_mensal(cdi_diario(inicio, fim, cache=cache))


def alinhar(cdi: pd.Series, indice: pd.Index) -> pd.Series:
    """Alinha a série do CDI ao índice dos retornos da carteira.

    Falha se faltar mês, em vez de preencher com zero. CDI zerado num mês
    inflaria o excesso de retorno sem deixar rastro, e o número sairia
    errado parecendo certo.
    """
    alinhado = cdi.reindex(indice)
    if alinhado.isna().any():
        faltando = alinhado[alinhado.isna()].index
        raise CDIIndisponivel(
            f"Sem CDI para {len(faltando)} mês(es), de "
            f"{faltando.min():%m/%Y} a {faltando.max():%m/%Y}."
        )
    return alinhado
