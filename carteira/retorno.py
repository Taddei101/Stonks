"""Time-weighted return (TWR) e money-weighted return (MWR).

As duas respondem perguntas diferentes e costumam divergir bastante:

  TWR  mede o desempenho dos ativos, neutralizando o efeito de quando você
       aportou. É o número comparável com o Ibovespa ou com um fundo.
  MWR  mede o retorno do seu dinheiro, levando em conta o tamanho e o momento
       de cada aporte. É o número que responde "o meu dinheiro rendeu quanto".

Em ambas, o valor da carteira inclui o caixa na corretora. Sem isso o
dividendo viraria vazamento: sai do preço da ação, não aparece em lugar
nenhum, e o retorno sai subestimado.

Sem matplotlib aqui de propósito — este módulo é só a matemática.
"""

import numpy as np
import pandas as pd

DIAS_NO_ANO = 365.0


def serie_valor(
    posicoes: pd.DataFrame,
    precos: pd.DataFrame,
    caixa_diario: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Monta a série diária de valor da carteira.

    Alinha quantidade, preço e caixa num calendário de dias úteis comum,
    propagando para frente o último valor conhecido de cada um — quantidade
    só muda quando há movimento, e preço não existe em feriado.

    Devolve colunas: acoes, caixa, total, aporte_externo.
    """
    if not len(posicoes) or not len(precos):
        return pd.DataFrame(columns=["acoes", "caixa", "total", "aporte_externo"])

    inicio = posicoes.index.min()
    fim = max(posicoes.index.max(), precos.index.max())
    calendario = pd.date_range(inicio, fim, freq="B")

    qtd = posicoes.reindex(posicoes.index.union(calendario)).ffill().reindex(calendario)
    qtd = qtd.fillna(0.0)

    px = precos.reindex(precos.index.union(calendario)).ffill().reindex(calendario)

    tickers = [t for t in qtd.columns if t in px.columns]
    problemas = [f"{t} (coluna ausente)" for t in sorted(set(qtd.columns) - set(px.columns))]

    # Coluna presente mas vazia é o caso traiçoeiro: para papel deslistado o
    # yfinance devolve a coluna cheia de NaN em vez de erro. Como a soma do
    # pandas ignora NaN, o papel seria avaliado em ZERO sem reclamar. Por isso
    # a checagem é por NaN no período em que o papel esteve em carteira, não
    # por ausência de coluna.
    for ticker in tickers:
        em_carteira = qtd[ticker] != 0
        if em_carteira.any() and px.loc[em_carteira, ticker].isna().any():
            periodo = qtd.index[em_carteira]
            problemas.append(
                f"{ticker} (sem cotação entre "
                f"{periodo.min():%d/%m/%Y} e {periodo.max():%d/%m/%Y})"
            )

    if problemas:
        raise ValueError(
            "Sem série de preço para: " + "; ".join(problemas) + ".\n"
            "Papel convertido ou deslistado não tem série no yfinance — "
            "acrescente uma linha data,ticker,close em config/precos_manuais.csv."
        )

    acoes = (qtd[tickers] * px[tickers]).sum(axis=1)

    if caixa_diario is not None and len(caixa_diario):
        caixa = (
            caixa_diario["saldo_caixa"]
            .reindex(caixa_diario.index.union(calendario))
            .ffill()
            .reindex(calendario)
            .fillna(0.0)
        )
        aporte = (
            caixa_diario["aporte_externo"]
            .reindex(calendario)
            .fillna(0.0)
        )
    else:
        caixa = pd.Series(0.0, index=calendario)
        aporte = pd.Series(0.0, index=calendario)

    return pd.DataFrame(
        {
            "acoes": acoes,
            "caixa": caixa,
            "total": acoes + caixa,
            "aporte_externo": aporte,
        }
    )


def twr(valores: pd.DataFrame) -> pd.Series:
    """Retorno acumulado time-weighted, dia a dia.

    Convenção: o fluxo externo do dia é tratado como ocorrido no fim do dia,
    então o retorno do dia é (V_t - F_t) / V_{t-1} - 1. Isso significa que
    comprar a preço de mercado não cria retorno artificial.
    """
    total = valores["total"]
    fluxo = valores["aporte_externo"]
    anterior = total.shift(1)

    retorno_diario = ((total - fluxo) / anterior - 1).where(anterior > 0)
    return ((1 + retorno_diario.fillna(0)).cumprod() - 1).rename("twr")


def _vpl(taxa: float, dias: np.ndarray, valores: np.ndarray) -> float:
    return float(np.sum(valores / (1.0 + taxa) ** (dias / DIAS_NO_ANO)))


def xirr(
    datas, valores, limites=(-0.999999, 100.0), tolerancia=1e-10, max_iter=300
) -> float:
    """Taxa interna de retorno anualizada para fluxos em datas irregulares.

    Bisseção em vez de Newton, e sem scipy: é mais lento e não importa com
    poucos fluxos, mas não diverge e não adiciona dependência.
    """
    datas = pd.to_datetime(pd.Series(list(datas))).reset_index(drop=True)
    valores = np.asarray(list(valores), dtype=float)

    if len(valores) < 2:
        return float("nan")
    if not (valores > 0).any() or not (valores < 0).any():
        # Sem troca de sinal não existe TIR.
        return float("nan")

    dias = (datas - datas.iloc[0]).dt.days.to_numpy(dtype=float)

    baixo, alto = limites
    f_baixo, f_alto = _vpl(baixo, dias, valores), _vpl(alto, dias, valores)
    if f_baixo * f_alto > 0:
        return float("nan")

    for _ in range(max_iter):
        meio = (baixo + alto) / 2
        f_meio = _vpl(meio, dias, valores)
        if abs(f_meio) < tolerancia or (alto - baixo) < tolerancia:
            return meio
        if f_baixo * f_meio < 0:
            alto, f_alto = meio, f_meio
        else:
            baixo, f_baixo = meio, f_meio

    return (baixo + alto) / 2


def mwr(valores: pd.DataFrame) -> dict:
    """MWR a partir dos aportes externos e do valor final da carteira.

    Proventos NÃO entram como fluxo: eles ficam no caixa, que está dentro do
    valor total. Entram como retorno, que é o lugar certo.
    """
    aportes = valores.loc[valores["aporte_externo"] > 0, "aporte_externo"]
    if not len(aportes):
        return {"mwr_anual": float("nan"), "aportado": 0.0, "valor_final": 0.0}

    valor_final = float(valores["total"].iloc[-1])
    fim = valores.index[-1]

    # Agrupa por data antes de resolver: um aporte no próprio dia da avaliação
    # final cairia na mesma data do valor terminal, e dois fluxos na mesma
    # data tornam a TIR indefinida. Somados, o fluxo líquido do dia é correto.
    fluxos = -aportes.copy()
    fluxos.loc[fim] = fluxos.get(fim, 0.0) + valor_final
    fluxos = fluxos.groupby(level=0).sum().sort_index()

    taxa = xirr(fluxos.index, fluxos.to_numpy())
    dias = (fim - aportes.index[0]).days

    return {
        "mwr_anual": taxa,
        "mwr_periodo": (1 + taxa) ** (dias / DIAS_NO_ANO) - 1
        if not np.isnan(taxa)
        else float("nan"),
        "aportado": float(aportes.sum()),
        "valor_final": valor_final,
        "dias": dias,
    }


def resumo(valores: pd.DataFrame) -> dict:
    """Junta TWR e MWR num dicionário só, para imprimir ou plotar."""
    serie = twr(valores)
    dados = mwr(valores)
    dias = (valores.index[-1] - valores.index[0]).days

    twr_total = float(serie.iloc[-1])
    dados["twr_periodo"] = twr_total
    dados["twr_anual"] = (
        (1 + twr_total) ** (DIAS_NO_ANO / dias) - 1 if dias > 0 else float("nan")
    )
    dados["serie_twr"] = serie
    return dados
