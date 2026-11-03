"""Cotação e histórico de preços via yfinance, com cache em parquet.

Esta é a única camada que conhece o sufixo `.SA`. Em todo o resto do projeto
o ticker é puro (`PETR4`), igual ao que o extrato da B3 emite — misturar os
dois produz join vazio e falha silenciosa.
"""

from pathlib import Path

import pandas as pd

SUFIXO_B3 = ".SA"


class PrecoIndisponivel(RuntimeError):
    """O yfinance não devolveu preço para um ou mais tickers."""


def _simbolo(ticker: str) -> str:
    return f"{ticker}{SUFIXO_B3}"


def _sem_sufixo(simbolo: str) -> str:
    return simbolo[: -len(SUFIXO_B3)] if simbolo.endswith(SUFIXO_B3) else simbolo


def _baixar(simbolos: list[str], **kwargs) -> pd.DataFrame:
    """Baixa e devolve só o Close, com colunas em ticker puro.

    `auto_adjust=False` é obrigatório e explícito: nas versões recentes o
    default do yfinance virou True, e aí o Close vem ajustado por proventos e
    desdobramentos. Para valor de mercado da carteira queremos o fechamento
    real multiplicado pela quantidade real — e usar ajustado junto com os
    dividendos do extrato contaria o provento duas vezes.
    """
    import yfinance as yf

    bruto = yf.download(
        simbolos,
        auto_adjust=False,
        progress=False,
        **kwargs,
    )
    if bruto is None or bruto.empty:
        raise PrecoIndisponivel(f"yfinance não devolveu dados para {simbolos}")

    # O formato das colunas varia conforme a versão e a quantidade de tickers:
    # pode ser MultiIndex (campo, ticker), (ticker, campo), ou plano.
    if isinstance(bruto.columns, pd.MultiIndex):
        if "Close" in bruto.columns.get_level_values(0):
            close = bruto.xs("Close", axis=1, level=0)
        elif "Close" in bruto.columns.get_level_values(1):
            close = bruto.xs("Close", axis=1, level=1)
        else:
            raise PrecoIndisponivel(f"Sem coluna Close em {bruto.columns}")
    else:
        close = bruto[["Close"]]
        close.columns = simbolos[:1]

    close = close.rename(columns={c: _sem_sufixo(str(c)) for c in close.columns})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def cotacao_atual(tickers: list[str]) -> pd.Series:
    """Último fechamento disponível por ticker.

    Usa uma janela de alguns dias e pega o último valor preenchido, para não
    quebrar em feriado, fim de semana ou papel sem negócio no dia.
    """
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.Series(dtype=float, name="preco")

    close = _baixar([_simbolo(t) for t in tickers], period="10d")
    preco = close.ffill().iloc[-1]

    faltando = [t for t in tickers if t not in preco.index or pd.isna(preco.get(t))]
    if faltando:
        raise PrecoIndisponivel(f"Sem cotação para: {faltando}")

    return preco.reindex(tickers).astype(float).rename("preco")


def historico(
    tickers: list[str],
    inicio,
    fim=None,
    cache: Path | None = None,
) -> pd.DataFrame:
    """Série de fechamento por ticker, index=data, columns=ticker.

    O cache é um parquet em formato longo (data, ticker, close). A política é
    simples de propósito: se o cache não cobrir toda a janela pedida para
    todos os tickers, rebaixa a janela inteira e reescreve. Com 8 papéis e
    dois anos isso é uma requisição — não vale a complexidade de baixar só o
    delta.
    """
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.DataFrame()

    inicio = pd.Timestamp(inicio)
    fim = pd.Timestamp(fim) if fim is not None else pd.Timestamp.today().normalize()

    if cache is not None and Path(cache).exists():
        longo = pd.read_parquet(cache)
        recorte = longo[
            longo["ticker"].isin(tickers)
            & (longo["data"] >= inicio)
            & (longo["data"] <= fim)
        ]
        cobre_tickers = set(recorte["ticker"].unique()) >= set(tickers)
        if cobre_tickers and len(recorte):
            cobre_inicio = recorte["data"].min() <= inicio + pd.Timedelta(days=5)
            cobre_fim = recorte["data"].max() >= fim - pd.Timedelta(days=5)
            if cobre_inicio and cobre_fim:
                return (
                    recorte.pivot(index="data", columns="ticker", values="close")
                    .sort_index()
                )

    close = _baixar(
        [_simbolo(t) for t in tickers],
        start=inicio.strftime("%Y-%m-%d"),
        end=(fim + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    if cache is not None:
        cache = Path(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        longo = (
            close.reset_index(names="data")
            .melt(id_vars="data", var_name="ticker", value_name="close")
            .dropna(subset=["close"])
        )
        if cache.exists():
            antigo = pd.read_parquet(cache)
            longo = pd.concat([antigo, longo], ignore_index=True)
        longo = longo.drop_duplicates(subset=["data", "ticker"], keep="last")
        longo.sort_values(["data", "ticker"]).to_parquet(cache, index=False)

    return close


def carregar_precos_manuais(caminho: Path) -> pd.DataFrame:
    """Preços informados à mão, para papel que o yfinance não serve.

    Necessário para papel convertido ou deslistado — CPLE5 e CPLE7 existiram
    por algumas semanas na carteira e não têm série pública. Sem eles o
    cálculo de valor da carteira não fecha naquele intervalo.

    CSV com colunas data,ticker,close. Basta uma linha por ticker: o valor é
    propagado para frente no período em que o papel esteve em carteira.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        return pd.DataFrame()

    longo = pd.read_csv(caminho, dtype=str, comment="#").dropna(how="all")
    if not len(longo):
        return pd.DataFrame()

    faltando = {"data", "ticker", "close"} - set(longo.columns)
    if faltando:
        raise ValueError(f"precos_manuais.csv sem as colunas {sorted(faltando)}")

    longo["data"] = pd.to_datetime(longo["data"], format="%Y-%m-%d")
    longo["close"] = longo["close"].astype(float)
    longo["ticker"] = longo["ticker"].str.strip()

    return longo.pivot(index="data", columns="ticker", values="close").sort_index()


def combinar(automatico: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Junta a série do yfinance com a manual.

    O manual só preenche ticker que o automático não tem — nunca sobrescreve
    preço de mercado.
    """
    if manual is None or not len(manual):
        return automatico
    if automatico is None or not len(automatico):
        return manual

    novos = [c for c in manual.columns if c not in automatico.columns]
    if not novos:
        return automatico

    return automatico.join(manual[novos], how="outer").sort_index()


def desdobramentos(tickers: list[str]) -> pd.DataFrame:
    """Desdobramentos e bonificações conhecidos pelo yfinance, por ticker.

    Serve para cruzar com a série de quantidade: se o yfinance aponta evento
    numa data em que o extrato não registrou ajuste, é sinal de evento não
    modelado. Devolve colunas data, ticker, fator.
    """
    import yfinance as yf

    linhas = []
    for ticker in sorted(set(tickers)):
        serie = yf.Ticker(_simbolo(ticker)).splits
        if serie is None or not len(serie):
            continue
        for data, fator in serie.items():
            linhas.append(
                {
                    "data": pd.Timestamp(data).tz_localize(None),
                    "ticker": ticker,
                    "fator": float(fator),
                }
            )

    return pd.DataFrame(linhas, columns=["data", "ticker", "fator"]).sort_values(
        ["data", "ticker"]
    ).reset_index(drop=True)
