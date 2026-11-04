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


def _baixar(simbolos: list[str], ajustado: bool = False, **kwargs) -> pd.DataFrame:
    """Baixa e devolve só o Close, com colunas em ticker puro.

    As duas séries servem a propósitos opostos e NÃO são intercambiáveis:

    `ajustado=False` (padrão) é o que vale para VALOR DE MERCADO da carteira:
    fechamento real vezes quantidade real. Usar ajustado aqui, junto com os
    dividendos que vêm do extrato, contaria o provento duas vezes.

    `ajustado=True` é o que vale para RISCO E RETORNO: a série vem ajustada
    por provento e desdobramento. Sem isso, toda ação que paga bem aparece
    com retorno subestimado e com um degrau artificial na data-ex — degrau
    que contamina a matriz de covariância e, por tabela, toda a análise de
    correlação. Ver `carteira.series`.

    Em qualquer caso o parâmetro é explícito, porque o default do yfinance
    mudou de False para True entre versões.
    """
    import yfinance as yf

    bruto = yf.download(
        simbolos,
        auto_adjust=ajustado,
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
    ajustado: bool = False,
) -> pd.DataFrame:
    """Série de fechamento por ticker, index=data, columns=ticker.

    `ajustado` escolhe entre preço de mercado e série de retorno total — ver
    `_baixar`. Para a série ajustada prefira `historico_total`, que é o mesmo
    com o nome certo.

    O cache é um parquet em formato longo (data, ticker, close, ajustado). A
    política é simples de propósito: se o cache não cobrir toda a janela
    pedida para todos os tickers, rebaixa a janela inteira e reescreve. Com 8
    papéis e dois anos isso é uma requisição — não vale a complexidade de
    baixar só o delta.

    A coluna `ajustado` existe para impedir um erro traiçoeiro: as duas séries
    no mesmo arquivo seriam indistinguíveis, e a leitura devolveria preço
    ajustado onde se esperava preço de mercado, sem erro nenhum. Cache escrito
    antes desta coluna existir é lido como não ajustado, que é o que ele é.
    """
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.DataFrame()

    inicio = pd.Timestamp(inicio)
    fim = pd.Timestamp(fim) if fim is not None else pd.Timestamp.today().normalize()

    if cache is not None and Path(cache).exists():
        longo = pd.read_parquet(cache)
        if "ajustado" not in longo.columns:
            longo["ajustado"] = False
        recorte = longo[
            longo["ticker"].isin(tickers)
            & (longo["data"] >= inicio)
            & (longo["data"] <= fim)
            & (longo["ajustado"] == ajustado)
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
        ajustado=ajustado,
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
            .assign(ajustado=ajustado)
        )
        if cache.exists():
            antigo = pd.read_parquet(cache)
            if "ajustado" not in antigo.columns:
                antigo["ajustado"] = False
            longo = pd.concat([antigo, longo], ignore_index=True)
        longo = longo.drop_duplicates(
            subset=["data", "ticker", "ajustado"], keep="last"
        )
        longo.sort_values(["data", "ticker"]).to_parquet(cache, index=False)

    return close


def historico_total(
    tickers: list[str],
    inicio,
    fim=None,
    cache: Path | None = None,
) -> pd.DataFrame:
    """Série de retorno total: ajustada por provento e desdobramento.

    Esta é a série para análise de risco. NÃO use para valor de carteira —
    para isso existe `historico`, e misturar as duas conta dividendo em
    dobro.

    Use um arquivo de cache diferente do usado por `historico`. Não é
    obrigatório, porque o cache sabe distinguir as duas séries, mas separar
    deixa óbvio qual é qual ao olhar a pasta.
    """
    return historico(tickers, inicio, fim=fim, cache=cache, ajustado=True)


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

    O manual preenche BURACO, não só coluna ausente. Isso importa porque o
    yfinance, para papel deslistado, não omite a coluna: devolve a coluna
    cheia de NaN. Mesclar só "coluna que não existe" descartaria o preço
    manual justamente nos casos para os quais ele foi criado.

    O manual tem uma linha por ticker, propagada para frente. Onde o
    automático tem preço de mercado, ele vence — o manual nunca sobrescreve.
    """
    if manual is None or not len(manual):
        return automatico
    if automatico is None or not len(automatico):
        return manual

    indice = automatico.index.union(manual.index)
    preenchimento = manual.reindex(indice).ffill()
    return automatico.reindex(indice).combine_first(preenchimento).sort_index()


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
