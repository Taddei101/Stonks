"""Saldo de caixa na corretora e separação de aporte externo.

Existe por um motivo só: o extrato não distingue "comprei com dinheiro novo"
de "comprei com dividendo que já estava na conta". As duas aparecem como
`Transferência - Liquidação` Credito. Se todas contarem como aporte, o
dividendo é contado duas vezes — uma como retorno, outra como capital
investido — e o MWR sai subestimado.

A regra é simples: compra consome primeiro o caixa disponível; só o que
faltar é aporte externo.

Hipóteses embutidas, ambas verificáveis no extrato da corretora:
  - os proventos ficaram na corretora (não houve saque para o banco);
  - `Valor da Operação` do JCP é o valor líquido creditado.
Se você saca proventos, o saldo aqui fica superestimado e os aportes
subestimados na mesma medida.
"""

import pandas as pd

MOV_NEGOCIO = "Transferência - Liquidação"


def fluxo_caixa(movimentacoes: pd.DataFrame) -> pd.DataFrame:
    """Percorre o extrato em ordem e devolve o fluxo de caixa linha a linha.

    Colunas: data, movimentacao, ticker, delta_caixa, aporte_externo,
    retirada, saldo_caixa.
    """
    ordenado = movimentacoes.sort_values(
        ["data", "movimentacao", "ticker"]
    ).reset_index(drop=True)

    saldo = 0.0
    linhas = []

    for linha in ordenado.itertuples():
        valor = linha.valor_operacao
        if pd.isna(valor):
            valor = 0.0

        delta_caixa = 0.0
        aporte_externo = 0.0
        retirada = 0.0

        if linha.classe == "caixa":
            # Provento, resgate, leilão de fração: entra dinheiro.
            delta_caixa = valor * linha.sinal
            saldo += delta_caixa

        elif linha.movimentacao == MOV_NEGOCIO:
            if linha.sinal > 0:
                # Compra: gasta o caixa que houver, o resto é dinheiro novo.
                do_caixa = min(saldo, valor)
                aporte_externo = valor - do_caixa
                delta_caixa = -do_caixa
                saldo -= do_caixa
            else:
                # Venda: o dinheiro fica na corretora como caixa.
                delta_caixa = valor
                saldo += valor

        # Bonificação e Fração em Ativos não movem dinheiro.

        linhas.append(
            {
                "data": linha.data,
                "movimentacao": linha.movimentacao,
                "ticker": linha.ticker,
                "delta_caixa": delta_caixa,
                "aporte_externo": aporte_externo,
                "retirada": retirada,
                "saldo_caixa": saldo,
            }
        )

    return pd.DataFrame(
        linhas,
        columns=[
            "data",
            "movimentacao",
            "ticker",
            "delta_caixa",
            "aporte_externo",
            "retirada",
            "saldo_caixa",
        ],
    )


def serie_diaria(fluxo: pd.DataFrame) -> pd.DataFrame:
    """Agrega o fluxo por dia: saldo_caixa (último do dia) e fluxos somados."""
    if not len(fluxo):
        return pd.DataFrame(
            columns=["saldo_caixa", "aporte_externo", "retirada"]
        )

    por_dia = fluxo.groupby("data").agg(
        saldo_caixa=("saldo_caixa", "last"),
        aporte_externo=("aporte_externo", "sum"),
        retirada=("retirada", "sum"),
    )
    return por_dia.sort_index()


def resumo(fluxo: pd.DataFrame) -> dict:
    """Totais que importam para interpretar o retorno."""
    caixa = fluxo[fluxo["delta_caixa"] > 0]
    proventos = caixa[
        caixa["movimentacao"].isin(
            ["Dividendo", "Juros Sobre Capital Próprio", "Rendimento"]
        )
    ]["delta_caixa"].sum()

    return {
        "aporte_externo": float(fluxo["aporte_externo"].sum()),
        "proventos": float(proventos),
        "saldo_caixa": float(fluxo["saldo_caixa"].iloc[-1]) if len(fluxo) else 0.0,
        "reinvestido": float(
            -fluxo.loc[fluxo["delta_caixa"] < 0, "delta_caixa"].sum()
        ),
    }
