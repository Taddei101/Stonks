"""Desvio da carteira em relação ao alvo e plano de aporte.

Política padrão: só compra. Não sugere venda — evita fato gerador de IR e é
o comportamento usual de buy-and-hold. Como consequência, o desvio só converge
a zero assintoticamente conforme você aporta; ele não zera num aporte só.

Com `permitir_venda=True` (ferramenta do site), quem está acima do alvo vende
o excedente em ações inteiras, arredondando para baixo, e o dinheiro soma ao
aporte. Ticker fora do alvo nunca é vendido: ausência no alvo é tratada como
esquecimento; para zerar uma posição, o alvo precisa dizer 0.

As funções aqui são puras: recebem posição, preços e alvo, e não vão à rede.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Folga aceita na soma dos pesos antes de considerar erro de digitação.
# 0,5pp cobre arredondamento de planilha (os 99,90% típicos).
TOLERANCIA_SOMA = 0.005


class AlvoInvalido(ValueError):
    """O arquivo de alvo não soma 100% dentro da tolerância."""


def carregar_alvo(caminho: Path, avisar=print) -> pd.Series:
    """Lê os pesos alvo e normaliza para somar exatamente 1.

    Ticker aqui é puro, sem `.SA` — o sufixo é assunto do `precos`.
    """
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    alvo = pd.Series(dados, dtype=float).rename("peso_alvo").sort_index()

    if (alvo < 0).any():
        negativos = alvo[alvo < 0].index.tolist()
        raise AlvoInvalido(f"Peso negativo em: {negativos}")

    soma = alvo.sum()
    if abs(soma - 1.0) > TOLERANCIA_SOMA:
        raise AlvoInvalido(
            f"Pesos somam {soma:.4%}, fora da tolerância de "
            f"{TOLERANCIA_SOMA:.1%}. Provável erro de digitação."
        )
    if not np.isclose(soma, 1.0) and avisar is not None:
        avisar(f"Alvo soma {soma:.2%}; normalizado para 100%.")

    return alvo / soma


def _distribuir_troco(
    troco: float, deficit_restante: pd.Series, precos: pd.Series
) -> pd.Series:
    """Gasta o troco comprando uma ação por vez, no maior déficit que couber.

    Sem isso o arredondamento para baixo deixa dinheiro parado — o que é
    bem visível quando a carteira tem papel caro (IVVB11 na casa dos R$ 400).
    """
    extra = pd.Series(0, index=precos.index, dtype=int)
    deficit_restante = deficit_restante.copy()

    while True:
        candidatos = deficit_restante[
            (deficit_restante > 0) & (precos <= troco) & (precos > 0)
        ]
        if candidatos.empty:
            break
        # ordena por déficit e, no empate, por ticker, para ser determinístico
        escolhido = candidatos.sort_values(ascending=False).index[0]
        troco -= precos[escolhido]
        extra[escolhido] += 1
        deficit_restante[escolhido] -= precos[escolhido]

    return extra


def plano_aporte(
    posicao: pd.Series,
    precos: pd.Series,
    alvo: pd.Series,
    aporte: float = 0.0,
    permitir_venda: bool = False,
) -> dict:
    """Compara a carteira com o alvo e distribui o aporte nos mais defasados.

    Devolve dict com:
      tabela          DataFrame por ticker, com desvio e sugestão de compra/venda
      total_atual     valor de mercado hoje
      total_apos      valor depois do aporte
      aporte          o aporte pedido
      vendido         quanto as vendas geraram (0 sem `permitir_venda`)
      investido       quanto do aporte + vendas foi efetivamente alocado
      troco           o que sobrou por indivisibilidade da ação
      fora_do_alvo    tickers em carteira ausentes do alvo (provável esquecimento)
    """
    tickers = sorted(set(posicao.index) | set(alvo.index))

    quantidade = posicao.reindex(tickers).fillna(0.0)
    preco = precos.reindex(tickers)
    if preco.isna().any():
        faltando = preco[preco.isna()].index.tolist()
        raise ValueError(f"Sem preço para: {faltando}")

    peso_alvo = alvo.reindex(tickers).fillna(0.0)

    valor = quantidade * preco
    total_atual = float(valor.sum())
    total_apos = total_atual + aporte

    peso_atual = valor / total_atual if total_atual else valor * 0.0
    valor_alvo = peso_alvo * total_apos
    deficit = (valor_alvo - valor).clip(lower=0)

    vender = pd.Series(0, index=tickers, dtype=int)
    if permitir_venda:
        excesso = (valor - valor_alvo).clip(lower=0).where(peso_alvo.index.isin(alvo.index), 0.0)
        vender = np.floor(excesso / preco).fillna(0).astype(int)
        vender[preco <= 0] = 0
    vendido = float((vender * preco).sum())
    caixa = aporte + vendido

    soma_deficit = float(deficit.sum())
    if caixa > 0 and soma_deficit > 0:
        rateio = caixa * deficit / soma_deficit
    else:
        rateio = pd.Series(0.0, index=tickers)

    acoes = np.floor(rateio / preco).fillna(0).astype(int)
    acoes[preco <= 0] = 0

    gasto = acoes * preco
    troco = caixa - float(gasto.sum())
    if troco > 0 and caixa > 0:
        extra = _distribuir_troco(troco, deficit - gasto, preco)
        acoes = acoes + extra
        gasto = acoes * preco
        troco = caixa - float(gasto.sum())

    tabela = pd.DataFrame(
        {
            "quantidade": quantidade,
            "preco": preco,
            "valor": valor,
            "peso_atual": peso_atual,
            "peso_alvo": peso_alvo,
            "desvio_pp": (peso_atual - peso_alvo) * 100,
            "valor_alvo": valor_alvo,
            "deficit": deficit,
            "comprar": acoes,
            "custo": gasto,
            "vender": vender,
            "receita": vender * preco,
        }
    ).sort_values("desvio_pp")

    fora_do_alvo = [
        t for t in posicao.index if t not in alvo.index and posicao[t] != 0
    ]

    return {
        "tabela": tabela,
        "total_atual": total_atual,
        "total_apos": total_apos,
        "aporte": aporte,
        "vendido": vendido,
        "investido": float(gasto.sum()),
        "troco": troco,
        "fora_do_alvo": fora_do_alvo,
    }


def formatar(plano: dict) -> str:
    """Renderiza o plano para o terminal."""
    tabela = plano["tabela"].copy()
    linhas = []

    linhas.append(f"Valor da carteira: R$ {plano['total_atual']:>12,.2f}")
    if plano["aporte"]:
        linhas.append(f"Aporte:            R$ {plano['aporte']:>12,.2f}")
        linhas.append(f"Total depois:      R$ {plano['total_apos']:>12,.2f}")

    visao = pd.DataFrame(
        {
            "qtd": tabela["quantidade"].map(lambda v: f"{v:,.0f}"),
            "preço": tabela["preco"].map(lambda v: f"{v:,.2f}"),
            "valor": tabela["valor"].map(lambda v: f"{v:,.2f}"),
            "atual%": tabela["peso_atual"].map(lambda v: f"{v:6.2%}"),
            "alvo%": tabela["peso_alvo"].map(lambda v: f"{v:6.2%}"),
            "desvio": tabela["desvio_pp"].map(lambda v: f"{v:+6.2f}pp"),
        }
    )
    movimenta = plano["aporte"] or plano["vendido"]
    if plano["vendido"]:
        visao["vender"] = tabela["vender"].map(lambda v: f"{v:,.0f}" if v else "-")
        visao["receita"] = tabela["receita"].map(lambda v: f"{v:,.2f}" if v else "-")
    if movimenta:
        visao["comprar"] = tabela["comprar"].map(lambda v: f"{v:,.0f}" if v else "-")
        visao["custo"] = tabela["custo"].map(lambda v: f"{v:,.2f}" if v else "-")

    linhas.append("")
    linhas.append(visao.to_string())

    if movimenta:
        linhas.append("")
        vendido = f"Vendido: R$ {plano['vendido']:,.2f}  |  " if plano["vendido"] else ""
        linhas.append(
            f"{vendido}Alocado: R$ {plano['investido']:,.2f}  |  "
            f"Troco: R$ {plano['troco']:,.2f}"
        )

    if plano["fora_do_alvo"]:
        linhas.append("")
        linhas.append(
            "Em carteira e FORA do alvo (tratados como 0%): "
            + ", ".join(plano["fora_do_alvo"])
        )
        linhas.append(
            "  Se é intencional, acrescente com peso 0.0 no alvo.json para "
            "silenciar este aviso."
        )

    return "\n".join(linhas)