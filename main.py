"""Ponto de entrada.

    python main.py                 # posição e desvio em relação ao alvo
    python main.py --aporte 2000   # distribui o aporte nos mais defasados
    python main.py --grafico       # gera o PNG de retorno (TWR, MWR, proventos)
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from carteira.caixa import fluxo_caixa, resumo as resumo_caixa, serie_diaria
from carteira.io_extrato import carregar_movimentacoes
from carteira.normalizar import normalizar
from carteira.posicao import (
    carregar_ajustes,
    conferir_atualizacoes,
    construir_ledger,
    posicao_em,
    serie_posicao,
)
from carteira.precos import (
    carregar_precos_manuais,
    combinar,
    cotacao_atual,
    historico,
)
from carteira.rebalance import carregar_alvo, formatar, plano_aporte
from carteira.retorno import resumo as resumo_retorno, serie_valor

# O export da B3 não declara estilo de célula padrão; o aviso é inofensivo.
warnings.filterwarnings(
    "ignore", message="Workbook contains no default style", category=UserWarning
)

RAIZ = Path(__file__).parent
PASTA_EXTRATOS = RAIZ / "extratos"
CONFIG = RAIZ / "config"
ARQUIVO_AJUSTES = CONFIG / "ajustes.csv"
ARQUIVO_PENDENTES = CONFIG / "pendentes.csv"
ARQUIVO_ALVO = CONFIG / "alvo.json"
ARQUIVO_PRECOS_MANUAIS = CONFIG / "precos_manuais.csv"
CACHE_PRECOS = RAIZ / ".cache" / "precos.parquet"
SAIDA_RETORNO = RAIZ / "saida" / "retorno.png"
SAIDA_PROVENTOS = RAIZ / "saida" / "proventos.png"

PROVENTOS = ["Dividendo", "Juros Sobre Capital Próprio", "Rendimento"]


def carregar_tudo():
    """Devolve (movimentações normalizadas, ledger de quantidade)."""
    movimentacoes = normalizar(carregar_movimentacoes(PASTA_EXTRATOS))
    ajustes = pd.concat(
        [
            carregar_ajustes(ARQUIVO_AJUSTES),
            # Compras ainda não liquidadas: o extrato usa data de liquidação
            # (D+2), então o negócio de hoje só aparece depois de amanhã.
            carregar_ajustes(ARQUIVO_PENDENTES),
        ],
        ignore_index=True,
    )
    return movimentacoes, construir_ledger(movimentacoes, ajustes)


def mostrar_conferencia(movimentacoes, ledger) -> None:
    conferencia = conferir_atualizacoes(movimentacoes, ledger)
    divergentes = conferencia[~conferencia["ok"]]
    if len(divergentes):
        print(
            f"\nATENÇÃO: {len(divergentes)} linha(s) de Atualização divergem "
            "do saldo derivado — evento não modelado:"
        )
        print(divergentes.to_string(index=False))
        print("Acrescente o ajuste em config/ajustes.csv.")
    else:
        print(
            f"Conferência: {len(conferencia)}/{len(conferencia)} Atualizações batem."
        )


def rodar_rebalanceamento(ledger, aporte: float) -> None:
    posicao = posicao_em(ledger)
    alvo = carregar_alvo(ARQUIVO_ALVO)
    precos = cotacao_atual(sorted(set(posicao.index) | set(alvo.index)))
    print()
    print(formatar(plano_aporte(posicao, precos, alvo, aporte)))


def rodar_grafico(movimentacoes, ledger) -> None:
    from carteira.graficos import grafico_proventos_mensais, grafico_retorno

    # Proventos não dependem de preço, então este gráfico sai mesmo que o
    # yfinance esteja fora do ar.
    linhas_proventos = movimentacoes[
        movimentacoes["movimentacao"].isin(PROVENTOS)
    ][["data", "valor_operacao"]]
    caminho_proventos = grafico_proventos_mensais(
        linhas_proventos, SAIDA_PROVENTOS
    )

    fluxo = fluxo_caixa(movimentacoes)
    caixa_diario = serie_diaria(fluxo)
    posicoes = serie_posicao(ledger)

    tickers = sorted(posicoes.columns)
    precos = combinar(
        historico(tickers, posicoes.index.min(), cache=CACHE_PRECOS),
        carregar_precos_manuais(ARQUIVO_PRECOS_MANUAIS),
    )

    valores = serie_valor(posicoes, precos, caixa_diario)
    dados = resumo_retorno(valores)
    totais = resumo_caixa(fluxo)

    proventos = (
        movimentacoes[movimentacoes["movimentacao"].isin(PROVENTOS)]
        .groupby("data")["valor_operacao"]
        .sum()
        .reindex(valores.index)
        .fillna(0)
        .cumsum()
    )

    print()
    print(f"Período:          {valores.index[0]:%d/%m/%Y} a "
          f"{valores.index[-1]:%d/%m/%Y} ({dados['dias']} dias)")
    print(f"Aporte externo:   R$ {totais['aporte_externo']:>12,.2f}")
    print(f"Reinvestido:      R$ {totais['reinvestido']:>12,.2f}  "
          "(compras pagas com caixa da corretora)")
    print(f"Proventos:        R$ {totais['proventos']:>12,.2f}")
    print(f"Caixa parado:     R$ {totais['saldo_caixa']:>12,.2f}")
    print(f"Valor da carteira:R$ {dados['valor_final']:>12,.2f}")
    print()
    print(f"TWR: {dados['twr_periodo']:+.2%} no período  |  "
          f"{dados['twr_anual']:+.2%} ao ano")
    print(f"MWR: {dados['mwr_periodo']:+.2%} no período  |  "
          f"{dados['mwr_anual']:+.2%} ao ano")

    caminho_retorno = grafico_retorno(valores, dados, proventos, SAIDA_RETORNO)
    print(f"\nGráficos: {caminho_retorno}")
    print(f"          {caminho_proventos}")


def main(aporte: float, grafico: bool) -> None:
    movimentacoes, ledger = carregar_tudo()

    print(
        f"{len(movimentacoes)} movimentações, "
        f"{movimentacoes['data'].min():%d/%m/%Y} a "
        f"{movimentacoes['data'].max():%d/%m/%Y}"
    )
    mostrar_conferencia(movimentacoes, ledger)

    if grafico:
        rodar_grafico(movimentacoes, ledger)
    else:
        rodar_rebalanceamento(ledger, aporte)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    parser = argparse.ArgumentParser(description="Carteira: posição, rebalanceamento e retorno")
    parser.add_argument("--aporte", type=float, default=0.0,
                        help="Valor a aportar; sem ele mostra só o desvio atual")
    parser.add_argument("--grafico", action="store_true",
                        help="Gera o PNG de retorno em vez do rebalanceamento")
    args = parser.parse_args()
    main(args.aporte, args.grafico)
