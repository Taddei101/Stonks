"""Ponto de entrada.

    python main.py                 # posição e desvio em relação ao alvo
    python main.py --aporte 2000   # distribui o aporte nos mais defasados
    python main.py --grafico       # gera os PNGs (retorno e proventos do ano)
    python main.py --grafico --ano 2025   # proventos de outro ano
    python main.py --grafico --ano 0      # proventos: histórico completo
    python main.py --risco         # diagnóstico de risco da posição atual
    python main.py --risco --alvo  # mesmo diagnóstico, sobre os pesos alvo
    python main.py --risco --janela 5     # outra janela, em anos
    python main.py --comparar      # compara regras de alocação
    python main.py --comparar --janela 5  # idem, em outra janela
    python main.py --validar       # mesma comparação, mas fora da amostra
    python main.py --validar --modo ancorada --treino 48
    python main.py --comparar --tickers PETR4,VALE3,WEGE3,IVVB11
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
ARQUIVO_BLOCOS = CONFIG / "blocos.json"
CACHE_PRECOS = RAIZ / ".cache" / "precos.parquet"
# Cache separado: a série ajustada por provento não é intercambiável com a
# série de valor de mercado, e manter arquivos distintos deixa óbvio qual é
# qual ao olhar a pasta.
CACHE_PRECOS_TOTAL = RAIZ / ".cache" / "precos_total.parquet"
CACHE_CDI = RAIZ / ".cache" / "cdi.parquet"
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


def rodar_risco(ledger, usar_alvo: bool, janela_anos: float) -> None:
    """Diagnóstico de estrutura de risco da carteira.

    Usa a série AJUSTADA por provento e desdobramento — a de valor de
    mercado produziria correlação contaminada pelo degrau da data-ex.
    """
    from carteira.blocos import carregar_blocos
    from carteira.precos import historico_total
    from carteira.relatorio import analise_risco, formatar as formatar_risco
    from carteira.series import retornos_mensais

    alvo = carregar_alvo(ARQUIVO_ALVO)
    if usar_alvo:
        pesos = alvo
        origem = "pesos alvo (config/alvo.json)"
    else:
        posicao = posicao_em(ledger)
        precos = cotacao_atual(sorted(posicao.index))
        valor = posicao * precos.reindex(posicao.index)
        pesos = (valor / valor.sum()).rename("peso")
        origem = "posição atual, a preço de mercado"

    inicio = pd.Timestamp.today().normalize() - pd.DateOffset(
        days=round(janela_anos * 365)
    )
    precos_totais = historico_total(
        sorted(pesos.index), inicio, cache=CACHE_PRECOS_TOTAL
    )

    analise = analise_risco(
        retornos_mensais(precos_totais),
        pesos,
        dimensoes=carregar_blocos(ARQUIVO_BLOCOS),
    )

    print()
    print(f"Pesos: {origem}")
    print(formatar_risco(analise))


def resolver_universo(tickers: str | None):
    """Decide quais papéis entram na análise, e com que pesos de referência.

    Sem `--tickers`, usa o alvo. Com `--tickers`, usa a lista dada e o alvo
    deixa de participar — é o que permite testar um conjunto hipotético sem
    mexer no arquivo que comanda o `--aporte`.

    Devolve (lista de tickers, pesos do alvo ou None).
    """
    if not tickers:
        alvo = carregar_alvo(ARQUIVO_ALVO)
        return sorted(alvo.index), alvo

    lista = sorted({t.strip().upper() for t in tickers.split(",") if t.strip()})
    if len(lista) < 2:
        raise SystemExit("--tickers precisa de pelo menos dois papéis.")
    return lista, None


def rodar_comparacao(
    ledger, janela_anos: float, sem_cdi: bool, tickers: str | None = None
) -> None:
    """Compara regras de alocação sobre o mesmo período.

    Todas as regras são calculadas sobre a MESMA matriz de covariância
    encolhida, para que a diferença entre elas venha da regra e não de
    estimativas distintas da estrutura de risco.
    """
    from carteira.alocacoes import todas
    from carteira.cdi import CDIIndisponivel, alinhar, cdi_mensal
    from carteira.comparar import comparar, formatar as formatar_comparacao
    from carteira.precos import historico_total
    from carteira.risco import matriz_covariancia
    from carteira.series import retorno_carteira, retornos_mensais

    universo, alvo = resolver_universo(tickers)
    inicio = pd.Timestamp.today().normalize() - pd.DateOffset(
        days=round(janela_anos * 365)
    )
    precos = historico_total(universo, inicio, cache=CACHE_PRECOS_TOTAL)
    retornos = retornos_mensais(precos, avisar=print)

    carteiras = todas(matriz_covariancia(retornos))
    if alvo is not None:
        carteiras["alvo_atual"] = alvo.reindex(retornos.columns).fillna(0.0)
    else:
        print(f"\nUniverso: {', '.join(universo)} (alvo.json fora da análise)")

    series = {
        nome: retorno_carteira(retornos, pesos)
        for nome, pesos in carteiras.items()
    }

    cdi = None
    if not sem_cdi:
        try:
            # O indice dos retornos traz o ULTIMO dia de cada mes. Pedir o
            # CDI a partir dessa data deixa o primeiro mes com um pedaco de
            # mes - ou com nada, se o dia 31 cair num domingo, que foi o que
            # aconteceu com 31/12/2023. Pede-se desde o primeiro dia do mes.
            primeiro_dia = retornos.index.min().replace(day=1)
            cdi = alinhar(
                cdi_mensal(primeiro_dia, cache=CACHE_CDI), retornos.index
            )
        except CDIIndisponivel as erro:
            print(f"\nSem CDI: {erro}")

    print()
    print(formatar_comparacao(comparar(series, cdi)))

    print()
    print("PESOS DE CADA REGRA")
    print(pd.DataFrame(carteiras).map(lambda v: f"{v:6.1%}").to_string())


def rodar_validacao(
    janela_anos: float, treino: int, modo: str, sem_cdi: bool,
    tickers: str | None = None,
) -> None:
    """Validação deslizante: estima com o passado, mede no futuro.

    Diferente de `--comparar`, aqui nenhuma regra é avaliada na janela em
    que foi estimada. Os números saem bem menos impressionantes, e é esse o
    ponto.
    """
    from carteira.alocacoes import regras_sobre_retornos
    from carteira.cdi import CDIIndisponivel, alinhar, cdi_mensal
    from carteira.precos import historico_total
    from carteira.series import retornos_mensais
    from carteira.validacao import deslizante, formatar as formatar_validacao, resumo

    universo, alvo = resolver_universo(tickers)
    inicio = pd.Timestamp.today().normalize() - pd.DateOffset(
        days=round(janela_anos * 365)
    )
    precos = historico_total(universo, inicio, cache=CACHE_PRECOS_TOTAL)
    retornos = retornos_mensais(precos, avisar=print)

    regras = regras_sobre_retornos()
    if alvo is not None:
        pesos_alvo = alvo.reindex(retornos.columns).fillna(0.0)
        # O alvo é fixo por definição: não se reestima, então entra na
        # validação como uma regra que ignora os dados. É a comparação mais
        # honesta possível, porque ele também não pagou custo de estimativa.
        regras["alvo_atual"] = lambda _dados, p=pesos_alvo: p

    resultado = deslizante(retornos, regras, treino=treino, modo=modo)

    cdi = None
    if not sem_cdi:
        try:
            indice = resultado["retornos"].index
            cdi = alinhar(
                cdi_mensal(indice.min().replace(day=1), cache=CACHE_CDI), indice
            )
        except CDIIndisponivel as erro:
            print(f"\nSem CDI: {erro}")

    print()
    print(formatar_validacao(resultado, resumo(resultado, cdi)))


def rodar_grafico(movimentacoes, ledger, ano: int | None) -> None:
    from carteira.graficos import (
        grafico_proventos_ano,
        grafico_proventos_mensais,
        grafico_retorno,
    )

    # Proventos não dependem de preço, então este gráfico sai mesmo que o
    # yfinance esteja fora do ar.
    linhas_proventos = movimentacoes[
        movimentacoes["movimentacao"].isin(PROVENTOS)
    ][["data", "valor_operacao"]]

    if ano == 0:
        caminho_proventos = grafico_proventos_mensais(
            linhas_proventos, SAIDA_PROVENTOS
        )
    else:
        caminho_proventos = grafico_proventos_ano(
            linhas_proventos, SAIDA_PROVENTOS, ano=ano
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


def main(
    aporte: float,
    grafico: bool,
    ano: int | None = None,
    risco: bool = False,
    usar_alvo: bool = False,
    janela: float = 3.0,
    comparar_alocacoes: bool = False,
    sem_cdi: bool = False,
    tickers: str | None = None,
    validar: bool = False,
    treino: int = 36,
    modo: str = "rolante",
) -> None:
    movimentacoes, ledger = carregar_tudo()

    print(
        f"{len(movimentacoes)} movimentações, "
        f"{movimentacoes['data'].min():%d/%m/%Y} a "
        f"{movimentacoes['data'].max():%d/%m/%Y}"
    )
    mostrar_conferencia(movimentacoes, ledger)

    if validar:
        rodar_validacao(janela, treino, modo, sem_cdi, tickers)
    elif comparar_alocacoes:
        rodar_comparacao(ledger, janela, sem_cdi, tickers)
    elif risco:
        rodar_risco(ledger, usar_alvo, janela)
    elif grafico:
        rodar_grafico(movimentacoes, ledger, ano)
    else:
        rodar_rebalanceamento(ledger, aporte)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    parser = argparse.ArgumentParser(description="Carteira: posição, rebalanceamento e retorno")
    parser.add_argument("--aporte", type=float, default=0.0,
                        help="Valor a aportar; sem ele mostra só o desvio atual")
    parser.add_argument("--grafico", action="store_true",
                        help="Gera os PNGs em vez do rebalanceamento")
    parser.add_argument("--ano", type=int, default=None,
                        help="Ano do gráfico de proventos (padrão: ano atual; "
                             "use 0 para o histórico completo)")
    parser.add_argument("--risco", action="store_true",
                        help="Diagnóstico de risco: apostas efetivas, "
                             "componentes principais e contribuição por bloco")
    parser.add_argument("--alvo", action="store_true",
                        help="Com --risco: analisa os pesos alvo em vez da "
                             "posição atual")
    parser.add_argument("--janela", type=float, default=3.0,
                        help="Com --risco: janela de histórico em anos "
                             "(padrão: 3)")
    parser.add_argument("--comparar", action="store_true",
                        help="Compara regras de alocação (1/N, inverso da "
                             "volatilidade, paridade de risco, mínima "
                             "variância) com o alvo atual")
    parser.add_argument("--validar", action="store_true",
                        help="Validação deslizante: estima com o passado e "
                             "mede no futuro, sem circularidade")
    parser.add_argument("--treino", type=int, default=36,
                        help="Com --validar: meses de treino (padrão: 36)")
    parser.add_argument("--modo", type=str, default="rolante",
                        choices=["rolante", "ancorada"],
                        help="Com --validar: janela de tamanho fixo (rolante) "
                             "ou crescente (ancorada)")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Lista separada por vírgula para analisar em vez "
                             "do alvo.json, p.ex. PETR4,VALE3,WEGE3. Não "
                             "altera o alvo nem o --aporte")
    parser.add_argument("--sem-cdi", action="store_true",
                        help="Com --comparar: não busca o CDI no Banco Central")
    args = parser.parse_args()
    main(args.aporte, args.grafico, args.ano, args.risco, args.alvo, args.janela,
         args.comparar, args.sem_cdi, args.tickers, args.validar,
         args.treino, args.modo)
