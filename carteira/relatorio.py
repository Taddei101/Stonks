"""Monta o diagnóstico de risco da carteira.

SEPARAÇÃO DELIBERADA: `analise_risco` devolve estrutura de dados, nunca
texto. `formatar` transforma em texto para o terminal. Quando o Flask
entrar, ele consome `analise_risco` direto e renderiza como quiser — sem
reescrever nada de análise, e sem precisar extrair número de string.

É o mesmo padrão que `rebalance.plano_aporte` e `rebalance.formatar` já
seguem neste projeto.

Sem rede aqui: quem chama traz os retornos prontos.
"""

import pandas as pd

from carteira import componentes as cp
from carteira import risco as rsk
from carteira.blocos import validar_cobertura
from carteira.series import anualizar_volatilidade, retorno_carteira

LIMITE_PAR_CORRELACIONADO = 0.7
ALVO_VARIANCIA_EXPLICADA = 0.90


def analise_risco(
    retornos: pd.DataFrame,
    pesos: pd.Series,
    dimensoes: dict[str, pd.Series] | None = None,
    metodo: str = "ledoit_wolf",
    periodos_no_ano: int = 12,
) -> dict:
    """Diagnóstico completo da estrutura de risco.

    `retornos` são retornos simples por ativo, na frequência de
    `periodos_no_ano` (mensal por padrão). `pesos` é a alocação a analisar —
    posição real ou alvo, quem chama decide.

    Devolve um dicionário com tudo que o relatório precisa. Nada aqui é
    formatado: números são números.
    """
    tickers = [t for t in pesos.index if t in retornos.columns]
    faltando = sorted(set(pesos.index) - set(retornos.columns))
    if not tickers:
        raise ValueError("Nenhum ativo com série de retorno e peso ao mesmo tempo.")

    pesos = pesos.reindex(tickers).astype(float)
    pesos = pesos / pesos.sum()
    retornos = retornos[tickers].dropna(how="any")

    covariancia = rsk.matriz_covariancia(retornos, metodo=metodo)
    correlacao = rsk.matriz_correlacao(retornos)
    cov_anual = rsk.anualizar_covariancia(covariancia, periodos_no_ano)

    intensidade = (
        rsk.encolhimento_ledoit_wolf(retornos)[1]
        if metodo == "ledoit_wolf"
        else 0.0
    )

    serie_carteira = retorno_carteira(retornos, pesos)

    por_bloco = {}
    if dimensoes:
        validar_cobertura(dimensoes, tickers)
        por_bloco = {
            nome: rsk.contribuicao_por_bloco(covariancia, pesos, mapa)
            for nome, mapa in dimensoes.items()
        }

    return {
        "periodo": {
            "inicio": retornos.index.min(),
            "fim": retornos.index.max(),
            "observacoes": len(retornos),
        },
        "metodo": metodo,
        "intensidade_encolhimento": intensidade,
        "pesos": pesos,
        "ignorados": faltando,
        "n_ativos": len(tickers),
        "apostas_efetivas": cp.apostas_efetivas(correlacao),
        "componentes": cp.decomposicao(correlacao),
        "cargas": cp.cargas(correlacao, quantos=3),
        "componentes_para_90": cp.componentes_para_explicar(
            correlacao, ALVO_VARIANCIA_EXPLICADA
        ),
        "pares_correlacionados": cp.pares_mais_correlacionados(
            correlacao, LIMITE_PAR_CORRELACIONADO
        ),
        "correlacao": correlacao,
        "covariancia_anual": cov_anual,
        "volatilidade_anual": rsk.volatilidade_carteira(cov_anual, pesos),
        "volatilidade_realizada": anualizar_volatilidade(
            serie_carteira, periodos_no_ano
        ),
        "contribuicao": rsk.tabela_contribuicao(cov_anual, pesos),
        "por_bloco": por_bloco,
    }


def _barra(fracao: float, largura: int = 20) -> str:
    cheio = int(round(max(0.0, min(1.0, fracao)) * largura))
    return "█" * cheio + "·" * (largura - cheio)


def formatar(analise: dict) -> str:
    """Renderiza o diagnóstico para o terminal."""
    linhas = []
    periodo = analise["periodo"]
    n = analise["n_ativos"]
    efetivas = analise["apostas_efetivas"]

    linhas.append("=" * 68)
    linhas.append("DIAGNÓSTICO DE RISCO")
    linhas.append("=" * 68)
    linhas.append(
        f"Período: {periodo['inicio']:%m/%Y} a {periodo['fim']:%m/%Y}  "
        f"({periodo['observacoes']} observações mensais)"
    )
    linhas.append(
        f"Covariância: {analise['metodo']}"
        + (
            f"  (encolhimento {analise['intensidade_encolhimento']:.1%})"
            if analise["intensidade_encolhimento"]
            else ""
        )
    )
    if analise["ignorados"]:
        linhas.append(
            "Sem série de retorno, fora da análise: "
            + ", ".join(analise["ignorados"])
        )

    if periodo["observacoes"] < 36:
        linhas.append(
            f"AVISO: {periodo['observacoes']} observações é pouco. As estimativas "
            "têm margem de erro larga."
        )

    # ---- a resposta principal ----
    linhas.append("")
    linhas.append("-" * 68)
    linhas.append(
        f"APOSTAS EFETIVAS: {efetivas:.2f}   (de {n} ativos em carteira)"
    )
    linhas.append(
        f"  {analise['componentes_para_90']} componente(s) explicam 90% da variação"
    )
    linhas.append(
        f"  Volatilidade anual estimada: {analise['volatilidade_anual']:.2%}"
        f"   |  realizada: {analise['volatilidade_realizada']:.2%}"
    )
    if efetivas < n * 0.6:
        linhas.append(
            f"  LEITURA: a diversificação é menor do que o número de papéis "
            f"sugere ({efetivas:.1f} contra {n})."
        )

    # ---- componentes ----
    linhas.append("")
    linhas.append("COMPONENTES PRINCIPAIS")
    comp = analise["componentes"]
    for nome, linha in comp.iterrows():
        linhas.append(
            f"  {nome:5s} {_barra(linha['explicada'])} "
            f"{linha['explicada']:6.1%}  (acum. {linha['acumulada']:5.1%})"
        )

    linhas.append("")
    linhas.append("  Cargas (como cada ativo entra em cada componente)")
    cargas = analise["cargas"]
    linhas.append(
        "    " + "ativo".ljust(10) + "".join(c.rjust(9) for c in cargas.columns)
    )
    for ticker, linha in cargas.iterrows():
        linhas.append(
            "    "
            + str(ticker).ljust(10)
            + "".join(f"{v:9.2f}" for v in linha.to_numpy())
        )

    # ---- pares ----
    pares = analise["pares_correlacionados"]
    linhas.append("")
    if len(pares):
        linhas.append(
            f"PARES COM CORRELAÇÃO ACIMA DE {LIMITE_PAR_CORRELACIONADO:.0%}"
        )
        for linha in pares.itertuples():
            linhas.append(f"  {linha.a:8s} x {linha.b:8s}  {linha.correlacao:+.2f}")
    else:
        linhas.append(
            f"Nenhum par acima de {LIMITE_PAR_CORRELACIONADO:.0%} de correlação."
        )

    # ---- capital contra risco ----
    linhas.append("")
    linhas.append("CAPITAL CONTRA RISCO, POR ATIVO")
    linhas.append(
        "  " + "ativo".ljust(9) + "capital".rjust(9) + "risco".rjust(9)
        + "razão".rjust(8) + "   vol.anual"
    )
    for ticker, linha in analise["contribuicao"].iterrows():
        marca = "  <<" if linha["razao"] > 1.25 else ""
        linhas.append(
            f"  {str(ticker).ljust(9)}"
            f"{linha['peso_capital']:8.1%} "
            f"{linha['peso_risco']:8.1%} "
            f"{linha['razao']:7.2f}   "
            f"{linha['volatilidade']:8.1%}{marca}"
        )
    linhas.append("  (razão acima de 1: o ativo ocupa mais risco do que capital)")

    # ---- blocos ----
    for dimensao, tabela in analise["por_bloco"].items():
        linhas.append("")
        linhas.append(f"POR {dimensao.upper()}")
        for categoria, linha in tabela.iterrows():
            linhas.append(
                f"  {str(categoria).ljust(22)}"
                f"capital {linha['peso_capital']:6.1%}   "
                f"risco {linha['peso_risco']:6.1%}   "
                f"razão {linha['razao']:5.2f}"
            )

    linhas.append("")
    linhas.append("=" * 68)
    return "\n".join(linhas)
