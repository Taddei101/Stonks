"""Comparação entre alternativas de alocação, com honestidade estatística.

O QUE ESTE MÓDULO EXISTE PARA EVITAR

Uma tabela com "carteira A: 14,2% ao ano / carteira B: 12,8% ao ano" convida
à conclusão de que A é melhor. Com 36 observações mensais, essa diferença é
indistinguível de sorte na quase totalidade dos casos. A tabela parece
precisa e não é.

Por isso toda estatística aqui vem acompanhada de intervalo, e a comparação
entre duas carteiras é reportada como PROBABILIDADE DE SUPERAR, não como
veredito.

REAMOSTRAGEM EM BLOCOS, não em observações isoladas

Reamostrar mês a mês destruiria o agrupamento de volatilidade — a tendência
de mês ruim vir perto de mês ruim. Isso estreitaria artificialmente o
intervalo, que é o oposto do que se quer. Blocos de alguns meses preservam
essa estrutura.

RETORNO GEOMÉTRICO, não aritmético

O que interessa é o patrimônio no fim, e patrimônio compõe. Duas carteiras
com a mesma média aritmética terminam com dinheiro diferente se tiverem
volatilidades diferentes — a mais volátil termina com menos. A média
geométrica já embute esse desconto; a aritmética não, e por isso superestima.
"""

import numpy as np
import pandas as pd

from carteira.series import anualizar_retorno, anualizar_volatilidade

MESES_NO_ANO = 12
TAMANHO_BLOCO = 3
REAMOSTRAGENS = 2000
FRACAO_PIORES = 0.10
NIVEL = 0.90


def perda_maxima(retornos: pd.Series) -> dict:
    """Maior queda do topo ao fundo, e quanto tempo levou para recuperar.

    É a métrica que mais se aproxima da experiência real de carregar a
    carteira — e a que determina se a pessoa aguenta ficar até o fim.
    """
    limpos = pd.Series(retornos).dropna()
    if len(limpos) < 2:
        return {"perda_maxima": float("nan"), "meses_ate_recuperar": float("nan")}

    # O 1,0 inicial é obrigatório. Sem ele, `cummax` começa no patrimônio
    # DEPOIS do primeiro mês, e uma queda que comece logo no primeiro mês
    # aparece como perda zero — o acumulado nunca esteve mais alto.
    patrimonio = np.concatenate([[1.0], (1 + limpos).cumprod().to_numpy()])
    topo = np.maximum.accumulate(patrimonio)
    queda = patrimonio / topo - 1.0

    fundo = int(np.argmin(queda))
    inicio = int(np.argmax(patrimonio[: fundo + 1]))

    recuperados = np.flatnonzero(patrimonio[fundo:] >= patrimonio[inicio])
    meses = float(fundo + recuperados[0] - inicio) if len(recuperados) else float("nan")

    return {"perda_maxima": float(queda.min()), "meses_ate_recuperar": meses}


def media_piores(retornos: pd.Series, fracao: float = FRACAO_PIORES) -> float:
    """Retorno médio dos piores meses — a cauda esquerda.

    Desvio padrão trata alta e baixa igual. Isto olha só o lado que dói, e
    distingue carteira que oscila para os dois lados de carteira que tem
    poucos meses muito ruins.
    """
    limpos = pd.Series(retornos).dropna().sort_values()
    if not len(limpos):
        return float("nan")

    quantos = max(1, int(round(len(limpos) * fracao)))
    return float(limpos.iloc[:quantos].mean())


def sharpe(retornos: pd.Series, cdi: pd.Series | None = None) -> float:
    """Retorno excedente por unidade de volatilidade, em base anual.

    Sem CDI, calcula sobre o retorno bruto e o número passa a medir outra
    coisa — fica comparável entre carteiras, mas não é índice de Sharpe.
    """
    limpos = pd.Series(retornos).dropna()
    if len(limpos) < 2:
        return float("nan")

    excesso = limpos - cdi.reindex(limpos.index) if cdi is not None else limpos
    excesso = excesso.dropna()

    volatilidade = anualizar_volatilidade(excesso, MESES_NO_ANO)
    # Comparar com zero não basta: série constante dá desvio padrão da ordem
    # de 1e-18, não 0, e a divisão devolveria um Sharpe astronômico em vez
    # de dizer que ele é indefinido.
    if np.isnan(volatilidade) or volatilidade < 1e-12:
        return float("nan")

    return anualizar_retorno(excesso, MESES_NO_ANO) / volatilidade


def metricas(retornos: pd.Series, cdi: pd.Series | None = None) -> dict:
    """Painel completo de uma carteira."""
    limpos = pd.Series(retornos).dropna()
    resultado = {
        "meses": len(limpos),
        "retorno_anual": anualizar_retorno(limpos, MESES_NO_ANO),
        "volatilidade_anual": anualizar_volatilidade(limpos, MESES_NO_ANO),
        "sharpe": sharpe(limpos, cdi),
        "media_piores": media_piores(limpos),
    }
    resultado.update(perda_maxima(limpos))

    if cdi is not None:
        alinhado = cdi.reindex(limpos.index).dropna()
        resultado["cdi_anual"] = anualizar_retorno(alinhado, MESES_NO_ANO)
        resultado["excesso_anual"] = (
            resultado["retorno_anual"] - resultado["cdi_anual"]
        )

    return resultado


def _indices_em_blocos(
    n: int, tamanho_bloco: int, rng: np.random.Generator
) -> np.ndarray:
    """Sorteia posições em blocos contíguos, com reposição."""
    if n <= tamanho_bloco:
        return rng.integers(0, n, size=n)

    inicios = rng.integers(0, n - tamanho_bloco + 1, size=n // tamanho_bloco + 1)
    posicoes = np.concatenate(
        [np.arange(inicio, inicio + tamanho_bloco) for inicio in inicios]
    )
    return posicoes[:n]


def intervalo(
    retornos: pd.Series,
    estatistica,
    cdi: pd.Series | None = None,
    reamostragens: int = REAMOSTRAGENS,
    tamanho_bloco: int = TAMANHO_BLOCO,
    nivel: float = NIVEL,
    semente: int = 0,
) -> tuple[float, float]:
    """Intervalo de confiança por reamostragem em blocos.

    `estatistica` recebe (retornos, cdi) e devolve um número. Carteira e CDI
    são reamostrados PELAS MESMAS posições, senão o pareamento entre mês de
    bolsa e mês de juros se perde e o excesso vira ruído.
    """
    limpos = pd.Series(retornos).dropna()
    n = len(limpos)
    if n < 4:
        return (float("nan"), float("nan"))

    valores_carteira = limpos.to_numpy()
    valores_cdi = (
        cdi.reindex(limpos.index).to_numpy() if cdi is not None else None
    )

    rng = np.random.default_rng(semente)
    amostras = []
    for _ in range(reamostragens):
        posicoes = _indices_em_blocos(n, tamanho_bloco, rng)
        serie = pd.Series(valores_carteira[posicoes])
        serie_cdi = (
            pd.Series(valores_cdi[posicoes]) if valores_cdi is not None else None
        )
        amostras.append(estatistica(serie, serie_cdi))

    amostras = np.array([a for a in amostras if np.isfinite(a)])
    if not len(amostras):
        return (float("nan"), float("nan"))

    margem = (1.0 - nivel) / 2.0
    return (
        float(np.quantile(amostras, margem)),
        float(np.quantile(amostras, 1.0 - margem)),
    )


def probabilidade_de_superar(
    candidata: pd.Series,
    referencia: pd.Series,
    cdi: pd.Series | None = None,
    reamostragens: int = REAMOSTRAGENS,
    tamanho_bloco: int = TAMANHO_BLOCO,
    semente: int = 0,
) -> float:
    """Em que fração das reamostragens a candidata tem Sharpe maior.

    Reamostragem PAREADA: as duas carteiras são avaliadas nos mesmos meses
    sorteados. Isso remove o efeito do período em comum e isola a diferença
    entre as alocações, que é a pergunta de interesse.

    Como ler: 0,50 significa moeda ao ar — não há evidência de diferença.
    Valores entre 0,35 e 0,65 devem ser tratados como empate. Só acima de
    0,90 ou abaixo de 0,10 a amostra está dizendo alguma coisa.
    """
    juntas = pd.DataFrame({"a": candidata, "b": referencia}).dropna()
    n = len(juntas)
    if n < 4:
        return float("nan")

    valores_a = juntas["a"].to_numpy()
    valores_b = juntas["b"].to_numpy()
    valores_cdi = (
        cdi.reindex(juntas.index).to_numpy() if cdi is not None else None
    )

    rng = np.random.default_rng(semente)
    vitorias = 0
    validas = 0
    for _ in range(reamostragens):
        posicoes = _indices_em_blocos(n, tamanho_bloco, rng)
        serie_cdi = (
            pd.Series(valores_cdi[posicoes]) if valores_cdi is not None else None
        )
        sa = sharpe(pd.Series(valores_a[posicoes]), serie_cdi)
        sb = sharpe(pd.Series(valores_b[posicoes]), serie_cdi)
        if np.isfinite(sa) and np.isfinite(sb):
            validas += 1
            vitorias += sa > sb

    return vitorias / validas if validas else float("nan")


def comparar(
    series: dict[str, pd.Series],
    cdi: pd.Series | None = None,
    referencia: str = "iguais",
    reamostragens: int = REAMOSTRAGENS,
) -> pd.DataFrame:
    """Painel de todas as carteiras, com intervalo e comparação pareada.

    `referencia` é a carteira contra a qual as outras são comparadas. O
    padrão é 1/N de propósito: é a régua que a literatura mostra ser difícil
    de superar.
    """
    if referencia not in series:
        raise KeyError(f"Carteira de referência {referencia!r} não está na lista.")

    base = series[referencia]
    linhas = {}
    for nome, serie in series.items():
        dados = metricas(serie, cdi)
        baixo, alto = intervalo(
            serie, sharpe, cdi, reamostragens=reamostragens
        )
        dados["sharpe_min"] = baixo
        dados["sharpe_max"] = alto
        dados["prob_superar_ref"] = (
            float("nan")
            if nome == referencia
            else probabilidade_de_superar(
                serie, base, cdi, reamostragens=reamostragens
            )
        )
        linhas[nome] = dados

    return pd.DataFrame(linhas).T.sort_values("sharpe", ascending=False)


def formatar(tabela: pd.DataFrame, referencia: str = "iguais") -> str:
    """Renderiza a comparação para o terminal."""
    linhas = []
    linhas.append("=" * 76)
    linhas.append("COMPARAÇÃO DE ALOCAÇÕES")
    linhas.append("=" * 76)

    meses = int(tabela["meses"].iloc[0])
    linhas.append(f"{meses} observações mensais ({meses / 12:.1f} anos)")
    if "cdi_anual" in tabela.columns:
        linhas.append(f"CDI no período: {tabela['cdi_anual'].iloc[0]:+.2%} ao ano")
    linhas.append("")

    cabecalho = (
        "  " + "carteira".ljust(18) + "retorno".rjust(9) + "vol".rjust(8)
        + "Sharpe".rjust(8) + "   intervalo".ljust(18) + "perda máx".rjust(10)
    )
    linhas.append(cabecalho)
    linhas.append("  " + "-" * 72)

    for nome, linha in tabela.iterrows():
        marca = "  (ref)" if nome == referencia else ""
        linhas.append(
            f"  {str(nome).ljust(18)}"
            f"{linha['retorno_anual']:+8.2%} "
            f"{linha['volatilidade_anual']:7.2%} "
            f"{linha['sharpe']:7.2f}  "
            f"[{linha['sharpe_min']:+.2f}, {linha['sharpe_max']:+.2f}]".ljust(18)
            + f"{linha['perda_maxima']:9.1%}{marca}"
        )

    linhas.append("")
    linhas.append(f"PROBABILIDADE DE SUPERAR {referencia.upper()}")
    linhas.append("  (reamostragem pareada em blocos; 0,50 = moeda ao ar)")
    for nome, linha in tabela.iterrows():
        if nome == referencia or not np.isfinite(linha["prob_superar_ref"]):
            continue
        p = linha["prob_superar_ref"]
        veredito = (
            "diferença real" if p > 0.90 or p < 0.10
            else "empate" if 0.35 <= p <= 0.65
            else "sugestivo, não conclusivo"
        )
        linhas.append(f"  {str(nome).ljust(18)}{p:6.1%}   {veredito}")

    linhas.append("")
    linhas.append("CAUDA")
    for nome, linha in tabela.iterrows():
        meses_rec = linha["meses_ate_recuperar"]
        texto_rec = (
            "não recuperou" if not np.isfinite(meses_rec)
            else f"{int(meses_rec)} meses até recuperar"
        )
        linhas.append(
            f"  {str(nome).ljust(18)}"
            f"média dos 10% piores meses: {linha['media_piores']:+7.2%}   "
            f"{texto_rec}"
        )

    largura = tabela["sharpe_max"] - tabela["sharpe_min"]
    linhas.append("")
    linhas.append(
        f"Largura típica do intervalo de Sharpe: {largura.median():.2f}. "
        "Diferenças menores\nque isso entre carteiras não são distinguíveis "
        "nesta amostra."
    )
    linhas.append("=" * 76)
    return "\n".join(linhas)
