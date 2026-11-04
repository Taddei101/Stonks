"""Matrizes de correlação e covariância, com encolhimento.

O PROBLEMA QUE ESTE MÓDULO RESOLVE

Com N ativos é preciso estimar N(N+1)/2 parâmetros de covariância. Com 7
ativos são 28, a partir de uma janela que raramente passa de 36 meses. A
covariância amostral é tecnicamente não enviesada, mas mal condicionada: os
autovalores extremos ficam muito errados — o maior sai grande demais, o
menor pequeno demais.

Isso seria irrelevante se a matriz fosse só olhada. Não é: otimizador de
carteira inverte a matriz, e inverter amplifica justamente o autovalor menor,
que é o pior estimado. O resultado é a patologia clássica de Markowitz —
pesos extremos, soluções de canto, e uma carteira que muda radicalmente
quando se mexe um mês na janela.

A SOLUÇÃO

Encolhimento: combinar a matriz amostral com um alvo estruturado, que tem
viés mas variância muito menor. Ledoit e Wolf mostraram que existe uma
intensidade ótima de combinação, calculável a partir dos próprios dados, e
que ela reduz o erro quadrático de forma substancial.

O alvo usado aqui é o de CORRELAÇÃO CONSTANTE: mantém as variâncias
individuais como estão e substitui toda correlação pela média das
correlações observadas. É o alvo adequado para carteira de ações, que têm
de fato uma correlação média positiva e parecida entre si. O alvo
alternativo, identidade escalonada, supõe correlação zero — premissa que
para ações brasileiras é claramente falsa e puxaria a matriz para longe
demais.

CONVENÇÃO: todas as funções recebem retornos SIMPLES em frequência mensal
(ver `carteira.series`) e devolvem matrizes na mesma frequência. Anualizar
é responsabilidade de quem chama, via `anualizar_covariancia`.
"""

import numpy as np
import pandas as pd

MESES_NO_ANO = 12


class AmostraInsuficiente(ValueError):
    """Observações de menos para estimar a matriz."""


def _matriz_de_retornos(retornos: pd.DataFrame, minimo: int = 3) -> np.ndarray:
    limpos = retornos.dropna(how="any")
    if len(limpos) < minimo:
        raise AmostraInsuficiente(
            f"{len(limpos)} observação(ões) completas, mínimo é {minimo}."
        )
    return limpos.to_numpy(dtype=float)


def matriz_correlacao(retornos: pd.DataFrame) -> pd.DataFrame:
    """Correlação de Pearson entre os ativos."""
    _matriz_de_retornos(retornos)
    return retornos.dropna(how="any").corr()


def covariancia_amostral(retornos: pd.DataFrame) -> pd.DataFrame:
    """Covariância amostral, sem encolhimento.

    Mantida para comparação: é ela que se quer superar. Usa ddof=1, a
    convenção de estimativa não enviesada.
    """
    _matriz_de_retornos(retornos)
    return retornos.dropna(how="any").cov()


def encolhimento_ledoit_wolf(retornos: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Covariância encolhida para o alvo de correlação constante.

    Devolve (matriz, intensidade). A intensidade está em [0, 1]: 0 significa
    confiar inteiramente na amostra, 1 significa descartá-la em favor do
    alvo. Ela cai conforme a janela cresce, que é o comportamento desejado —
    com muitos dados a amostra passa a valer por si.

    Implementação do estimador de Ledoit e Wolf (2003) com alvo de
    correlação constante. Os quatro termos:

      pi     soma das variâncias dos elementos da matriz amostral — quanto
             a amostra é ruidosa
      rho    covariância entre o erro da amostra e o erro do alvo, que
             desconta a parte do ruído que os dois compartilham
      gamma  distância quadrática entre alvo e amostra — quanto o alvo
             está enviesado
      kappa  (pi - rho) / gamma, e a intensidade é kappa / T, truncada

    A lógica é direta: encolhe mais quando a amostra é ruidosa (pi alto) e
    menos quando o alvo está longe dela (gamma alto).

    Aqui a covariância usa divisor T (estimador de máxima verossimilhança),
    não T-1, porque é a convenção sobre a qual as fórmulas de pi, rho e
    gamma foram derivadas.
    """
    limpos = retornos.dropna(how="any")
    x = _matriz_de_retornos(limpos)
    T, N = x.shape

    if N < 2:
        raise AmostraInsuficiente("Encolhimento exige ao menos dois ativos.")

    desvio = x - x.mean(axis=0)
    amostral = desvio.T @ desvio / T

    variancias = np.diag(amostral)

    # Comparar com zero não basta: um ativo constante tem variância da ordem
    # de 1e-36, não 0, porque a média de N cópias do mesmo número não é
    # exatamente esse número em ponto flutuante. O valor passaria pela
    # verificação e a correlação dividiria por ~1e-18, produzindo lixo sem
    # erro. A tolerância é relativa à escala das variâncias observadas.
    escala = float(np.mean(variancias))
    limite = max(escala * 1e-12, 1e-300)
    degenerados = variancias <= limite
    if degenerados.any():
        zerados = [limpos.columns[i] for i in np.where(degenerados)[0]]
        raise AmostraInsuficiente(
            f"Variância praticamente nula em: {zerados}. "
            "Série constante ou preço repetido — confira a origem dos dados."
        )

    dp = np.sqrt(variancias)

    # ---- alvo: correlação constante, variâncias preservadas ----
    correlacao = amostral / np.outer(dp, dp)
    fora_da_diagonal = ~np.eye(N, dtype=bool)
    r_medio = correlacao[fora_da_diagonal].mean()

    alvo = r_medio * np.outer(dp, dp)
    np.fill_diagonal(alvo, variancias)

    # ---- pi: variância de cada elemento da matriz amostral ----
    desvio2 = desvio**2
    pi_matriz = (desvio2.T @ desvio2) / T - amostral**2
    pi = pi_matriz.sum()

    # ---- rho: parte do ruído que alvo e amostra compartilham ----
    # theta[i, j] = cov( d_i^2 , d_i * d_j ), onde d = desvio em relação à média.
    # Expandindo: (1/T) * sum_t d_ti^3 * d_tj - s_ii * s_ij.
    theta = ((desvio2 * desvio).T @ desvio) / T - variancias[:, None] * amostral
    termo_cruzado = (
        np.outer(1.0 / dp, dp) * theta + np.outer(dp, 1.0 / dp) * theta.T
    )
    rho = np.diag(pi_matriz).sum() + (
        (r_medio / 2.0) * termo_cruzado[fora_da_diagonal]
    ).sum()

    # ---- gamma: viés do alvo ----
    gamma = ((alvo - amostral) ** 2).sum()

    if gamma <= 0:
        intensidade = 0.0
    else:
        intensidade = float(np.clip((pi - rho) / gamma / T, 0.0, 1.0))

    encolhida = intensidade * alvo + (1.0 - intensidade) * amostral

    return (
        pd.DataFrame(encolhida, index=limpos.columns, columns=limpos.columns),
        intensidade,
    )


def matriz_covariancia(
    retornos: pd.DataFrame, metodo: str = "ledoit_wolf"
) -> pd.DataFrame:
    """Covariância por um dos métodos disponíveis.

    `metodo` em {"ledoit_wolf", "amostral"}. O padrão é o encolhimento: com
    as janelas típicas deste projeto a amostral não é confiável o suficiente
    para alimentar otimização.
    """
    if metodo == "ledoit_wolf":
        return encolhimento_ledoit_wolf(retornos)[0]
    if metodo == "amostral":
        return covariancia_amostral(retornos)
    raise ValueError(
        f"metodo deve ser 'ledoit_wolf' ou 'amostral', veio {metodo!r}"
    )


def anualizar_covariancia(
    covariancia: pd.DataFrame, periodos_no_ano: int = MESES_NO_ANO
) -> pd.DataFrame:
    """Escala a covariância para base anual.

    Covariância escala linearmente com o tempo, não pela raiz — a raiz vale
    para desvio padrão. Confundir os dois erra a volatilidade por um fator
    de 3,46 em base mensal, o que é grande demais para passar despercebido.
    """
    return covariancia * periodos_no_ano


def volatilidade_carteira(covariancia: pd.DataFrame, pesos: pd.Series) -> float:
    """Desvio padrão da carteira: raiz de wᵀΣw."""
    w = _alinhar_pesos(covariancia, pesos)
    variancia = float(w @ covariancia.to_numpy() @ w)
    return float(np.sqrt(max(variancia, 0.0)))


def _alinhar_pesos(covariancia: pd.DataFrame, pesos: pd.Series) -> np.ndarray:
    """Ordena os pesos na ordem da matriz e normaliza para somar 1."""
    faltando = sorted(set(covariancia.index) - set(pesos.index))
    if faltando:
        raise KeyError(f"Sem peso para: {faltando}")

    w = pesos.reindex(covariancia.index).astype(float)
    soma = float(w.sum())
    if np.isclose(soma, 0.0):
        raise ValueError("Pesos somam zero.")
    return (w / soma).to_numpy()


def e_positiva_definida(matriz: pd.DataFrame, tolerancia: float = 1e-12) -> bool:
    """Toda carteira tem variância estritamente positiva?

    Matriz que falha aqui produz volatilidade negativa sob a raiz e quebra
    qualquer otimização. É a verificação que justifica o encolhimento.
    """
    autovalores = np.linalg.eigvalsh(matriz.to_numpy())
    return bool(autovalores.min() > tolerancia)
