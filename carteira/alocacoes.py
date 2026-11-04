"""Regras de alocação que não exigem estimar retorno esperado.

POR QUE NENHUMA DELAS USA RETORNO ESPERADO

Estimar retorno médio futuro é o ponto mais fraco de qualquer otimização de
carteira. Com 36 observações mensais, o intervalo de confiança em torno da
média é tão largo que engloba quase qualquer valor plausível — e o erro na
média custa muito mais, em resultado final, que erro equivalente na
variância (Chopra e Ziemba, 1993). O otimizador de Markowitz amplifica
exatamente esse erro: ele carrega peso no ativo que PARECEU render mais, e
"pareceu" é ruído na maior parte das vezes.

As regras aqui usam só a estrutura de risco, que é estimável com margem
razoável na mesma amostra. Isso é abrir mão de perseguir retorno. A aposta
implícita é que uma carteira estável e bem estruturada bate uma carteira
otimizada sobre estimativas ruins — que é o que a literatura de comparação
com 1/N vem mostrando há décadas.

Todas devolvem pesos que somam 1, sem posição vendida.
"""

import numpy as np
import pandas as pd

MAX_ITERACOES = 10_000
TOLERANCIA = 1e-10
# Desvio ainda aceitavel quando a iteracao estaciona antes da tolerancia
# ideal. 1e-6 na fracao de risco e irrelevante para qualquer decisao.
TOLERANCIA_ACEITAVEL = 1e-6


class NaoConvergiu(RuntimeError):
    """A otimização não convergiu dentro do limite de iterações."""


def _validar(covariancia: pd.DataFrame) -> np.ndarray:
    matriz = covariancia.to_numpy(dtype=float)
    if matriz.shape[0] != matriz.shape[1]:
        raise ValueError("Matriz de covariância não é quadrada.")
    if not np.allclose(matriz, matriz.T, atol=1e-10):
        raise ValueError("Matriz de covariância não é simétrica.")
    return matriz


def _serie(pesos: np.ndarray, covariancia: pd.DataFrame, nome: str) -> pd.Series:
    return pd.Series(pesos, index=covariancia.index, name=nome)


def projetar_no_simplex(v: np.ndarray) -> np.ndarray:
    """Ponto do simplex mais próximo de `v`: pesos não negativos somando 1.

    Algoritmo de Duchi et al. (2008). Ordena decrescente, encontra o maior
    índice cujo valor ainda supera a média acumulada ajustada, e desloca
    tudo por esse limiar, zerando o que ficar negativo.

    Projetar é o que permite otimizar com restrição sem usar programação
    quadrática: a cada passo o gradiente leva para fora do conjunto viável,
    e a projeção traz de volta pelo caminho mais curto.
    """
    n = len(v)
    ordenado = np.sort(v)[::-1]
    acumulado = np.cumsum(ordenado)
    indices = np.arange(1, n + 1)

    viaveis = ordenado - (acumulado - 1) / indices > 0
    if not viaveis.any():
        return np.full(n, 1.0 / n)

    rho = indices[viaveis][-1]
    limiar = (acumulado[rho - 1] - 1) / rho
    return np.maximum(v - limiar, 0.0)


def pesos_iguais(covariancia: pd.DataFrame) -> pd.Series:
    """1/N. A referência difícil de superar.

    Não usa nem a covariância, só a lista de ativos. Está aqui como régua:
    regra que não bate 1/N depois de custos não está agregando nada.
    """
    n = len(covariancia)
    return _serie(np.full(n, 1.0 / n), covariancia, "iguais")


def inverso_volatilidade(covariancia: pd.DataFrame) -> pd.Series:
    """Peso proporcional a 1/σ de cada ativo, ignorando correlação.

    Versão ingênua da paridade de risco: iguala a contribuição de risco
    apenas se todos os ativos tiverem a mesma correlação entre si. Onde há
    um par muito correlacionado — BBAS3 e ITUB4, por exemplo — ela carrega
    peso demais nesse par, porque não vê que os dois trazem o mesmo risco.
    """
    volatilidades = np.sqrt(np.diag(_validar(covariancia)))
    if np.any(volatilidades <= 0):
        raise ValueError("Ativo com volatilidade nula.")

    inverso = 1.0 / volatilidades
    return _serie(inverso / inverso.sum(), covariancia, "inverso_vol")


def paridade_risco(
    covariancia: pd.DataFrame, max_iteracoes: int = MAX_ITERACOES
) -> pd.Series:
    """Pesos tais que cada ativo contribui a MESMA parcela do risco total.

    Diferente do inverso da volatilidade porque leva a correlação em conta:
    dois ativos muito correlacionados são tratados quase como um só, e o
    peso conjunto deles cai.

    Resolve por ponto fixo com amortecimento. A cada passo, o peso de quem
    contribui risco demais encolhe e o de quem contribui de menos cresce; o
    expoente 0,5 amortece o passo e evita oscilação em torno da solução.
    """
    matriz = _validar(covariancia)
    n = len(matriz)
    alvo = 1.0 / n

    w = np.full(n, 1.0 / n)
    melhor_w, melhor_erro = w, np.inf

    for _ in range(max_iteracoes):
        sigma_w = matriz @ w
        variancia = float(w @ sigma_w)
        if variancia <= 0:
            raise ValueError("Variância nula: paridade de risco indefinida.")

        fracao = w * sigma_w / variancia
        erro = float(np.max(np.abs(fracao - alvo)))
        if erro < melhor_erro:
            melhor_w, melhor_erro = w, erro
        if erro < TOLERANCIA:
            break

        proximo = w * np.sqrt(alvo / np.maximum(fracao, 1e-300))
        proximo = np.maximum(proximo, 1e-300)
        proximo = proximo / proximo.sum()

        # Estagnação: os pesos pararam de mudar dentro da precisão da
        # máquina. Continuar iterando não melhora nada — e foi exatamente
        # isso que fazia a versão anterior rodar 10.000 vezes e então falhar
        # com uma solução boa na mão, só porque o erro tinha parado num
        # patamar pouco acima de uma tolerância escolhida no chute.
        if np.max(np.abs(proximo - w)) < 1e-15:
            w = proximo
            break
        w = proximo

    # O critério de aceitação é a QUALIDADE da resposta, não o número de
    # iterações. 1e-6 de desvio na fração de risco é irrelevante para
    # qualquer decisão de carteira; recusar isso seria rigor de fachada.
    if melhor_erro > TOLERANCIA_ACEITAVEL:
        raise NaoConvergiu(
            f"Paridade de risco parou com desvio de {melhor_erro:.2e} na "
            f"fração de risco, acima do aceitável de "
            f"{TOLERANCIA_ACEITAVEL:.0e}.\n"
            "Isso costuma indicar covariância quase singular — confira se "
            "algum ativo tem série de preço repetida ou com buraco."
        )

    return _serie(melhor_w / melhor_w.sum(), covariancia, "paridade_risco")


def minima_variancia(
    covariancia: pd.DataFrame, max_iteracoes: int = MAX_ITERACOES
) -> pd.Series:
    """Carteira de menor variância possível, sem posição vendida.

    Sem a restrição existe solução fechada, mas ela costuma pedir posição
    vendida pesada — e vendido não é o caso de uso aqui. Com a restrição é
    preciso otimizar, e a rota escolhida é gradiente projetado: passo na
    direção de -∇(wᵀΣw) seguido de projeção no simplex.

    O passo é 1/(2·λmax), o maior valor que garante decréscimo monótono
    para esta função. Convergência é lenta perto do ótimo, mas o problema
    tem 7 variáveis — lentidão aqui são milissegundos.

    ATENÇÃO ao usar: mínima variância tende a concentrar no ativo menos
    volátil, o que é o oposto de diversificar. Ela entra na comparação como
    extremo do espectro, não como recomendação.
    """
    matriz = _validar(covariancia)
    n = len(matriz)

    maior_autovalor = float(np.linalg.eigvalsh(matriz).max())
    if maior_autovalor <= 0:
        raise ValueError("Matriz de covariância sem autovalor positivo.")
    passo = 1.0 / (2.0 * maior_autovalor)

    w = np.full(n, 1.0 / n)
    for _ in range(max_iteracoes):
        anterior = w
        w = projetar_no_simplex(w - passo * 2.0 * (matriz @ w))
        if np.max(np.abs(w - anterior)) < TOLERANCIA:
            break
    else:
        raise NaoConvergiu(
            f"Mínima variância não convergiu em {max_iteracoes} iterações."
        )

    return _serie(w, covariancia, "minima_variancia")


def minima_variancia_sem_restricao(covariancia: pd.DataFrame) -> pd.Series:
    """Solução fechada Σ⁻¹1 / 1ᵀΣ⁻¹1, que PODE pedir posição vendida.

    Existe para comparação e para teste: quando o resultado dela já é todo
    não negativo, ele coincide com a versão restrita, o que valida o
    gradiente projetado contra uma resposta analítica.

    Note que ela inverte a matriz — é exatamente o passo que amplifica o
    autovalor pior estimado, e a razão de o encolhimento existir.
    """
    matriz = _validar(covariancia)
    uns = np.ones(len(matriz))

    try:
        inversa_uns = np.linalg.solve(matriz, uns)
    except np.linalg.LinAlgError as erro:
        raise ValueError("Matriz de covariância singular.") from erro

    return _serie(inversa_uns / inversa_uns.sum(), covariancia, "minvar_livre")


REGRAS = {
    "iguais": pesos_iguais,
    "inverso_vol": inverso_volatilidade,
    "paridade_risco": paridade_risco,
    "minima_variancia": minima_variancia,
}


def todas(covariancia: pd.DataFrame) -> dict[str, pd.Series]:
    """Aplica todas as regras e devolve {nome: pesos}."""
    return {nome: regra(covariancia) for nome, regra in REGRAS.items()}
