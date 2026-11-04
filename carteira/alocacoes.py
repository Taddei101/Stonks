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

    COMO RESOLVE, E POR QUE NÃO POR PONTO FIXO

    A primeira versão usava ponto fixo amortecido: encolher o peso de quem
    contribui risco demais, aumentar o de quem contribui de menos. Simples,
    e funciona na maioria dos casos — mas quebra quando algum ativo tem
    correlação suficientemente NEGATIVA com o resto. A partir de −0,3 a
    iteração passa a divergir, porque empurra o peso do ativo protetor para
    cima sem nunca conseguir fazê-lo carregar sua fatia de risco.

    Isso não significa que a solução não exista. A paridade de risco é o
    mínimo de ½·wᵀΣw − (1/N)·Σ ln(wᵢ), que é estritamente convexo no ortante
    positivo: existe e é única para qualquer Σ positiva definida. A barreira
    logarítmica impede peso zero, e a condição de primeira ordem é
    exatamente wᵢ·(Σw)ᵢ constante, que é a definição da paridade.

    Resolve-se por descida coordenada cíclica. Fixados os demais pesos, a
    condição de otimalidade em wᵢ vira uma equação do segundo grau:

        Σᵢᵢ·wᵢ² + bᵢ·wᵢ − 1/N = 0,   com bᵢ = (Σw)ᵢ − Σᵢᵢ·wᵢ

    cuja raiz positiva tem forma fechada. Sem passo, sem amortecimento, sem
    ajuste — e converge com correlação negativa, que é o caso do ativo de
    proteção numa carteira local.
    """
    matriz = _validar(covariancia)
    n = len(matriz)
    diagonal = np.diag(matriz)
    if np.any(diagonal <= 0):
        zerados = [covariancia.index[i] for i in np.where(diagonal <= 0)[0]]
        raise ValueError(f"Variância nula ou negativa em: {zerados}")

    # Começa fora do simplex de propósito: a formulação com barreira
    # trabalha com pesos não normalizados, e a normalização é o último passo.
    w = np.full(n, 1.0 / np.sqrt(n))

    for _ in range(max_iteracoes):
        anterior = w.copy()
        for i in range(n):
            b = float(matriz[i] @ w - diagonal[i] * w[i])
            w[i] = (-b + np.sqrt(b * b + 4.0 * diagonal[i] / n)) / (
                2.0 * diagonal[i]
            )
        if np.max(np.abs(w - anterior)) < TOLERANCIA:
            break

    w = w / w.sum()

    sigma_w = matriz @ w
    fracao = w * sigma_w / float(w @ sigma_w)
    desvio = float(np.max(np.abs(fracao - 1.0 / n)))
    if desvio > TOLERANCIA_ACEITAVEL:
        raise NaoConvergiu(
            f"Paridade de risco parou com desvio de {desvio:.2e} na fração "
            f"de risco, acima do aceitável de {TOLERANCIA_ACEITAVEL:.0e}.\n"
            "Confira se a covariância é positiva definida."
        )

    return _serie(w, covariancia, "paridade_risco")


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


# ==========================================================================
# Regras que USAM retorno esperado
# ==========================================================================
#
# As de cima deliberadamente ignoram retorno. Estas não — e por isso herdam
# o problema que as outras evitam: a estimativa de retorno médio é ruidosa
# demais, e o otimizador carrega peso justamente no ativo que PARECEU render
# mais, o que na maior parte das vezes é ruído.
#
# Existem aqui para serem TESTADAS, não para serem recomendadas. A pergunta
# "incluir retorno ajuda?" só tem resposta fora da amostra, e é para isso
# que serve `carteira.validacao`.


def _max_sharpe(
    retornos_esperados: np.ndarray,
    matriz: np.ndarray,
    max_iteracoes: int = 2000,
) -> np.ndarray:
    """Máximo índice de Sharpe no simplex, por gradiente projetado.

    S(w) = wᵀμ / √(wᵀΣw), e o gradiente é μ/σ − (wᵀμ)(Σw)/σ³.

    Sharpe é invariante a escala, então o passo é normalizado pela norma do
    gradiente — sem isso o tamanho do passo dependeria das unidades dos
    retornos, e o mesmo código se comportaria diferente em base mensal e
    anual.

    Vários pontos de partida porque a função não é côncava no simplex. Com
    7 ativos o custo é irrelevante e remove a dependência do ponto inicial.
    """
    n = len(retornos_esperados)
    if np.all(retornos_esperados <= 0):
        # Sem nenhum retorno positivo, maximizar Sharpe não faz sentido:
        # qualquer carteira tem numerador negativo e "melhor" vira "menos
        # pior", o que é outra pergunta. Devolve mínima variância.
        return None

    partidas = [np.full(n, 1.0 / n)]
    for i in range(n):
        canto = np.full(n, 0.01 / (n - 1))
        canto[i] = 0.99
        partidas.append(canto / canto.sum())

    melhor_w, melhor_s = None, -np.inf
    for inicio in partidas:
        w = inicio.copy()
        for passo in range(max_iteracoes):
            sigma = float(np.sqrt(max(w @ matriz @ w, 1e-300)))
            retorno = float(w @ retornos_esperados)
            gradiente = (
                retornos_esperados / sigma - retorno * (matriz @ w) / sigma**3
            )
            norma = np.linalg.norm(gradiente)
            if norma < 1e-14:
                break
            # Expoente 0,85 e não 0,5: calibrado contra busca exaustiva numa
            # grade de 1201x1201. Com 0,5 o passo final ainda é de 1e-2 e a
            # solução para a 6e-3 do ótimo; com 0,85 chega a 5e-4, que é
            # abaixo do ruído de estimativa de qualquer entrada real.
            tamanho = 0.5 / (passo + 1) ** 0.85
            novo = projetar_no_simplex(w + tamanho * gradiente / norma)
            if np.max(np.abs(novo - w)) < 1e-12:
                w = novo
                break
            w = novo

        sigma = float(np.sqrt(max(w @ matriz @ w, 1e-300)))
        s = float(w @ retornos_esperados) / sigma
        if s > melhor_s:
            melhor_w, melhor_s = w, s

    return melhor_w


def markowitz(retornos: pd.DataFrame, encolher_medias: bool = False) -> pd.Series:
    """Máximo Sharpe com a média histórica como retorno esperado.

    É o método que a maioria das pessoas chama de "Markowitz", e o que
    produz pesos extremos e instáveis entre janelas. Está aqui para ser
    medido contra as alternativas, não como recomendação.

    A covariância usa encolhimento mesmo aqui — sem isso a comparação
    confundiria "usar retorno" com "usar covariância ruim", e a pergunta de
    interesse é só a primeira.
    """
    from carteira.risco import matriz_covariancia

    covariancia = matriz_covariancia(retornos)
    medias = retornos.mean().to_numpy()
    if encolher_medias:
        medias = bayes_stein(retornos, covariancia)

    pesos = _max_sharpe(medias, covariancia.to_numpy())
    if pesos is None:
        return minima_variancia(covariancia).rename("markowitz")

    nome = "markowitz_encolhido" if encolher_medias else "markowitz"
    return _serie(pesos, covariancia, nome)


def markowitz_encolhido(retornos: pd.DataFrame) -> pd.Series:
    """Markowitz com as médias puxadas por Bayes-Stein. Ver `bayes_stein`."""
    return markowitz(retornos, encolher_medias=True)


def bayes_stein(retornos: pd.DataFrame, covariancia: pd.DataFrame) -> np.ndarray:
    """Encolhe as médias amostrais na direção da carteira de mínima variância.

    Estimador de Jorion (1986). A lógica é a mesma do Ledoit-Wolf, aplicada
    ao primeiro momento: a média amostral de cada ativo é muito ruidosa, e
    puxá-las todas em direção a um valor comum reduz o erro agregado mesmo
    introduzindo viés em cada uma.

    A intensidade sai dos dados:

        w = (N + 2) / ((N + 2) + T · (μ − μ₀·1)ᵀ Σ⁻¹ (μ − μ₀·1))

    Encolhe mais quando há poucos dados (T pequeno) ou quando as médias
    estão próximas entre si — caso em que a diferença observada é
    provavelmente ruído.

    O alvo μ₀ é o retorno da carteira de mínima variância, e não a média
    geral simples, porque é o ponto do conjunto eficiente que independe
    completamente das estimativas de retorno.
    """
    T, N = retornos.shape
    medias = retornos.mean().to_numpy()
    matriz = covariancia.to_numpy()

    try:
        inversa_uns = np.linalg.solve(matriz, np.ones(N))
    except np.linalg.LinAlgError:
        return medias

    w_minvar = inversa_uns / inversa_uns.sum()
    mu_zero = float(w_minvar @ medias)

    desvio = medias - mu_zero
    try:
        quadratico = float(desvio @ np.linalg.solve(matriz, desvio))
    except np.linalg.LinAlgError:
        return medias

    if quadratico <= 0:
        return np.full(N, mu_zero)

    intensidade = (N + 2) / ((N + 2) + T * quadratico)
    intensidade = float(np.clip(intensidade, 0.0, 1.0))

    return intensidade * mu_zero + (1.0 - intensidade) * medias


def inclinacao_log(precos_ou_retornos: pd.DataFrame, de_retornos: bool = True):
    """Coeficiente angular da regressão do log do preço contra o tempo.

    Por que o LOG e não o preço: regredir o preço em reais faz o coeficiente
    depender do nível de preço, e aí uma ação de R$ 40 e outra de R$ 8 com a
    mesma valorização percentual dão números diferentes — a comparação entre
    papéis fica inválida.

    Com log, o coeficiente é aproximadamente o retorno médio por período.
    Essa equivalência é o ponto importante: usar a inclinação como sinal de
    retorno é usar a média histórica, que é exatamente a estimativa cujo
    intervalo de confiança é largo demais para decidir.
    """
    if de_retornos:
        log_preco = np.log((1 + precos_ou_retornos).cumprod())
    else:
        log_preco = np.log(precos_ou_retornos)

    t = np.arange(len(log_preco), dtype=float)
    t = t - t.mean()
    denominador = float(t @ t)
    if denominador <= 0:
        return pd.Series(0.0, index=log_preco.columns)

    centrado = log_preco - log_preco.mean()
    return (centrado.mul(t, axis=0).sum() / denominador).rename("inclinacao")


def tendencia(retornos: pd.DataFrame) -> pd.Series:
    """Peso proporcional à inclinação positiva do log-preço.

    Formalização da regra intuitiva "se a tendência é de alta, ela vai bem".
    Ativos com inclinação negativa ficam zerados; o resto divide o capital
    na proporção da inclinação.

    Quando nenhum ativo tem tendência positiva, cai para pesos iguais — a
    alternativa seria ficar sem carteira, que não é uma resposta.
    """
    inclinacoes = inclinacao_log(retornos)
    positivas = inclinacoes.clip(lower=0.0)

    total = float(positivas.sum())
    if total <= 0:
        return pd.Series(
            1.0 / len(inclinacoes), index=inclinacoes.index, name="tendencia"
        )

    return (positivas / total).rename("tendencia")
