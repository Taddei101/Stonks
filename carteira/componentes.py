"""Quantas apostas independentes uma carteira realmente tem.

Contar papéis não mede diversificação. Sete ações que se movem juntas são
uma aposta com sete nomes, e a conta de papéis não distingue esse caso de
sete apostas de verdade.

A decomposição em componentes principais da matriz de CORRELAÇÃO resolve
isso. Cada componente é uma combinação dos ativos que se move de forma
independente das demais, e o autovalor correspondente diz quanta da
variação total aquele movimento explica. Se o primeiro componente sozinho
explica 70%, existe um fator único dominando tudo — tipicamente "risco
Brasil" numa carteira local.

Por que correlação e não covariância: a covariância é dominada pelo ativo
mais volátil, e aí os componentes medem volatilidade em vez de estrutura
comum. A correlação padroniza todo mundo antes, que é o que se quer para
responder "quantas apostas", e não "quem é mais volátil".

LIMITE DO MÉTODO, que importa aqui: componentes principais só enxergam o
que aconteceu na janela analisada. Um fator que ficou quieto no período —
risco político que não se materializou, choque regulatório que não veio —
não aparece, por mais exposta que a carteira esteja a ele. Por isso esta
análise anda junto com o agrupamento explícito de `carteira.blocos`, que
não depende de o risco ter se manifestado.
"""

import numpy as np
import pandas as pd


def _autodecomposicao(correlacao: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Autovalores e autovetores em ordem decrescente de autovalor.

    `eigh` em vez de `eig`: a matriz é simétrica, e o algoritmo simétrico
    garante autovalores reais. Com `eig` apareceria parte imaginária
    residual de arredondamento, que depois contamina qualquer conta.
    """
    autovalores, autovetores = np.linalg.eigh(correlacao.to_numpy())
    ordem = np.argsort(autovalores)[::-1]
    return autovalores[ordem], autovetores[:, ordem]


def decomposicao(correlacao: pd.DataFrame) -> pd.DataFrame:
    """Tabela por componente: autovalor, variância explicada e acumulada.

    A soma dos autovalores de uma matriz de correlação é sempre N, o número
    de ativos, então a fração explicada por componente é autovalor/N.
    """
    autovalores, _ = _autodecomposicao(correlacao)
    n = len(autovalores)

    explicada = autovalores / autovalores.sum()
    return pd.DataFrame(
        {
            "autovalor": autovalores,
            "explicada": explicada,
            "acumulada": np.cumsum(explicada),
        },
        index=pd.Index([f"CP{i + 1}" for i in range(n)], name="componente"),
    )


def cargas(correlacao: pd.DataFrame, quantos: int = 3) -> pd.DataFrame:
    """Peso de cada ativo nos primeiros componentes.

    Lê-se assim: no CP1, carga de mesmo sinal e magnitude parecida para
    todos os ativos é o fator de mercado — tudo subindo e descendo junto. No
    CP2, cargas de sinais opostos separam grupos, e é aí que se descobre o
    eixo que de fato divide a carteira (exportadora contra doméstica, por
    exemplo).

    O sinal de um autovetor é arbitrário. Para a leitura não mudar a cada
    execução, cada componente é orientado de modo que a maior carga em
    módulo seja positiva.
    """
    _, autovetores = _autodecomposicao(correlacao)
    quantos = min(quantos, autovetores.shape[1])

    vetores = autovetores[:, :quantos].copy()
    for j in range(quantos):
        dominante = np.argmax(np.abs(vetores[:, j]))
        if vetores[dominante, j] < 0:
            vetores[:, j] *= -1

    return pd.DataFrame(
        vetores,
        index=correlacao.index,
        columns=[f"CP{i + 1}" for i in range(quantos)],
    )


def apostas_efetivas(correlacao: pd.DataFrame) -> float:
    """Número efetivo de apostas independentes.

    Exponencial da entropia dos autovalores normalizados. A interpretação é
    direta e os dois extremos fecham:

      ativos independentes      todos os autovalores iguais a 1
                                -> resultado igual a N
      ativos idênticos          um autovalor igual a N, resto zero
                                -> resultado igual a 1

    Entre os extremos o número é contínuo e tem a unidade certa: "esta
    carteira de 7 papéis se comporta como 2,8 apostas".
    """
    autovalores, _ = _autodecomposicao(correlacao)
    p = autovalores / autovalores.sum()
    p = p[p > 1e-15]  # 0*ln(0) = 0, mas numpy devolveria nan

    entropia = -np.sum(p * np.log(p))
    return float(np.exp(entropia))


def componentes_para_explicar(
    correlacao: pd.DataFrame, alvo: float = 0.90
) -> int:
    """Quantos componentes somam pelo menos `alvo` da variância total."""
    if not 0 < alvo <= 1:
        raise ValueError(f"alvo deve estar em (0, 1], veio {alvo!r}")

    acumulada = decomposicao(correlacao)["acumulada"].to_numpy()
    return int(np.searchsorted(acumulada, alvo - 1e-12) + 1)


def pares_mais_correlacionados(
    correlacao: pd.DataFrame, limite: float = 0.7
) -> pd.DataFrame:
    """Pares acima do limite de correlação, do maior para o menor.

    Complementa a visão por componentes: o componente diz que existe um
    fator comum, e o par diz quais papéis especificamente o compartilham.
    """
    matriz = correlacao.to_numpy()
    linhas = []
    for i in range(len(correlacao)):
        for j in range(i + 1, len(correlacao)):
            if abs(matriz[i, j]) >= limite:
                linhas.append(
                    {
                        "a": correlacao.index[i],
                        "b": correlacao.columns[j],
                        "correlacao": matriz[i, j],
                    }
                )

    return (
        pd.DataFrame(linhas, columns=["a", "b", "correlacao"])
        .sort_values("correlacao", ascending=False, key=abs)
        .reset_index(drop=True)
    )
