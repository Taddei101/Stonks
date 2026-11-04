"""Testes da decomposição de risco por ativo e por bloco.

A propriedade central é a decomposição de Euler: as contribuições somam
exatamente a volatilidade da carteira. Não é aproximação, e se a
implementação errar a fórmula a soma deixa de fechar — é o teste que pega
praticamente qualquer erro aqui.
"""

import numpy as np
import pandas as pd
import pytest

from carteira.blocos import BlocosInvalidos
from carteira.risco import (
    contribuicao_marginal,
    contribuicao_por_bloco,
    contribuicao_risco,
    tabela_contribuicao,
    volatilidade_carteira,
)

NOMES = ["CALMO", "MEDIO", "AGITADO"]


@pytest.fixture
def covariancia():
    """Volatilidades 10%, 20% e 40%, com correlação 0,5 entre todos."""
    dp = np.array([0.10, 0.20, 0.40])
    corr = np.full((3, 3), 0.5)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr * np.outer(dp, dp), index=NOMES, columns=NOMES)


@pytest.fixture
def iguais():
    return pd.Series(1 / 3, index=NOMES)


# --------------------------------------------------------------------------
# Euler: a soma tem de fechar
# --------------------------------------------------------------------------


def test_contribuicoes_somam_a_volatilidade_da_carteira(covariancia, iguais):
    soma = float(contribuicao_risco(covariancia, iguais).sum())

    assert soma == pytest.approx(volatilidade_carteira(covariancia, iguais))


@pytest.mark.parametrize(
    "pesos",
    [
        {"CALMO": 0.8, "MEDIO": 0.1, "AGITADO": 0.1},
        {"CALMO": 0.1, "MEDIO": 0.1, "AGITADO": 0.8},
        {"CALMO": 0.0, "MEDIO": 0.5, "AGITADO": 0.5},
    ],
)
def test_euler_fecha_para_qualquer_peso(covariancia, pesos):
    serie = pd.Series(pesos)
    soma = float(contribuicao_risco(covariancia, serie).sum())

    assert soma == pytest.approx(volatilidade_carteira(covariancia, serie))


def test_peso_risco_soma_um(covariancia, iguais):
    assert tabela_contribuicao(covariancia, iguais)["peso_risco"].sum() == (
        pytest.approx(1.0)
    )


# --------------------------------------------------------------------------
# a distinção que motiva o módulo
# --------------------------------------------------------------------------


def test_capital_igual_nao_significa_risco_igual(covariancia, iguais):
    """Um terço do dinheiro em cada, mas o volátil domina o risco."""
    tabela = tabela_contribuicao(covariancia, iguais)

    assert tabela.index[0] == "AGITADO"
    assert tabela.loc["AGITADO", "peso_risco"] > 0.55
    assert tabela.loc["CALMO", "peso_risco"] < 0.15


def test_razao_indica_quem_pesa_mais_no_risco(covariancia, iguais):
    """Razão acima de 1: o ativo ocupa mais risco do que capital."""
    tabela = tabela_contribuicao(covariancia, iguais)

    assert tabela.loc["AGITADO", "razao"] > 1
    assert tabela.loc["CALMO", "razao"] < 1


def test_ativo_zerado_nao_contribui(covariancia):
    pesos = pd.Series({"CALMO": 0.5, "MEDIO": 0.5, "AGITADO": 0.0})

    assert contribuicao_risco(covariancia, pesos)["AGITADO"] == pytest.approx(0.0)


def test_carteira_de_um_ativo_concentra_todo_o_risco(covariancia):
    pesos = pd.Series({"CALMO": 1.0, "MEDIO": 0.0, "AGITADO": 0.0})
    tabela = tabela_contribuicao(covariancia, pesos)

    assert tabela.loc["CALMO", "peso_risco"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# contribuição marginal
# --------------------------------------------------------------------------


def test_marginal_confere_com_a_derivada_numerica(covariancia, iguais):
    """A definição é a derivada de sigma em relação ao peso. Confere com
    diferença finita, mantendo a soma dos pesos livre (sem renormalizar)."""
    analitica = contribuicao_marginal(covariancia, iguais)

    h = 1e-7
    w = iguais.to_numpy()
    m = covariancia.to_numpy()

    def sigma(v):
        return float(np.sqrt(v @ m @ v))

    for i, nome in enumerate(NOMES):
        frente, tras = w.copy(), w.copy()
        frente[i] += h
        tras[i] -= h
        numerica = (sigma(frente) - sigma(tras)) / (2 * h)
        assert analitica[nome] == pytest.approx(numerica, rel=1e-5)


def test_ativo_descorrelacionado_tem_marginal_menor():
    """Volatilidade alta não implica contribuição marginal alta.

    Aqui os dois ativos têm a MESMA volatilidade; o que difere é a
    correlação com o resto. É o mecanismo da diversificação.
    """
    nomes = ["BASE", "JUNTO", "SOZINHO"]
    dp = np.array([0.20, 0.20, 0.20])
    corr = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cov = pd.DataFrame(corr * np.outer(dp, dp), index=nomes, columns=nomes)

    marginal = contribuicao_marginal(cov, pd.Series(1 / 3, index=nomes))

    assert marginal["SOZINHO"] < marginal["JUNTO"]


def test_volatilidade_nula_falha():
    cov = pd.DataFrame(np.zeros((2, 2)), index=["A", "B"], columns=["A", "B"])

    with pytest.raises(ValueError, match="Volatilidade nula"):
        contribuicao_marginal(cov, pd.Series({"A": 0.5, "B": 0.5}))


# --------------------------------------------------------------------------
# por bloco
# --------------------------------------------------------------------------


def test_agregacao_por_bloco_preserva_o_total(covariancia, iguais):
    mapa = pd.Series({"CALMO": "defensivo", "MEDIO": "defensivo", "AGITADO": "ciclico"})

    tabela = contribuicao_por_bloco(covariancia, iguais, mapa)

    assert tabela["peso_capital"].sum() == pytest.approx(1.0)
    assert tabela["peso_risco"].sum() == pytest.approx(1.0)


def test_bloco_com_o_ativo_volatil_domina_o_risco(covariancia, iguais):
    mapa = pd.Series({"CALMO": "defensivo", "MEDIO": "defensivo", "AGITADO": "ciclico"})

    tabela = contribuicao_por_bloco(covariancia, iguais, mapa)

    assert tabela.loc["ciclico", "peso_capital"] == pytest.approx(1 / 3)
    assert tabela.loc["ciclico", "peso_risco"] > 0.5
    assert tabela.loc["ciclico", "razao"] > 1.5


def test_bloco_sem_mapeamento_falha(covariancia, iguais):
    mapa = pd.Series({"CALMO": "defensivo"})

    with pytest.raises(BlocosInvalidos, match="AGITADO"):
        contribuicao_por_bloco(covariancia, iguais, mapa)


def test_bloco_unico_concentra_tudo(covariancia, iguais):
    mapa = pd.Series({nome: "tudo" for nome in NOMES})

    tabela = contribuicao_por_bloco(covariancia, iguais, mapa)

    assert tabela.loc["tudo", "peso_risco"] == pytest.approx(1.0)
    assert tabela.loc["tudo", "razao"] == pytest.approx(1.0)
