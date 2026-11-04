"""Testes das regras de alocação.

Cada otimizador é verificado contra uma propriedade que tem resposta
conhecida, não contra números gravados. Para a mínima variância existe uma
solução fechada quando a restrição de não negatividade não está ativa — é
a melhor validação possível de um método iterativo.
"""

import numpy as np
import pandas as pd
import pytest

from carteira.alocacoes import (
    inverso_volatilidade,
    minima_variancia,
    minima_variancia_sem_restricao,
    paridade_risco,
    pesos_iguais,
    projetar_no_simplex,
    todas,
)
from carteira.risco import contribuicao_risco, volatilidade_carteira

NOMES = ["CALMO", "MEDIO", "AGITADO"]


def _cov(dp, correlacao, nomes=NOMES):
    dp = np.asarray(dp, dtype=float)
    corr = np.full((len(dp), len(dp)), correlacao)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr * np.outer(dp, dp), index=nomes, columns=nomes)


@pytest.fixture
def covariancia():
    return _cov([0.10, 0.20, 0.40], 0.3)


# --------------------------------------------------------------------------
# projeção no simplex
# --------------------------------------------------------------------------


def test_projecao_soma_um_e_nao_tem_negativo():
    for v in ([0.5, 0.9, -0.3], [10.0, -5.0, 2.0], [-1.0, -1.0, -1.0]):
        p = projetar_no_simplex(np.array(v))
        assert p.sum() == pytest.approx(1.0)
        assert (p >= 0).all()


def test_projecao_preserva_ponto_ja_viavel():
    ja_viavel = np.array([0.2, 0.3, 0.5])

    assert projetar_no_simplex(ja_viavel) == pytest.approx(ja_viavel)


def test_projecao_e_o_ponto_mais_proximo():
    """Confere contra busca por força bruta numa grade fina."""
    alvo = np.array([0.8, 0.5, -0.2])
    projetado = projetar_no_simplex(alvo)

    melhor = np.inf
    for a in np.linspace(0, 1, 201):
        for b in np.linspace(0, 1 - a, 201):
            candidato = np.array([a, b, 1 - a - b])
            melhor = min(melhor, np.sum((candidato - alvo) ** 2))

    assert np.sum((projetado - alvo) ** 2) <= melhor + 1e-6


# --------------------------------------------------------------------------
# regras simples
# --------------------------------------------------------------------------


def test_pesos_iguais(covariancia):
    pesos = pesos_iguais(covariancia)

    assert pesos.tolist() == pytest.approx([1 / 3] * 3)


def test_inverso_volatilidade_favorece_o_menos_volatil(covariancia):
    pesos = inverso_volatilidade(covariancia)

    assert pesos["CALMO"] > pesos["MEDIO"] > pesos["AGITADO"]
    # Volatilidades 10/20/40 -> pesos proporcionais a 1/1, 1/2, 1/4.
    assert pesos["CALMO"] / pesos["AGITADO"] == pytest.approx(4.0)


@pytest.mark.parametrize("regra", ["iguais", "inverso_vol", "paridade_risco",
                                   "minima_variancia"])
def test_toda_regra_devolve_pesos_validos(covariancia, regra):
    pesos = todas(covariancia)[regra]

    assert pesos.sum() == pytest.approx(1.0)
    assert (pesos >= -1e-12).all()
    assert list(pesos.index) == NOMES


# --------------------------------------------------------------------------
# paridade de risco
# --------------------------------------------------------------------------


def test_paridade_iguala_a_contribuicao_de_risco(covariancia):
    """A definição da regra. Se isso falhar, o ponto fixo está errado."""
    pesos = paridade_risco(covariancia)
    contribuicao = contribuicao_risco(covariancia, pesos)
    fracao = contribuicao / contribuicao.sum()

    assert fracao.to_numpy() == pytest.approx([1 / 3] * 3, abs=1e-6)


def test_paridade_com_ativos_identicos_da_pesos_iguais():
    identicos = _cov([0.2, 0.2, 0.2], 0.4)

    assert paridade_risco(identicos).to_numpy() == pytest.approx([1 / 3] * 3)


def test_paridade_difere_do_inverso_da_volatilidade_quando_ha_par_correlacionado():
    """O inverso da volatilidade não vê correlação; a paridade vê.

    Dois ativos de mesma volatilidade, um deles colado num terceiro: a
    paridade reduz o peso do par, o inverso da volatilidade não.
    """
    nomes = ["SOZINHO", "PAR_A", "PAR_B"]
    dp = np.array([0.2, 0.2, 0.2])
    corr = np.array([[1.0, 0.1, 0.1], [0.1, 1.0, 0.9], [0.1, 0.9, 1.0]])
    cov = pd.DataFrame(corr * np.outer(dp, dp), index=nomes, columns=nomes)

    paridade = paridade_risco(cov)
    inverso = inverso_volatilidade(cov)

    # Mesma volatilidade nos três: o inverso da volatilidade não distingue.
    assert inverso.to_numpy() == pytest.approx([1 / 3] * 3)

    # A paridade enxerga que o par traz risco repetido e reduz o conjunto.
    assert paridade["SOZINHO"] > 1 / 3
    assert paridade["PAR_A"] + paridade["PAR_B"] < 2 / 3
    assert paridade["PAR_A"] == pytest.approx(paridade["PAR_B"], abs=1e-6)


# --------------------------------------------------------------------------
# mínima variância
# --------------------------------------------------------------------------


def test_minima_variancia_bate_a_solucao_fechada_quando_nao_ha_restricao_ativa():
    """Validação forte do gradiente projetado contra resposta analítica.

    Com correlação baixa e volatilidades parecidas, a solução sem restrição
    já é toda positiva — então as duas têm de coincidir.
    """
    cov = _cov([0.18, 0.20, 0.22], 0.2)
    livre = minima_variancia_sem_restricao(cov)

    assert (livre > 0).all(), "cenário mal escolhido: a restrição está ativa"
    assert minima_variancia(cov).to_numpy() == pytest.approx(
        livre.to_numpy(), abs=1e-5
    )


def test_minima_variancia_tem_a_menor_volatilidade(covariancia):
    """Tem de bater todas as outras regras no quesito que ela otimiza."""
    minima = volatilidade_carteira(covariancia, minima_variancia(covariancia))

    for nome, pesos in todas(covariancia).items():
        assert minima <= volatilidade_carteira(covariancia, pesos) + 1e-9, nome


def test_minima_variancia_nao_bate_aleatorios(covariancia):
    rng = np.random.default_rng(5)
    minima = volatilidade_carteira(covariancia, minima_variancia(covariancia))

    for _ in range(200):
        bruto = rng.random(3)
        aleatorio = pd.Series(bruto / bruto.sum(), index=NOMES)
        assert minima <= volatilidade_carteira(covariancia, aleatorio) + 1e-9


def test_minima_variancia_concentra_no_menos_volatil(covariancia):
    """O comportamento que o docstring avisa: ela não diversifica."""
    pesos = minima_variancia(covariancia)

    assert pesos["CALMO"] > 0.6


def test_minima_variancia_respeita_a_restricao_quando_ela_e_ativa():
    """Com correlação alta a solução livre pede vendido; a restrita não pode."""
    cov = _cov([0.10, 0.30, 0.35], 0.85)

    livre = minima_variancia_sem_restricao(cov)
    restrita = minima_variancia(cov)

    assert (livre < 0).any(), "cenário mal escolhido: a restrição não está ativa"
    assert (restrita >= -1e-12).all()
    assert restrita.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# validação de entrada
# --------------------------------------------------------------------------


def test_matriz_assimetrica_falha():
    ruim = pd.DataFrame(
        [[0.04, 0.01], [0.02, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    with pytest.raises(ValueError, match="simétrica"):
        paridade_risco(ruim)


def test_volatilidade_nula_falha():
    cov = pd.DataFrame(np.zeros((2, 2)), index=["A", "B"], columns=["A", "B"])

    with pytest.raises(ValueError):
        inverso_volatilidade(cov)
