"""Testes das matrizes de risco.

O encolhimento é testado por PROPRIEDADE, não contra números fixos: as
fórmulas de Ledoit-Wolf não têm forma fechada simples para conferir à mão, e
um teste contra valores gravados só detectaria mudança, não erro.

As propriedades abaixo são consequências diretas da teoria, e cada uma falha
de um jeito diferente se a implementação estiver errada.
"""

import numpy as np
import pandas as pd
import pytest

from carteira.risco import (
    AmostraInsuficiente,
    anualizar_covariancia,
    covariancia_amostral,
    e_positiva_definida,
    encolhimento_ledoit_wolf,
    matriz_correlacao,
    matriz_covariancia,
    volatilidade_carteira,
)

ATIVOS = list("ABCDE")


def _retornos(T, N=5, semente=7, correlacionar=True):
    """Retornos sintéticos com correlação embutida entre os dois primeiros."""
    rng = np.random.default_rng(semente)
    x = rng.normal(0.0, 0.05, (T, N))
    if correlacionar and N >= 2:
        x[:, 1] = 0.7 * x[:, 0] + 0.3 * x[:, 1]
    return pd.DataFrame(x, columns=ATIVOS[:N])


# --------------------------------------------------------------------------
# básico
# --------------------------------------------------------------------------


def test_correlacao_tem_diagonal_unitaria():
    corr = matriz_correlacao(_retornos(60))

    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
    assert corr.loc["A", "B"] > 0.5


def test_amostra_pequena_demais_falha():
    with pytest.raises(AmostraInsuficiente):
        covariancia_amostral(_retornos(2))


def test_metodo_invalido():
    with pytest.raises(ValueError, match="ledoit_wolf"):
        matriz_covariancia(_retornos(60), metodo="chute")


# --------------------------------------------------------------------------
# propriedades do encolhimento
# --------------------------------------------------------------------------


def test_intensidade_esta_entre_zero_e_um():
    for T in (6, 12, 36, 120):
        _, delta = encolhimento_ledoit_wolf(_retornos(T))
        assert 0.0 <= delta <= 1.0


def test_intensidade_cai_conforme_a_janela_cresce():
    """Com muitos dados a amostra passa a valer por si. É o teste que pega
    erro de sinal ou de divisor nas fórmulas de pi, rho e gamma."""
    curta = encolhimento_ledoit_wolf(_retornos(24))[1]
    media = encolhimento_ledoit_wolf(_retornos(120))[1]
    longa = encolhimento_ledoit_wolf(_retornos(1200))[1]

    assert curta > media > longa
    assert longa < 0.1


def test_variancias_individuais_sao_preservadas():
    """O alvo de correlação constante mexe só na correlação.

    Como alvo e amostra têm a mesma diagonal, a combinação também tem.
    """
    retornos = _retornos(60)
    encolhida, _ = encolhimento_ledoit_wolf(retornos)
    amostral = covariancia_amostral(retornos)

    # amostral usa ddof=1 e o encolhimento usa ddof=0; a razão é constante.
    razao = np.diag(encolhida.to_numpy()) / np.diag(amostral.to_numpy())
    assert np.allclose(razao, razao[0])


def test_encolhimento_puxa_a_correlacao_para_a_media():
    """O par mais correlacionado tem de ficar menos correlacionado."""
    retornos = _retornos(24)
    encolhida, delta = encolhimento_ledoit_wolf(retornos)

    assert delta > 0

    def correlacao(m):
        dp = np.sqrt(np.diag(m.to_numpy()))
        return (m.to_numpy() / np.outer(dp, dp))[0, 1]

    assert correlacao(encolhida) < correlacao(covariancia_amostral(retornos))


def test_encolhida_e_positiva_definida_com_mais_ativos_que_observacoes():
    """A razão de ser do módulo.

    Com T < N a covariância amostral é singular — tem autovalor zero, e
    qualquer inversão explode. A encolhida continua utilizável.
    """
    poucos_dados = _retornos(4, N=5)

    assert not e_positiva_definida(covariancia_amostral(poucos_dados))
    assert e_positiva_definida(encolhimento_ledoit_wolf(poucos_dados)[0])


def test_dois_ativos_nao_tem_o_que_encolher():
    """Com N=2 só existe uma correlação, então a média dela é ela mesma:
    o alvo coincide com a amostra e a intensidade perde sentido."""
    encolhida, delta = encolhimento_ledoit_wolf(_retornos(60, N=2))

    assert delta == pytest.approx(0.0)
    assert e_positiva_definida(encolhida)


def test_um_ativo_so_falha():
    with pytest.raises(AmostraInsuficiente, match="dois ativos"):
        encolhimento_ledoit_wolf(_retornos(60, N=1, correlacionar=False))


def test_ativo_sem_variacao_falha_nomeando_o_ativo():
    retornos = _retornos(60)
    retornos["C"] = 0.01

    with pytest.raises(AmostraInsuficiente, match="C"):
        encolhimento_ledoit_wolf(retornos)


def test_matriz_e_simetrica():
    encolhida, _ = encolhimento_ledoit_wolf(_retornos(36))

    assert np.allclose(encolhida.to_numpy(), encolhida.to_numpy().T)


# --------------------------------------------------------------------------
# anualização e volatilidade
# --------------------------------------------------------------------------


def test_covariancia_escala_linearmente_no_tempo():
    """Covariância escala com o tempo; desvio padrão, com a raiz do tempo."""
    mensal = covariancia_amostral(_retornos(60))
    anual = anualizar_covariancia(mensal)

    assert anual.iloc[0, 0] == pytest.approx(mensal.iloc[0, 0] * 12)
    assert np.sqrt(anual.iloc[0, 0]) == pytest.approx(
        np.sqrt(mensal.iloc[0, 0]) * np.sqrt(12)
    )


def test_volatilidade_de_ativo_unico_e_o_desvio_padrao_dele():
    cov = covariancia_amostral(_retornos(60))
    so_a = pd.Series({a: (1.0 if a == "A" else 0.0) for a in ATIVOS})

    assert volatilidade_carteira(cov, so_a) == pytest.approx(
        np.sqrt(cov.loc["A", "A"])
    )


def test_diversificar_reduz_a_volatilidade():
    """Com ativos não perfeitamente correlacionados, a volatilidade da
    carteira fica abaixo da média ponderada das individuais."""
    cov = covariancia_amostral(_retornos(120))
    iguais = pd.Series(1.0 / len(ATIVOS), index=ATIVOS)

    media_das_individuais = float(np.sqrt(np.diag(cov.to_numpy())).mean())

    assert volatilidade_carteira(cov, iguais) < media_das_individuais


def test_volatilidade_ignora_escala_dos_pesos():
    cov = covariancia_amostral(_retornos(60))
    unitarios = pd.Series(1.0 / len(ATIVOS), index=ATIVOS)
    centenas = pd.Series(100.0, index=ATIVOS)

    assert volatilidade_carteira(cov, unitarios) == pytest.approx(
        volatilidade_carteira(cov, centenas)
    )


def test_peso_faltando_falha_nomeando_o_ativo():
    cov = covariancia_amostral(_retornos(60))

    with pytest.raises(KeyError, match="E"):
        volatilidade_carteira(cov, pd.Series({a: 0.25 for a in "ABCD"}))
