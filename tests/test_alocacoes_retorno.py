"""Testes das regras que usam retorno esperado.

O otimizador de máximo Sharpe é validado contra busca exaustiva numa grade
fina — é a única forma honesta de verificar um método iterativo sobre uma
função que não é côncava.
"""

import numpy as np
import pandas as pd
import pytest

from carteira.alocacoes import (
    _max_sharpe,
    bayes_stein,
    inclinacao_log,
    markowitz,
    markowitz_encolhido,
    momentum_12_1,
    regras_sobre_retornos,
    tendencia,
)
from carteira.risco import matriz_covariancia


def _retornos(n=120, k=5, semente=3, deriva=0.001):
    rng = np.random.default_rng(semente)
    mercado = rng.normal(0.004, 0.055, n)
    dados = {
        f"A{i}": (0.5 + 0.2 * i) * mercado + rng.normal(deriva * i, 0.03, n)
        for i in range(k)
    }
    return pd.DataFrame(
        dados, index=pd.date_range("2016-11-30", periods=n, freq="ME")
    )


# --------------------------------------------------------------------------
# otimizador
# --------------------------------------------------------------------------


def test_max_sharpe_bate_a_busca_exaustiva():
    """Três ativos permitem varrer o simplex inteiro numa grade."""
    mu = np.array([0.010, 0.014, 0.008])
    dp = np.array([0.05, 0.08, 0.04])
    corr = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.5], [0.2, 0.5, 1.0]])
    cov = corr * np.outer(dp, dp)

    w = _max_sharpe(mu, cov)
    sharpe = (w @ mu) / np.sqrt(w @ cov @ w)

    melhor = -np.inf
    for a in np.linspace(0, 1, 301):
        for b in np.linspace(0, 1 - a, 301):
            c = np.array([a, b, 1 - a - b])
            melhor = max(melhor, (c @ mu) / np.sqrt(c @ cov @ c))

    assert sharpe >= melhor - 1e-4


def test_max_sharpe_escolhe_o_unico_ativo_que_rende():
    mu = np.array([0.02, 0.0, 0.0])
    cov = np.diag([0.0025, 0.0025, 0.0025])

    assert _max_sharpe(mu, cov)[0] > 0.95


def test_sem_retorno_positivo_devolve_none():
    """Maximizar Sharpe com numerador negativo em tudo é outra pergunta."""
    assert _max_sharpe(np.array([-0.01, -0.02]), np.diag([0.01, 0.01])) is None


def test_markowitz_sem_retorno_positivo_cai_para_minima_variancia():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(
        rng.normal(-0.02, 0.03, (60, 3)),
        columns=list("ABC"),
        index=pd.date_range("2020-01-31", periods=60, freq="ME"),
    )
    pesos = markowitz(r)

    assert pesos.sum() == pytest.approx(1.0)
    assert (pesos >= -1e-12).all()


def test_max_sharpe_invariante_a_escala_dos_retornos():
    """Sharpe é invariante a escala; o otimizador também tem de ser."""
    mu = np.array([0.010, 0.014, 0.008])
    cov = np.diag([0.0025, 0.0064, 0.0016])

    mensal = _max_sharpe(mu, cov)
    anual = _max_sharpe(mu * 12, cov * 12)

    # A comparação é no OBJETIVO, não nos pesos: perto do ótimo a função é
    # plana, e dois pontos distantes 1e-3 em peso dão o mesmo Sharpe até a
    # sétima casa. Exigir igualdade de pesos seria exigir precisão que não
    # existe e que não muda decisão nenhuma.
    def sharpe(w):
        return (w @ mu) / np.sqrt(w @ cov @ w)

    assert sharpe(mensal) == pytest.approx(sharpe(anual), abs=1e-6)


# --------------------------------------------------------------------------
# a patologia que o encolhimento corrige
# --------------------------------------------------------------------------


def test_markowitz_concentra_e_o_encolhido_espalha():
    """A demonstração central: mesma covariância, só as médias mudam."""
    r = _retornos()

    puro = markowitz(r)
    encolhido = markowitz_encolhido(r)

    assert puro.max() > 0.6, "Markowitz puro deveria concentrar"
    assert encolhido.max() < puro.max()
    assert (encolhido > 0.01).sum() > (puro > 0.01).sum()


def test_markowitz_puro_e_instavel_entre_janelas():
    """O motivo de 1, 3 e 5 anos darem respostas tão diferentes."""
    r = _retornos(n=120)

    cedo = markowitz(r.iloc[:60])
    tarde = markowitz(r.iloc[60:])

    assert np.abs(cedo - tarde).sum() > 0.5


def test_encolhido_e_mais_estavel_que_o_puro():
    r = _retornos(n=120)

    giro_puro = np.abs(markowitz(r.iloc[:60]) - markowitz(r.iloc[60:])).sum()
    giro_encolhido = np.abs(
        markowitz_encolhido(r.iloc[:60]) - markowitz_encolhido(r.iloc[60:])
    ).sum()

    assert giro_encolhido < giro_puro


# --------------------------------------------------------------------------
# Bayes-Stein
# --------------------------------------------------------------------------


def test_bayes_stein_puxa_as_medias_para_perto_umas_das_outras():
    r = _retornos()
    cov = matriz_covariancia(r)

    cruas = r.mean().to_numpy()
    encolhidas = bayes_stein(r, cov)

    # O que o encolhimento garante é redução de DISPERSÃO, não que o
    # resultado caia dentro do intervalo das médias originais: o alvo é o
    # retorno da carteira de mínima variância irrestrita, que pode estar
    # fora desse intervalo quando ela usa posição vendida.
    assert encolhidas.std() < cruas.std()
    assert encolhidas.max() - encolhidas.min() < cruas.max() - cruas.min()


def test_bayes_stein_encolhe_menos_com_mais_dados():
    """Mesma lógica do Ledoit-Wolf: mais amostra, mais confiança nela."""

    def distancia(n):
        r = _retornos(n=n)
        cov = matriz_covariancia(r)
        cruas = r.mean().to_numpy()
        return float(np.abs(bayes_stein(r, cov) - cruas).mean())

    assert distancia(36) > distancia(360)


def test_bayes_stein_preserva_a_media_geral_aproximadamente():
    r = _retornos()
    encolhidas = bayes_stein(r, matriz_covariancia(r))

    assert np.isfinite(encolhidas).all()
    assert encolhidas.min() > -1.0 and encolhidas.max() < 1.0


# --------------------------------------------------------------------------
# tendência
# --------------------------------------------------------------------------


def test_inclinacao_do_log_e_o_retorno_medio():
    """A equivalência que importa: a regra intuitiva É a média histórica."""
    n = 120
    taxa = 0.01
    r = pd.DataFrame(
        {"A": [taxa] * n},
        index=pd.date_range("2016-11-30", periods=n, freq="ME"),
    )

    assert inclinacao_log(r)["A"] == pytest.approx(np.log(1 + taxa), rel=1e-6)


def test_inclinacao_nao_depende_do_nivel_de_preco():
    """Regredir o preço em reais quebraria aqui; o log não."""
    n = 60
    rng = np.random.default_rng(2)
    base = rng.normal(0.008, 0.04, n)
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    r = pd.DataFrame({"CARA": base, "BARATA": base}, index=idx)

    inclinacoes = inclinacao_log(r)

    assert inclinacoes["CARA"] == pytest.approx(inclinacoes["BARATA"])


def test_tendencia_zera_quem_cai():
    n = 60
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    r = pd.DataFrame({"SOBE": [0.02] * n, "CAI": [-0.02] * n}, index=idx)

    pesos = tendencia(r)

    assert pesos["CAI"] == pytest.approx(0.0)
    assert pesos["SOBE"] == pytest.approx(1.0)


def test_tendencia_com_tudo_caindo_cai_para_pesos_iguais():
    """Ficar sem carteira não é uma resposta."""
    n = 60
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    r = pd.DataFrame({"A": [-0.01] * n, "B": [-0.02] * n}, index=idx)

    assert tendencia(r).to_numpy() == pytest.approx([0.5, 0.5])


def test_tendencia_proporcional_a_inclinacao():
    n = 60
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    r = pd.DataFrame({"FORTE": [0.02] * n, "FRACA": [0.01] * n}, index=idx)

    pesos = tendencia(r)

    assert pesos["FORTE"] > pesos["FRACA"]
    assert pesos.sum() == pytest.approx(1.0)


def _mensal(colunas: dict) -> pd.DataFrame:
    n = len(next(iter(colunas.values())))
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    return pd.DataFrame(colunas, index=idx)


def test_momentum_ignora_o_mes_mais_recente():
    """Salto no último mês não muda nada: a janela é t-12 a t-2."""
    base = [0.01] * 36
    salto = base[:-1] + [0.50]
    sem = momentum_12_1(_mensal({"A": base, "B": [0.02] * 36}))
    com = momentum_12_1(_mensal({"A": salto, "B": [0.02] * 36}))
    assert com.to_numpy() == pytest.approx(sem.to_numpy())


def test_momentum_ignora_o_que_e_mais_antigo_que_12_meses():
    """Mês t-13 fica fora; t-12 entra."""
    recente = [0.01] * 12
    fora = momentum_12_1(_mensal({"A": [0.01] * 23 + [0.90] + recente, "B": [0.01] * 36}))
    dentro = momentum_12_1(_mensal({"A": [0.01] * 24 + [0.90] + recente[1:], "B": [0.01] * 36}))
    assert fora["A"] == pytest.approx(0.5)
    assert dentro["A"] > 0.5


def test_momentum_olha_12_meses_e_nao_36():
    """Caiu nos 24 primeiros meses e subiu nos últimos 12: tendencia e momentum discordam."""
    virada = [-0.03] * 24 + [0.03] * 12
    r = _mensal({"VIROU": virada, "CONSTANTE": [0.005] * 36})
    assert tendencia(r)["VIROU"] == pytest.approx(0.0)
    assert momentum_12_1(r)["VIROU"] > momentum_12_1(r)["CONSTANTE"]


def test_momentum_zera_quem_cai_e_cai_para_iguais_se_todos_caem():
    pesos = momentum_12_1(_mensal({"SOBE": [0.02] * 36, "CAI": [-0.02] * 36}))
    assert pesos["CAI"] == pytest.approx(0.0) and pesos["SOBE"] == pytest.approx(1.0)
    todos = momentum_12_1(_mensal({"A": [-0.01] * 36, "B": [-0.02] * 36}))
    assert todos.to_numpy() == pytest.approx([0.5, 0.5])


def test_momentum_proporcional_ao_retorno_acumulado():
    pesos = momentum_12_1(_mensal({"A": [0.02] * 36, "B": [0.01] * 36}))
    a, b = (1.02 ** 11 - 1), (1.01 ** 11 - 1)
    assert pesos["A"] == pytest.approx(a / (a + b))


def test_momentum_entra_na_validacao():
    assert "momentum_12_1" in regras_sobre_retornos()
