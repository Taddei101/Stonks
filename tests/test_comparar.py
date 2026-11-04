"""Testes da comparação entre alocações.

O comportamento mais importante a fixar é o negativo: quando duas carteiras
vêm da mesma distribuição, a comparação tem de dizer "empate", não eleger
uma vencedora. É o erro que o módulo existe para evitar.
"""

import numpy as np
import pandas as pd
import pytest

from carteira.comparar import (
    comparar,
    formatar,
    intervalo,
    media_piores,
    metricas,
    perda_maxima,
    probabilidade_de_superar,
    sharpe,
)

RAPIDO = 300  # reamostragens suficientes para teste, sem custo


def _mensal(valores, inicio="2022-01-31"):
    return pd.Series(
        valores, index=pd.date_range(inicio, periods=len(valores), freq="ME")
    )


# --------------------------------------------------------------------------
# perda máxima
# --------------------------------------------------------------------------


def test_perda_maxima_de_queda_conhecida():
    """+10%, -20%, +5%: o fundo é 20% abaixo do topo."""
    resultado = perda_maxima(_mensal([0.10, -0.20, 0.05]))

    assert resultado["perda_maxima"] == pytest.approx(-0.20)


def test_perda_maxima_acumula_quedas_seguidas():
    """Duas quedas de 10% compõem para 19%, não somam para 20%."""
    resultado = perda_maxima(_mensal([-0.10, -0.10, 0.50]))

    assert resultado["perda_maxima"] == pytest.approx(-0.19)


def test_carteira_so_de_alta_nao_tem_perda():
    assert perda_maxima(_mensal([0.01] * 12))["perda_maxima"] == pytest.approx(0.0)


def test_meses_ate_recuperar():
    """Cai 50% e volta: só recupera quando o patrimônio reencontra o topo."""
    resultado = perda_maxima(_mensal([-0.50, 0.40, 0.50]))

    assert resultado["perda_maxima"] == pytest.approx(-0.50)
    assert resultado["meses_ate_recuperar"] == 3


def test_sem_recuperacao_devolve_nan():
    resultado = perda_maxima(_mensal([0.10, -0.50, 0.01]))

    assert not np.isfinite(resultado["meses_ate_recuperar"])


# --------------------------------------------------------------------------
# cauda
# --------------------------------------------------------------------------


def test_media_dos_piores_olha_so_a_esquerda():
    valores = [-0.30, -0.20, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    assert media_piores(_mensal(valores), fracao=0.20) == pytest.approx(-0.25)


def test_cauda_distingue_carteiras_de_mesma_volatilidade():
    """Desvio padrão trata alta e baixa igual; a cauda não."""
    simetrica = _mensal([0.10, -0.10] * 10)
    assimetrica = _mensal([0.0526] * 18 + [-0.40, -0.40])

    assert media_piores(assimetrica) < media_piores(simetrica)


# --------------------------------------------------------------------------
# Sharpe
# --------------------------------------------------------------------------


def test_sharpe_desconta_o_cdi():
    retornos = _mensal([0.02] * 24)
    cdi = _mensal([0.01] * 24)

    assert sharpe(retornos, cdi) != pytest.approx(sharpe(retornos))


def test_carteira_que_so_empata_com_o_cdi_tem_excesso_zero():
    iguais = _mensal([0.01] * 24)

    assert metricas(iguais, iguais)["excesso_anual"] == pytest.approx(0.0, abs=1e-9)


def test_volatilidade_zero_nao_quebra():
    constante = _mensal([0.01] * 12)

    assert not np.isfinite(sharpe(constante))


# --------------------------------------------------------------------------
# métricas: geométrico, não aritmético
# --------------------------------------------------------------------------


def test_retorno_anual_e_geometrico():
    """+50% e -50% alternados: média aritmética zero, resultado negativo."""
    volatil = _mensal([0.5, -0.5] * 12)

    assert volatil.mean() == pytest.approx(0.0)
    assert metricas(volatil)["retorno_anual"] < -0.2


def test_mesma_media_aritmetica_retornos_geometricos_diferentes():
    """O ponto central: volatilidade desconta retorno composto."""
    estavel = _mensal([0.02] * 24)
    volatil = _mensal([0.22, -0.18] * 12)

    assert estavel.mean() == pytest.approx(volatil.mean())
    assert metricas(estavel)["retorno_anual"] > metricas(volatil)["retorno_anual"]


# --------------------------------------------------------------------------
# honestidade estatística
# --------------------------------------------------------------------------


def test_intervalo_contem_a_estimativa_pontual():
    rng = np.random.default_rng(1)
    serie = _mensal(rng.normal(0.01, 0.05, 48))

    baixo, alto = intervalo(serie, sharpe, reamostragens=RAPIDO)

    assert baixo < sharpe(serie) < alto


def test_amostra_menor_gera_intervalo_mais_largo():
    """Menos dados, mais incerteza. É o que o intervalo existe para mostrar."""
    rng = np.random.default_rng(2)
    longa = _mensal(rng.normal(0.01, 0.05, 120))
    curta = longa.iloc[:24]

    largura_longa = np.diff(intervalo(longa, sharpe, reamostragens=RAPIDO))[0]
    largura_curta = np.diff(intervalo(curta, sharpe, reamostragens=RAPIDO))[0]

    assert largura_curta > largura_longa


def test_amostra_curta_demais_devolve_nan():
    assert not np.isfinite(intervalo(_mensal([0.01, 0.02]), sharpe)[0])


def test_carteiras_identicas_dao_meio_a_meio():
    """A âncora do método: nenhuma diferença, nenhuma vencedora."""
    rng = np.random.default_rng(3)
    serie = _mensal(rng.normal(0.01, 0.05, 36))

    p = probabilidade_de_superar(serie, serie.copy(), reamostragens=RAPIDO)

    assert p == pytest.approx(0.0, abs=1e-9)


def test_calibracao_com_carteiras_da_mesma_distribuicao():
    """Falso positivo tem de ser raro, não impossível.

    Testar isso com UM sorteio não funciona, e a primeira versão deste teste
    caiu nessa: com a semente 7 as duas amostras diferiram por mais de dois
    erros padrão, e o método corretamente apontou diferença. Não era falha
    do método, era azar de sorteio — exatamente o fenômeno que todo este
    módulo existe para tornar visível.

    O teste certo é de calibração: ao longo de muitos pares independentes da
    MESMA distribuição, a fração de vezes em que o método grita "diferença
    real" tem de ficar perto do nível nominal de 10%, não perto de zero nem
    perto da metade.
    """
    rng = np.random.default_rng(99)
    gritou = 0
    pares = 30

    for _ in range(pares):
        a = _mensal(rng.normal(0.01, 0.05, 36))
        b = _mensal(rng.normal(0.01, 0.05, 36))
        p = probabilidade_de_superar(a, b, reamostragens=200, semente=1)
        if p > 0.90 or p < 0.10:
            gritou += 1

    assert gritou / pares < 0.30, (
        f"{gritou}/{pares} falsos positivos: o intervalo está estreito demais"
    )


def test_diferenca_de_amostra_nao_e_diferenca_de_alocacao():
    """Documenta o limite honesto do método.

    O bootstrap responde "a diferença NESTA amostra é robusta a
    reamostragem", não "estas carteiras têm distribuições diferentes".
    Quando o sorteio é atípico, as duas perguntas divergem — e é por isso
    que o relatório nunca diz "a carteira X é melhor", só reporta a
    probabilidade e deixa a conclusão com quem lê.
    """
    rng = np.random.default_rng(7)
    a = _mensal(rng.normal(0.01, 0.05, 36))
    b = _mensal(rng.normal(0.01, 0.05, 36))

    # Mesma distribuição geradora, médias amostrais bem diferentes.
    assert abs(a.mean() - b.mean()) > 2 * (a - b).std() / np.sqrt(36)
    assert probabilidade_de_superar(a, b, reamostragens=RAPIDO) < 0.10


def test_diferenca_grande_e_detectada():
    """O método não pode ser cego a tudo: diferença gritante tem de aparecer."""
    rng = np.random.default_rng(11)
    ruim = _mensal(rng.normal(-0.01, 0.08, 60))
    boa = _mensal(rng.normal(0.02, 0.02, 60))

    assert probabilidade_de_superar(boa, ruim, reamostragens=RAPIDO) > 0.90


# --------------------------------------------------------------------------
# painel
# --------------------------------------------------------------------------


@pytest.fixture
def series():
    rng = np.random.default_rng(5)
    base = rng.normal(0.01, 0.05, 48)
    return {
        "iguais": _mensal(base),
        "defensiva": _mensal(base * 0.6 + 0.002),
        "agressiva": _mensal(base * 1.5),
    }


@pytest.fixture
def cdi():
    return _mensal([0.009] * 48)


def test_painel_tem_uma_linha_por_carteira(series, cdi):
    tabela = comparar(series, cdi, reamostragens=RAPIDO)

    assert set(tabela.index) == set(series)
    assert tabela["sharpe"].is_monotonic_decreasing


def test_referencia_nao_se_compara_consigo(series, cdi):
    tabela = comparar(series, cdi, reamostragens=RAPIDO)

    assert not np.isfinite(tabela.loc["iguais", "prob_superar_ref"])


def test_referencia_inexistente_falha(series):
    with pytest.raises(KeyError, match="minha_carteira"):
        comparar(series, referencia="minha_carteira")


def test_painel_funciona_sem_cdi(series):
    tabela = comparar(series, reamostragens=RAPIDO)

    assert "cdi_anual" not in tabela.columns
    assert tabela["sharpe"].notna().all()


def test_formatar_mostra_o_aviso_sobre_a_largura(series, cdi):
    texto = formatar(comparar(series, cdi, reamostragens=RAPIDO))

    assert "COMPARAÇÃO DE ALOCAÇÕES" in texto
    assert "não são distinguíveis" in texto
    assert "PROBABILIDADE DE SUPERAR" in texto
