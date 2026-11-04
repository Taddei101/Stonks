"""Testes das séries de retorno.

Os valores são conferidos à mão de propósito. Erro em retorno não levanta
exceção: propaga para a covariância, de lá para a análise inteira, e o
resultado continua parecendo plausível.
"""

import numpy as np
import pandas as pd
import pytest

from carteira.series import (
    HistoricoInsuficiente,
    acumulado,
    anualizar_retorno,
    anualizar_volatilidade,
    periodo_comum,
    retorno_carteira,
    retornos_diarios,
    retornos_mensais,
)


@pytest.fixture
def precos():
    """Três dias úteis, dois ativos, números redondos."""
    return pd.DataFrame(
        {"AAAA3": [100.0, 110.0, 99.0], "BBBB4": [50.0, 50.0, 55.0]},
        index=pd.date_range("2024-01-02", periods=3, freq="B"),
    )


# --------------------------------------------------------------------------
# periodo_comum
# --------------------------------------------------------------------------


def test_recorta_para_janela_com_todos_os_ativos():
    """Ativo que começa depois encurta a janela de todo mundo. É proposital."""
    precos = pd.DataFrame(
        {"ANTIGO": [10.0, 11.0, 12.0, 13.0], "NOVO": [np.nan, np.nan, 20.0, 22.0]},
        index=pd.date_range("2024-01-02", periods=4, freq="B"),
    )
    recorte = periodo_comum(precos)

    assert len(recorte) == 2
    assert recorte.index[0] == pd.Timestamp("2024-01-04")


def test_sem_periodo_comum_falha_dizendo_quem_limita():
    precos = pd.DataFrame(
        {"ANTIGO": [10.0, 11.0, 12.0], "NOVO": [np.nan, np.nan, np.nan]},
        index=pd.date_range("2024-01-02", periods=3, freq="B"),
    )
    with pytest.raises(HistoricoInsuficiente, match="NOVO"):
        periodo_comum(precos)


# --------------------------------------------------------------------------
# retornos
# --------------------------------------------------------------------------


def test_retornos_simples(precos):
    r = retornos_diarios(precos, metodo="simples")

    assert r["AAAA3"].tolist() == pytest.approx([0.10, -0.10])
    assert r["BBBB4"].tolist() == pytest.approx([0.0, 0.10])


def test_retornos_log(precos):
    r = retornos_diarios(precos, metodo="log")

    assert r["AAAA3"].iloc[0] == pytest.approx(np.log(1.10))
    assert r["AAAA3"].iloc[1] == pytest.approx(np.log(0.90))


def test_metodo_invalido(precos):
    with pytest.raises(ValueError, match="simples"):
        retornos_diarios(precos, metodo="geometrico")


def test_mensal_usa_o_ultimo_preco_do_mes():
    dias = pd.date_range("2024-01-01", "2024-03-31", freq="B")
    precos = pd.DataFrame({"AAAA3": np.linspace(100.0, 160.0, len(dias))}, index=dias)

    mensal = retornos_mensais(precos)
    fim_do_mes = precos.resample("ME").last()["AAAA3"]

    assert len(mensal) == 2
    assert mensal["AAAA3"].iloc[0] == pytest.approx(
        fim_do_mes.iloc[1] / fim_do_mes.iloc[0] - 1
    )


# --------------------------------------------------------------------------
# A razão de existir do módulo: provento tem de entrar no retorno
# --------------------------------------------------------------------------


def test_dividendo_nao_pode_virar_queda_de_preco():
    """Confere o motivo pelo qual a série de risco precisa ser ajustada.

    Caso: ação a R$ 100 que paga R$ 5 de dividendo e, na data-ex, negocia a
    R$ 95. Quem recebeu o provento não perdeu nada — o retorno total é zero.

    Na série NÃO ajustada o preço cai 5%, e é isso que entraria na matriz de
    covariância se `historico` fosse usado no lugar de `historico_total`.
    Na série ajustada, o preço anterior é reescalado e o retorno dá zero.
    """
    bruto = pd.DataFrame(
        {"AAAA3": [100.0, 95.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    # Ajuste do yfinance: multiplica o histórico anterior por (1 - D/P).
    fator = 1 - 5.0 / 100.0
    ajustado = pd.DataFrame(
        {"AAAA3": [100.0 * fator, 95.0]},
        index=bruto.index,
    )

    assert retornos_diarios(bruto)["AAAA3"].iloc[0] == pytest.approx(-0.05)
    assert retornos_diarios(ajustado)["AAAA3"].iloc[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# retorno_carteira
# --------------------------------------------------------------------------


def test_carteira_e_media_ponderada(precos):
    r = retornos_diarios(precos)
    pesos = pd.Series({"AAAA3": 0.5, "BBBB4": 0.5})

    carteira = retorno_carteira(r, pesos)

    # dia 1: 0,5*0,10 + 0,5*0,00 = 0,05
    # dia 2: 0,5*(-0,10) + 0,5*0,10 = 0,00
    assert carteira.tolist() == pytest.approx([0.05, 0.0])


def test_pesos_sao_normalizados(precos):
    r = retornos_diarios(precos)
    cheio = retorno_carteira(r, pd.Series({"AAAA3": 0.5, "BBBB4": 0.5}))
    dobrado = retorno_carteira(r, pd.Series({"AAAA3": 50.0, "BBBB4": 50.0}))

    assert cheio.tolist() == pytest.approx(dobrado.tolist())


def test_peso_sem_serie_de_retorno_falha(precos):
    r = retornos_diarios(precos)
    with pytest.raises(KeyError, match="CCCC11"):
        retorno_carteira(r, pd.Series({"AAAA3": 0.5, "CCCC11": 0.5}))


def test_log_nao_serve_para_carteira(precos):
    """Documenta por que `retorno_carteira` exige retorno simples.

    A média ponderada de log-retornos NÃO é o log-retorno da carteira. Aqui a
    diferença já aparece na segunda casa; com volatilidade maior, cresce.
    """
    pesos = pd.Series({"AAAA3": 0.5, "BBBB4": 0.5})
    certo = retorno_carteira(retornos_diarios(precos, "simples"), pesos)
    errado = retorno_carteira(retornos_diarios(precos, "log"), pesos)

    assert certo.iloc[0] != pytest.approx(errado.iloc[0], abs=1e-4)


# --------------------------------------------------------------------------
# anualização
# --------------------------------------------------------------------------


def test_acumulado(precos):
    acum = acumulado(retornos_diarios(precos)["AAAA3"])

    # 1,10 * 0,90 - 1 = -0,01
    assert acum.iloc[-1] == pytest.approx(-0.01)


def test_retorno_anualizado_e_geometrico():
    um_por_cento = pd.Series([0.01] * 12)

    assert anualizar_retorno(um_por_cento) == pytest.approx(1.01**12 - 1)


def test_geometrico_difere_do_aritmetico_em_serie_volatil():
    """+50% seguido de -50% dá -25%, não 0%. É o furo da média aritmética."""
    volatil = pd.Series([0.5, -0.5] * 6)

    assert volatil.mean() == pytest.approx(0.0)
    assert anualizar_retorno(volatil) < -0.2


def test_volatilidade_anualizada():
    mensal = pd.Series([0.02, -0.02] * 6)
    esperado = mensal.std(ddof=1) * np.sqrt(12)

    assert anualizar_volatilidade(mensal) == pytest.approx(esperado)


def test_volatilidade_com_uma_observacao_e_nan():
    assert np.isnan(anualizar_volatilidade(pd.Series([0.01])))


# --------------------------------------------------------------------------
# mês em andamento
# --------------------------------------------------------------------------


def test_mes_corrente_incompleto_e_descartado():
    """Três pregões de novembro não podem virar uma observação mensal.

    `resample("ME")` rotula o balde do mês corrente com 30/11 mesmo tendo
    só o começo do mês. O retorno parcial entraria com o mesmo peso de um
    mês cheio e puxaria a volatilidade estimada para baixo.
    """
    dias = pd.bdate_range("2024-01-02", "2024-03-05")
    precos = pd.DataFrame({"AAAA3": np.linspace(100.0, 130.0, len(dias))}, index=dias)

    mensal = retornos_mensais(precos)

    # Jan e fev fecham; março tem 3 pregões e fica de fora.
    assert len(mensal) == 1
    assert mensal.index[-1] == pd.Timestamp("2024-02-29")


def test_mes_completo_e_mantido():
    """Se a série vai até o último pregão do mês, o mês conta."""
    dias = pd.bdate_range("2024-01-02", "2024-03-29")
    precos = pd.DataFrame({"AAAA3": np.linspace(100.0, 130.0, len(dias))}, index=dias)

    mensal = retornos_mensais(precos)

    assert len(mensal) == 2
    assert mensal.index[-1] == pd.Timestamp("2024-03-31")


def test_mes_parcial_fica_fora_da_amostra():
    """O retorno de poucos dias não entra na série mensal.

    O problema não é a DIREÇÃO do viés — um retorno de dois dias pode
    aumentar ou reduzir o desvio padrão medido, dependendo de onde ele cai
    em relação aos demais. O problema é que ele vem de outra distribuição e
    entra com o mesmo peso de um mês inteiro.
    """
    dias = pd.bdate_range("2024-01-02", "2024-06-04")
    rng = np.random.default_rng(3)
    precos = pd.DataFrame(
        {"AAAA3": 100 * np.cumprod(1 + rng.normal(0, 0.02, len(dias)))}, index=dias
    )

    sem_correcao = precos.resample("ME").last().pct_change().dropna()
    com_correcao = retornos_mensais(precos)

    assert len(com_correcao) == len(sem_correcao) - 1
    assert pd.Timestamp("2024-06-30") in sem_correcao.index
    assert pd.Timestamp("2024-06-30") not in com_correcao.index


def test_mes_que_fecha_em_feriado_nao_e_descartado():
    """Março de 2024 terminou na sexta-feira santa, sem pregão no dia 29.

    Um critério baseado no rótulo do balde diria que março está incompleto.
    A contagem de pregões não cai nessa.
    """
    dias = pd.bdate_range("2024-01-02", "2024-03-28")  # sem a sexta santa
    precos = pd.DataFrame({"AAAA3": np.linspace(100.0, 130.0, len(dias))}, index=dias)

    mensal = retornos_mensais(precos)

    assert pd.Timestamp("2024-03-31") in mensal.index


# --------------------------------------------------------------------------
# aviso de truncamento
# --------------------------------------------------------------------------


def test_avisa_quando_um_ativo_corta_o_periodo():
    """Truncar em silêncio produz relatório com aparência normal e errado."""
    dias = pd.bdate_range("2016-01-01", "2026-01-01")
    precos = pd.DataFrame(
        {"VELHO": np.linspace(10, 50, len(dias)), "NOVO": np.nan}, index=dias
    )
    precos.loc[precos.index >= "2024-01-01", "NOVO"] = 20.0

    avisos = []
    retornos_mensais(precos, avisar=avisos.append)

    assert len(avisos) == 1
    assert "NOVO" in avisos[0]
    assert "01/2024" in avisos[0]


def test_nao_avisa_quando_nao_ha_truncamento():
    dias = pd.bdate_range("2020-01-01", "2024-01-01")
    precos = pd.DataFrame(
        {"A": np.linspace(10, 20, len(dias)), "B": np.linspace(5, 9, len(dias))},
        index=dias,
    )

    avisos = []
    retornos_mensais(precos, avisar=avisos.append)

    assert avisos == []


def test_limitantes_lista_quem_encurta_a_janela():
    from carteira.series import limitantes

    dias = pd.bdate_range("2016-01-01", "2026-01-01")
    precos = pd.DataFrame(
        {"VELHO": 1.0, "MEDIO": np.nan, "NOVO": np.nan}, index=dias
    )
    precos.loc[precos.index >= "2020-01-01", "MEDIO"] = 2.0
    precos.loc[precos.index >= "2024-01-01", "NOVO"] = 3.0

    tabela = limitantes(precos)

    assert list(tabela.index) == ["NOVO", "MEDIO"]
    assert "VELHO" not in tabela.index
