"""Testes do plano de rebalanceamento (`rebalance.plano_aporte`).

Carteira fictícia de 3 papéis a R$ 10, total R$ 1.000, alvo 1/3 cada:
AAAA3 está 20pp acima, CCCC3 20pp abaixo, BBBB4 no alvo.
"""

import pandas as pd
import pytest

import json

from carteira.rebalance import AlvoInvalido, carregar_alvo, formatar, plano_aporte, validar_alvo

PRECOS = pd.Series({"AAAA3": 10.0, "BBBB4": 10.0, "CCCC3": 10.0})
ALVO = pd.Series({"AAAA3": 1 / 3, "BBBB4": 1 / 3, "CCCC3": 1 / 3})


@pytest.fixture
def posicao():
    return pd.Series({"AAAA3": 53.0, "BBBB4": 33.0, "CCCC3": 14.0})


def _col(plano, coluna):
    return plano["tabela"][coluna].to_dict()


# ---- comportamento antigo: só compra -------------------------------------


def test_sem_aporte_e_sem_venda_nao_mexe(posicao):
    plano = plano_aporte(posicao, PRECOS, ALVO)
    assert set(_col(plano, "comprar").values()) == {0}
    assert set(_col(plano, "vender").values()) == {0}


def test_aporte_vai_para_o_mais_defasado(posicao):
    # total depois = 1.100, alvo = 366,67 cada; déficits B 36,67 e C 226,67
    # rateio dá B 1 e C 8; o troco de R$ 10 vai para C (maior déficit)
    plano = plano_aporte(posicao, PRECOS, ALVO, aporte=100.0)
    assert _col(plano, "comprar") == {"AAAA3": 0, "BBBB4": 1, "CCCC3": 9}
    assert plano["troco"] == pytest.approx(0.0)


def test_padrao_continua_sem_venda(posicao):
    plano = plano_aporte(posicao, PRECOS, ALVO, aporte=100.0)
    assert plano["vendido"] == 0
    assert set(_col(plano, "vender").values()) == {0}


# ---- venda ---------------------------------------------------------------


def test_venda_sem_aporte_zera_o_desvio(posicao):
    plano = plano_aporte(posicao, PRECOS, ALVO, permitir_venda=True)
    assert _col(plano, "vender") == {"AAAA3": 19, "BBBB4": 0, "CCCC3": 0}
    assert _col(plano, "comprar") == {"AAAA3": 0, "BBBB4": 0, "CCCC3": 19}
    assert plano["vendido"] == pytest.approx(190.0)
    assert plano["investido"] == pytest.approx(190.0)


def test_venda_soma_ao_aporte(posicao):
    plano = plano_aporte(posicao, PRECOS, ALVO, aporte=50.0, permitir_venda=True)
    # total depois = 1.050, alvo = 350 cada
    assert _col(plano, "vender") == {"AAAA3": 18, "BBBB4": 0, "CCCC3": 0}
    assert _col(plano, "comprar") == {"AAAA3": 0, "BBBB4": 2, "CCCC3": 21}
    assert plano["investido"] + plano["troco"] == pytest.approx(50.0 + plano["vendido"])


def test_venda_arredonda_para_baixo():
    posicao = pd.Series({"AAAA3": 10.0, "BBBB4": 5.0})
    precos = pd.Series({"AAAA3": 7.0, "BBBB4": 7.0})
    alvo = pd.Series({"AAAA3": 0.5, "BBBB4": 0.5})
    # excesso de AAAA3 = 70 - 52,5 = 17,5 → 2 ações (14), não 3 (21)
    plano = plano_aporte(posicao, precos, alvo, permitir_venda=True)
    assert _col(plano, "vender")["AAAA3"] == 2


def test_peso_zero_no_alvo_vende_tudo(posicao):
    alvo = pd.Series({"AAAA3": 0.0, "BBBB4": 0.5, "CCCC3": 0.5})
    plano = plano_aporte(posicao, PRECOS, alvo, permitir_venda=True)
    assert _col(plano, "vender")["AAAA3"] == 53


def test_fora_do_alvo_nao_e_vendido(posicao):
    alvo = pd.Series({"BBBB4": 0.5, "CCCC3": 0.5})
    plano = plano_aporte(posicao, PRECOS, alvo, permitir_venda=True)
    assert _col(plano, "vender")["AAAA3"] == 0
    assert plano["fora_do_alvo"] == ["AAAA3"]


def test_formatar_mostra_venda(posicao):
    texto = formatar(plano_aporte(posicao, PRECOS, ALVO, permitir_venda=True))
    assert "vender" in texto
    assert "Vendido" in texto


# ---- validar_alvo: o que vem do navegador ---------------------------------


def test_alvo_valido_vira_serie_ordenada():
    alvo = validar_alvo({"PETR4": 0.5, "BBAS3": 0.5}, avisar=None)
    assert list(alvo.index) == ["BBAS3", "PETR4"]
    assert alvo.sum() == pytest.approx(1.0)


def test_ticker_e_padronizado():
    alvo = validar_alvo({" petr4 ": 1.0}, avisar=None)
    assert list(alvo.index) == ["PETR4"]


def test_ticker_repetido_depois_de_padronizar_falha():
    with pytest.raises(AlvoInvalido, match="repetido"):
        validar_alvo({"PETR4": 0.5, "petr4": 0.5}, avisar=None)


def test_soma_9990_e_normalizada_com_aviso():
    avisos = []
    alvo = validar_alvo({"A": 0.4995, "B": 0.4995}, avisar=avisos.append)
    assert alvo.sum() == pytest.approx(1.0)
    assert len(avisos) == 1


@pytest.mark.parametrize(
    "pesos, trecho",
    [
        ({}, "vazio"),
        ([["PETR4", 1.0]], "vazio"),
        ({"PETR4": 0.6, "BBAS3": 0.6}, "somam"),
        ({"PETR4": 1.2, "BBAS3": -0.2}, "negativo"),
        ({"PETR4": "1"}, "inválido"),
        ({"PETR4": None}, "inválido"),
        ({"PETR4": True}, "inválido"),
        ({"PETR4": float("nan")}, "inválido"),
        ({"PETR4": float("inf")}, "inválido"),
        ({"": 1.0}, "Ticker inválido"),
    ],
)
def test_alvo_invalido_falha_com_mensagem(pesos, trecho):
    with pytest.raises(AlvoInvalido, match=trecho):
        validar_alvo(pesos, avisar=None)


def test_carregar_alvo_continua_lendo_arquivo(tmp_path):
    arq = tmp_path / "alvo.json"
    arq.write_text(json.dumps({"PETR4": 0.5, "BBAS3": 0.5}), encoding="utf-8")
    assert carregar_alvo(arq, avisar=None).to_dict() == {"BBAS3": 0.5, "PETR4": 0.5}
