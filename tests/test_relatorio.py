"""Testes do relatório de risco.

Verificam o contrato da camada: `analise_risco` devolve dados, `formatar`
devolve texto, e os dois não se misturam. É esse contrato que vai permitir
o Flask consumir a análise sem extrair número de string.
"""

import numpy as np
import pandas as pd
import pytest

from carteira.blocos import BlocosInvalidos
from carteira.relatorio import analise_risco, formatar

TICKERS = ["BANCO1", "BANCO2", "MINERIO", "ELETRICA", "EXTERIOR"]


@pytest.fixture
def retornos():
    """Retornos mensais com estrutura de blocos parecida com a real:
    dois bancos juntos, uma commodity, uma elétrica e um ativo externo
    com correlação negativa (sobe quando o Brasil cai)."""
    rng = np.random.default_rng(42)
    n = 48
    brasil = rng.normal(0.0, 0.05, n)

    dados = {
        "BANCO1": brasil * 1.1 + rng.normal(0, 0.02, n),
        "BANCO2": brasil * 1.0 + rng.normal(0, 0.02, n),
        "MINERIO": brasil * 0.6 + rng.normal(0, 0.07, n),
        "ELETRICA": brasil * 0.5 + rng.normal(0, 0.03, n),
        "EXTERIOR": -brasil * 0.3 + rng.normal(0, 0.04, n),
    }
    return pd.DataFrame(
        dados, index=pd.date_range("2022-01-31", periods=n, freq="ME")
    )


@pytest.fixture
def pesos():
    return pd.Series(
        {
            "BANCO1": 0.30,
            "BANCO2": 0.20,
            "MINERIO": 0.25,
            "ELETRICA": 0.20,
            "EXTERIOR": 0.05,
        }
    )


@pytest.fixture
def dimensoes():
    return {
        "setor": pd.Series(
            {
                "BANCO1": "bancos",
                "BANCO2": "bancos",
                "MINERIO": "commodities",
                "ELETRICA": "eletricas",
                "EXTERIOR": "exterior",
            }
        )
    }


# --------------------------------------------------------------------------
# contrato
# --------------------------------------------------------------------------


def test_analise_devolve_dados_nao_texto(retornos, pesos):
    analise = analise_risco(retornos, pesos)

    assert isinstance(analise["apostas_efetivas"], float)
    assert isinstance(analise["contribuicao"], pd.DataFrame)
    assert isinstance(analise["componentes"], pd.DataFrame)

    # String como metadado (`metodo`) é legítima; o que não pode haver é
    # texto já renderizado — percentual formatado, quebra de linha, barra de
    # progresso. Isso pertence a `formatar`, e um consumidor como o Flask
    # teria de desfazer para usar.
    textos = [v for v in analise.values() if isinstance(v, str)]
    assert textos == ["ledoit_wolf"]
    for texto in textos:
        assert "\n" not in texto and "%" not in texto


def test_formatar_devolve_texto(retornos, pesos, dimensoes):
    texto = formatar(analise_risco(retornos, pesos, dimensoes))

    assert isinstance(texto, str)
    assert "APOSTAS EFETIVAS" in texto
    assert "CAPITAL CONTRA RISCO" in texto
    assert "POR SETOR" in texto


# --------------------------------------------------------------------------
# conteúdo
# --------------------------------------------------------------------------


def test_pesos_sao_normalizados(retornos):
    analise = analise_risco(retornos, pd.Series({t: 10.0 for t in TICKERS}))

    assert analise["pesos"].sum() == pytest.approx(1.0)


def test_apostas_efetivas_abaixo_do_numero_de_ativos(retornos, pesos):
    """Com um fator comum forte, 5 papéis não são 5 apostas."""
    analise = analise_risco(retornos, pesos)

    assert 1.0 < analise["apostas_efetivas"] < 5.0
    assert analise["n_ativos"] == 5


def test_bancos_aparecem_como_par_correlacionado(retornos, pesos):
    pares = analise_risco(retornos, pesos)["pares_correlacionados"]

    assert {"BANCO1", "BANCO2"} <= set(pares["a"]) | set(pares["b"])


def test_ativo_de_correlacao_negativa_contribui_pouco(retornos, pesos):
    """Proteção real aparece como contribuição de risco baixa — ou negativa."""
    contribuicao = analise_risco(retornos, pesos)["contribuicao"]

    assert contribuicao.loc["EXTERIOR", "razao"] < 1.0


def test_bloco_agregado_soma_um(retornos, pesos, dimensoes):
    setor = analise_risco(retornos, pesos, dimensoes)["por_bloco"]["setor"]

    assert setor["peso_capital"].sum() == pytest.approx(1.0)
    assert setor["peso_risco"].sum() == pytest.approx(1.0)


def test_volatilidade_estimada_bate_com_a_realizada(retornos, pesos):
    """Duas rotas para o mesmo número: pela matriz e pela série da carteira.

    Não são idênticas — a matriz é encolhida e a série não — mas uma
    discrepância grande indicaria erro de anualização ou de alinhamento.
    """
    analise = analise_risco(retornos, pesos)

    assert analise["volatilidade_anual"] == pytest.approx(
        analise["volatilidade_realizada"], rel=0.15
    )


def test_metodo_amostral_nao_encolhe(retornos, pesos):
    analise = analise_risco(retornos, pesos, metodo="amostral")

    assert analise["intensidade_encolhimento"] == 0.0


# --------------------------------------------------------------------------
# casos de borda
# --------------------------------------------------------------------------


def test_ativo_sem_serie_e_reportado_e_nao_quebra(retornos, pesos):
    """Papel recém-comprado ainda sem histórico não pode derrubar a análise."""
    com_novato = pd.concat([pesos, pd.Series({"NOVATO": 0.05})])

    analise = analise_risco(retornos, com_novato)

    assert analise["ignorados"] == ["NOVATO"]
    assert analise["n_ativos"] == 5
    assert analise["pesos"].sum() == pytest.approx(1.0)


def test_nenhum_ativo_em_comum_falha(retornos):
    with pytest.raises(ValueError, match="Nenhum ativo"):
        analise_risco(retornos, pd.Series({"DESCONHECIDO": 1.0}))


def test_bloco_incompleto_falha(retornos, pesos):
    incompleto = {"setor": pd.Series({"BANCO1": "bancos"})}

    with pytest.raises(BlocosInvalidos, match="setor"):
        analise_risco(retornos, pesos, incompleto)


def test_janela_curta_gera_aviso(retornos, pesos):
    texto = formatar(analise_risco(retornos.head(18), pesos))

    assert "AVISO" in texto


def test_janela_longa_nao_gera_aviso(retornos, pesos):
    assert "AVISO" not in formatar(analise_risco(retornos, pesos))
