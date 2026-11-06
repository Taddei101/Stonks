"""Testes do agrupamento por fator de risco."""

import json

import pandas as pd
import pytest

from carteira.blocos import (
    BlocosInvalidos,
    agrupar,
    carregar_blocos,
    resumo_por_dimensao,
    validar_cobertura,
)
from carteira.rebalance import carregar_alvo

ARQUIVO_BLOCOS = "config/blocos.json"
ARQUIVO_ALVO = "config/alvo.json"


@pytest.fixture
def blocos():
    return carregar_blocos(ARQUIVO_BLOCOS)


def test_le_as_dimensoes(blocos):
    """Uma dimensão só, de propósito.

    Setor, moeda e receita saíram: os componentes principais já captam essa
    estrutura e a captam melhor. Sobra o que a estatística não alcança.
    """
    assert set(blocos) == {"controle"}


def test_nota_nao_vira_dimensao(blocos):
    """`_nota` é comentário: JSON não tem sintaxe própria para isso."""
    assert not any(d.startswith("_") for d in blocos)


def test_blocos_cobrem_o_alvo_real(blocos):
    """O arquivo versionado tem de cobrir a carteira versionada."""
    alvo = carregar_alvo(ARQUIVO_ALVO, avisar=None)
    validar_cobertura(blocos, list(alvo.index))


def test_ticker_sem_bloco_falha_nomeando_o_ticker(blocos):
    with pytest.raises(BlocosInvalidos, match="NOVO3"):
        validar_cobertura(blocos, ["ITUB4", "NOVO3"])


def test_dimensao_malformada_falha(tmp_path):
    arquivo = tmp_path / "blocos.json"
    arquivo.write_text(json.dumps({"setor": ["ITUB4"]}), encoding="utf-8")

    with pytest.raises(BlocosInvalidos, match="setor"):
        carregar_blocos(arquivo)


def test_arquivo_sem_dimensao_falha(tmp_path):
    arquivo = tmp_path / "blocos.json"
    arquivo.write_text(json.dumps({"_nota": "só comentário"}), encoding="utf-8")

    with pytest.raises(BlocosInvalidos, match="Nenhuma dimensão"):
        carregar_blocos(arquivo)


def test_agrupar_soma_dentro_da_categoria():
    pesos = pd.Series({"ITUB4": 0.275, "BBAS3": 0.08, "VALE3": 0.179})
    setor = pd.Series(
        {"ITUB4": "bancos", "BBAS3": "bancos", "VALE3": "commodities"}
    )

    agrupado = agrupar(pesos, setor)

    assert agrupado["bancos"] == pytest.approx(0.355)
    assert agrupado["commodities"] == pytest.approx(0.179)


def test_agrupar_preserva_o_total():
    """Agrupar não pode criar nem destruir peso."""
    pesos = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    mapa = pd.Series({"A": "x", "B": "x", "C": "y"})

    assert agrupar(pesos, mapa).sum() == pytest.approx(pesos.sum())


def test_agrupar_sem_categoria_falha():
    with pytest.raises(BlocosInvalidos, match="ZZZZ3"):
        agrupar(pd.Series({"ZZZZ3": 1.0}), pd.Series({"ITUB4": "bancos"}))


# --------------------------------------------------------------------------
# A concentração real da carteira versionada
# --------------------------------------------------------------------------


def test_exposicao_a_controle_estatal_do_alvo_atual(blocos):
    """Documenta o estado atual, não impõe meta.

    É a exposição que os componentes principais não conseguem medir: risco
    político não se manifestou na janela, então a correlação histórica entre
    PETR4 e BBAS3 não o registra. Se o alvo mudar, este teste muda junto.
    """
    alvo = carregar_alvo(ARQUIVO_ALVO, avisar=None)
    controle = resumo_por_dimensao(alvo, blocos)["controle"]

    assert len(alvo) == 7
    assert controle.sum() == pytest.approx(1.0)
    assert controle["estatal_br"] == pytest.approx(0.196, abs=0.001)
    assert controle["golden_share"] == pytest.approx(0.231, abs=0.001)
