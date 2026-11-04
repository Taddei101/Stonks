"""Testes do caminho extrato -> posição.

Cobre os módulos onde um erro é silencioso: leitura, taxonomia de
movimentação e derivação de quantidade. Erro aqui não levanta exceção —
produz uma carteira errada com aparência normal.
"""

import pandas as pd
import pytest

from carteira.io_extrato import (
    LayoutInesperado,
    carregar_movimentacoes,
    encontrar_extratos,
    validar_layout,
)
from carteira.normalizar import MovimentacaoDesconhecida, _para_float, normalizar
from carteira.posicao import conferir_atualizacoes, construir_ledger, posicao_em


# --------------------------------------------------------------------------
# io_extrato
# --------------------------------------------------------------------------


def test_encontra_extrato(pasta_extratos):
    assert len(encontrar_extratos(pasta_extratos)) == 1


def test_pasta_inexistente_devolve_lista_vazia(tmp_path):
    assert encontrar_extratos(tmp_path / "nao_existe") == []


def test_ignora_lock_do_excel(pasta_extratos):
    (pasta_extratos / "~$movimentacao-2024-12-31-10-00-00.xlsx").write_text("lock")
    assert len(encontrar_extratos(pasta_extratos)) == 1


def test_layout_errado_falha_alto(tmp_path):
    with pytest.raises(LayoutInesperado):
        validar_layout(pd.DataFrame(columns=["Data", "Outra Coisa"]))


def test_pasta_sem_extrato(tmp_path):
    vazia = tmp_path / "extratos"
    vazia.mkdir()
    with pytest.raises(FileNotFoundError):
        carregar_movimentacoes(vazia)


def test_deduplica_extratos_sobrepostos(pasta_extratos_duplicados, pasta_extratos):
    """Dois exports idênticos têm de produzir o mesmo que um só.

    E sem apagar as duas linhas de JCP legítimas: elas são iguais em todas as
    oito colunas da B3, e só o ordinal as distingue de uma repetição.
    """
    um = carregar_movimentacoes(pasta_extratos)
    dois = carregar_movimentacoes(pasta_extratos_duplicados)

    assert len(dois) == len(um)

    jcp = dois[dois["Movimentação"] == "Juros Sobre Capital Próprio"]
    assert len(jcp) == 2


# --------------------------------------------------------------------------
# normalizar
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bruto, esperado",
    [
        ("1855.56", 1855.56),
        ("1.855,56", 1855.56),  # formato brasileiro
        ("1855,56", 1855.56),
        ("10.00", 10.0),
    ],
)
def test_para_float_formatos(bruto, esperado):
    assert _para_float(bruto) == pytest.approx(esperado)


@pytest.mark.parametrize("vazio", ["-", "", "nan", None])
def test_para_float_ausente_vira_nan(vazio):
    assert pd.isna(_para_float(vazio))


def test_ticker_extraido_do_produto(movimentacoes):
    assert set(movimentacoes["ticker"]) == {"AAAA3", "BBBB4"}


def test_mesmo_ticker_com_razao_social_diferente(movimentacoes):
    """'S.A.', 'S.A' e 'S/A' convivem no mesmo arquivo para o mesmo papel."""
    alfa = movimentacoes[movimentacoes["ticker"] == "AAAA3"]
    assert alfa["produto"].nunique() > 1


def test_classificacao(movimentacoes):
    por_classe = movimentacoes.groupby("classe")["movimentacao"].apply(set)

    assert "Transferência - Liquidação" in por_classe["quantidade"]
    assert "Bonificação em Ativos" in por_classe["quantidade"]
    assert "Fração em Ativos" in por_classe["quantidade"]
    assert "Dividendo" in por_classe["caixa"]
    assert "Leilão de Fração" in por_classe["caixa"]
    assert por_classe["snapshot"] == {"Atualização"}


def test_movimentacao_nova_falha_em_vez_de_passar_batido():
    """A B3 passar a registrar um evento novo não pode corromper em silêncio."""
    bruto = pd.DataFrame(
        {
            "Entrada/Saída": ["Credito"],
            "Data": ["01/01/2024"],
            "Movimentação": ["Incorporação de Ações"],
            "Produto": ["AAAA3 - EMPRESA ALFA S.A."],
            "Instituição": ["CORRETORA EXEMPLO"],
            "Quantidade": ["10"],
            "Preço unitário": ["-"],
            "Valor da Operação": ["-"],
        }
    )
    with pytest.raises(MovimentacaoDesconhecida, match="Incorporação"):
        normalizar(bruto)


def test_entrada_saida_invalida_falha():
    bruto = pd.DataFrame(
        {
            "Entrada/Saída": ["Talvez"],
            "Data": ["01/01/2024"],
            "Movimentação": ["Dividendo"],
            "Produto": ["AAAA3 - EMPRESA ALFA S.A."],
            "Instituição": ["CORRETORA EXEMPLO"],
            "Quantidade": ["10"],
            "Preço unitário": ["1.00"],
            "Valor da Operação": ["10.00"],
        }
    )
    with pytest.raises(ValueError, match="Entrada/Saída"):
        normalizar(bruto)


# --------------------------------------------------------------------------
# posicao
# --------------------------------------------------------------------------


def test_posicao_derivada(ledger):
    """AAAA3: 100 comprada + 10 bonificada - 30 vendida - 0,5 de fração."""
    posicao = posicao_em(ledger)

    assert posicao["AAAA3"] == pytest.approx(79.5)
    assert posicao["BBBB4"] == pytest.approx(50.0)


def test_caixa_e_snapshot_nao_viram_quantidade(movimentacoes, ledger):
    """O erro clássico: somar 'Atualização' ou 'Dividendo' como se fosse compra."""
    assert set(ledger["ticker"]) == {"AAAA3", "BBBB4"}
    assert len(ledger) == len(movimentacoes[movimentacoes["classe"] == "quantidade"])


def test_posicao_em_data_passada(ledger):
    """Antes da bonificação de abril, AAAA3 tinha só as 100 compradas."""
    assert posicao_em(ledger, "2024-03-31")["AAAA3"] == pytest.approx(100.0)


def test_atualizacoes_conferem(movimentacoes, ledger):
    """Toda divergência aqui é evento não modelado — é a rede de segurança."""
    conferencia = conferir_atualizacoes(movimentacoes, ledger)

    assert len(conferencia) == 2
    assert conferencia["ok"].all()


def test_ajuste_quebra_a_conferencia(movimentacoes, ledger):
    """Confirma que a conferência realmente detecta divergência."""
    extra = pd.DataFrame(
        [{"data": pd.Timestamp("2024-02-01"), "ticker": "AAAA3",
          "delta": 5.0, "nota": "ajuste inventado"}]
    )
    conferencia = conferir_atualizacoes(
        movimentacoes, construir_ledger(movimentacoes, extra)
    )

    divergentes = conferencia[~conferencia["ok"]]
    assert list(divergentes["ticker"]) == ["AAAA3"]
    assert divergentes["dif"].iloc[0] == pytest.approx(-5.0)
