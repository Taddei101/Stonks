"""Testes do leitor do relatório de Posição da B3.

O xlsx é montado em tempo de teste com o layout real (nomes de aba,
cabeçalhos e rodapé de 3 linhas copiados de exports de 06/11/2026);
tickers e números são inventados.
"""

import openpyxl
import pandas as pd
import pytest

from carteira.io_posicao import LayoutInesperado, ler_posicao

CAB_ACOES = [
    "Produto", "Instituição", "Conta", "Código de Negociação", "CNPJ da Empresa",
    "Código ISIN / Distribuição", "Tipo", "Escriturador", "Quantidade",
    "Quantidade Disponível", "Quantidade Indisponível", "Motivo",
    "Preço de Fechamento", "Valor Atualizado",
]
CAB_ETF = [
    "Produto", "Instituição", "Conta", "Código de Negociação", "CNPJ do Fundo",
    "Código ISIN / Distribuição", "Tipo", "Quantidade", "Quantidade Disponível",
    "Quantidade Indisponível", "Motivo", "Preço de Fechamento", "Valor Atualizado",
]
CAB_FUNDO = [
    "Produto", "Instituição", "Conta", "Código de Negociação", "CNPJ do Fundo",
    "Código ISIN / Distribuição", "Tipo", "Administrador", "Quantidade",
    "Quantidade Disponível", "Quantidade Indisponível", "Motivo",
    "Preço de Fechamento", "Valor Atualizado",
]
CAB_RF = [
    "Produto", "Instituição", "Emissor", "Código", "Indexador", "Tipo de regime",
    "Data de Emissão", "Vencimento", "Quantidade", "Quantidade Disponível",
    "Quantidade Indisponível", "Motivo", "Contraparte", "Preço Atualizado MTM",
    "Valor Atualizado MTM", "Preço Atualizado CURVA", "Valor Atualizado CURVA",
    "Preço Atualizado FECHAMENTO", "Valor Atualizado FECHAMENTO",
]


def _linha(cab, **campos):
    """Linha com '-' onde a B3 põe '-' e texto genérico no resto."""
    padrao = {"Quantidade Indisponível": "-", "Motivo": "-"}
    return [campos.get(c, padrao.get(c, "x")) for c in cab]


def _ativo(cab, ticker, corretora, qtd, preco, tipo="x"):
    return _linha(
        cab,
        **{
            "Produto": f"{ticker} - EMPRESA FICTICIA",
            "Instituição": corretora,
            "Código de Negociação": ticker,
            "Tipo": tipo,
            "Quantidade": qtd,
            "Quantidade Disponível": qtd,
            "Preço de Fechamento": preco,
            "Valor Atualizado": qtd * preco,
        },
    )


def _escrever(caminho, abas):
    """abas: {nome: (cabeçalho, linhas)}. Acrescenta o rodapé da B3."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for nome, (cab, linhas) in abas.items():
        ws = wb.create_sheet(nome)
        ws.append(cab)
        for linha in linhas:
            ws.append(linha)
        ws.append([""])
        ws.append([""] * (len(cab) - 1) + ["Total"])
        ws.append([""] * (len(cab) - 1) + [sum(l[cab.index("Valor Atualizado")] for l in linhas) if "Valor Atualizado" in cab else 0])
    wb.save(caminho)
    return caminho


@pytest.fixture
def posicao_completa(tmp_path):
    return _escrever(
        tmp_path / "posicao.xlsx",
        {
            "Acoes": (CAB_ACOES, [
                _ativo(CAB_ACOES, "AAAA3", "CORRETORA A", 100, 10.0, "ON"),
                _ativo(CAB_ACOES, "AAAA3", "CORRETORA B", 50, 10.0, "ON"),
                _ativo(CAB_ACOES, "BBBB4", "CORRETORA A", 20, 30.0, "PN"),
            ]),
            "ETF": (CAB_ETF, [_ativo(CAB_ETF, "EEEE11", "CORRETORA A", 10, 100.0, "Internacional")]),
            "Fundo de Investimento": (CAB_FUNDO, [
                _ativo(CAB_FUNDO, "FFFF11", "CORRETORA A", 30, 90.0, "Cotas"),
                _ativo(CAB_FUNDO, "FFFF12", "CORRETORA A", 3, 1.5, "Direito"),
            ]),
            "Renda Fixa": (CAB_RF, [_linha(CAB_RF, **{"Valor Atualizado CURVA": 1000.0})]),
        },
    )


def test_soma_mesmo_ticker_em_corretoras_diferentes(posicao_completa):
    ativos = ler_posicao(posicao_completa).ativos.set_index("ticker")
    assert ativos.loc["AAAA3", "quantidade"] == 150
    assert ativos.loc["AAAA3", "valor"] == pytest.approx(1500.0)
    assert ativos.loc["AAAA3", "preco"] == pytest.approx(10.0)


def test_rodape_nao_vira_ativo(posicao_completa):
    assert sorted(ler_posicao(posicao_completa).ativos["ticker"]) == [
        "AAAA3", "BBBB4", "EEEE11", "FFFF11",
    ]


def test_classe_vem_da_aba(posicao_completa):
    classes = ler_posicao(posicao_completa).ativos.set_index("ticker")["classe"]
    assert classes.to_dict() == {
        "AAAA3": "Ação", "BBBB4": "Ação", "EEEE11": "ETF", "FFFF11": "FII",
    }


def test_direito_de_subscricao_fica_a_parte(posicao_completa):
    pos = ler_posicao(posicao_completa)
    assert list(pos.direitos["ticker"]) == ["FFFF12"]
    assert "FFFF12" not in set(pos.ativos["ticker"])


def test_renda_fixa_e_listada_como_ignorada(posicao_completa):
    assert ler_posicao(posicao_completa).abas_ignoradas == ["Renda Fixa"]


def test_so_renda_fixa_devolve_ativos_vazio(tmp_path):
    arq = _escrever(tmp_path / "rf.xlsx", {"Renda Fixa": (CAB_RF, [_linha(CAB_RF)])})
    pos = ler_posicao(arq)
    assert pos.ativos.empty
    assert list(pos.ativos.columns) == ["ticker", "classe", "quantidade", "preco", "valor"]
    assert pos.abas_ignoradas == ["Renda Fixa"]


def test_aceita_arquivo_aberto(posicao_completa):
    with open(posicao_completa, "rb") as f:
        assert len(ler_posicao(f).ativos) == 4


def test_layout_inesperado_falha_alto(tmp_path):
    cab = [c for c in CAB_ACOES if c != "Valor Atualizado"]
    arq = _escrever(tmp_path / "x.xlsx", {"Acoes": (cab, [_linha(cab, **{"Código de Negociação": "AAAA3"})])})
    with pytest.raises(LayoutInesperado, match="Valor Atualizado"):
        ler_posicao(arq)
