"""Testes da série do CDI.

Sem rede: a chamada ao SGS é substituída. O que se testa aqui é a conversão
de escala, a composição multiplicativa e o cache — não a API do Banco
Central.
"""

import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

from carteira import cdi as mod_cdi
from carteira.cdi import CDIIndisponivel, alinhar, cdi_diario, cdi_mensal, compor_mensal


@pytest.fixture
def sgs_falso(monkeypatch):
    """Devolve 0,05% ao dia em todo dia útil do período pedido."""
    chamadas = []

    def _falso(inicio, fim):
        chamadas.append((inicio, fim))
        return [
            {"data": dia.strftime("%d/%m/%Y"), "valor": "0.05"}
            for dia in pd.bdate_range(inicio, fim)
        ]

    monkeypatch.setattr(mod_cdi, "_buscar", _falso)
    return chamadas


# --------------------------------------------------------------------------
# escala
# --------------------------------------------------------------------------


def test_percentual_vira_fracao(sgs_falso):
    """O SGS devolve 0.05 querendo dizer 0,05% ao dia."""
    serie = cdi_diario("2024-01-01", "2024-01-31")

    assert serie.iloc[0] == pytest.approx(0.0005)


def test_serie_vazia_falha(monkeypatch):
    monkeypatch.setattr(mod_cdi, "_buscar", lambda i, f: [])

    with pytest.raises(CDIIndisponivel, match="vazia"):
        cdi_diario("2024-01-01", "2024-01-31")


def test_rede_fora_vira_erro_explicativo(monkeypatch):
    def _cai(*_, **__):
        raise urllib.error.URLError("sem rede")

    monkeypatch.setattr(mod_cdi.urllib.request, "urlopen", _cai)

    with pytest.raises(CDIIndisponivel, match="Banco Central"):
        cdi_diario("2024-01-01", "2024-01-31")


def test_json_invalido_vira_erro_explicativo(monkeypatch):
    class _Resposta:
        def read(self):
            return b"nao e json"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(mod_cdi.urllib.request, "urlopen", lambda *a, **k: _Resposta())

    with pytest.raises(CDIIndisponivel):
        cdi_diario("2024-01-01", "2024-01-31")


# --------------------------------------------------------------------------
# composição
# --------------------------------------------------------------------------


def test_composicao_e_multiplicativa():
    """21 dias a 0,05% dão mais que 21 x 0,05%."""
    dias = pd.bdate_range("2024-01-01", "2024-01-31")
    diario = pd.Series(0.0005, index=dias)

    mensal = compor_mensal(diario)
    n = len(dias)

    assert mensal.iloc[0] == pytest.approx(1.0005**n - 1)
    assert mensal.iloc[0] > n * 0.0005


def test_composicao_separa_os_meses():
    dias = pd.bdate_range("2024-01-01", "2024-03-31")
    mensal = compor_mensal(pd.Series(0.0005, index=dias))

    assert len(mensal) == 3
    assert list(mensal.index.month) == [1, 2, 3]


def test_cdi_mensal_encadeia_as_duas_etapas(sgs_falso):
    mensal = cdi_mensal("2024-01-01", "2024-02-29")

    assert len(mensal) == 2
    assert (mensal > 0).all()


def test_serie_diaria_vazia_nao_quebra():
    assert compor_mensal(pd.Series(dtype=float)).empty


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def test_cache_evita_segunda_busca(tmp_path, sgs_falso):
    cache = tmp_path / "cdi.parquet"
    hoje = pd.Timestamp.today().normalize()
    inicio = hoje - pd.Timedelta(days=60)

    cdi_diario(inicio, hoje, cache=cache)
    cdi_diario(inicio, hoje, cache=cache)

    assert len(sgs_falso) == 1


def test_cache_desatualizado_busca_de_novo(tmp_path, sgs_falso):
    """Cache que para muito antes do fim pedido não serve."""
    cache = tmp_path / "cdi.parquet"
    hoje = pd.Timestamp.today().normalize()

    cdi_diario(hoje - pd.Timedelta(days=400), hoje - pd.Timedelta(days=300),
               cache=cache)
    cdi_diario(hoje - pd.Timedelta(days=400), hoje, cache=cache)

    assert len(sgs_falso) == 2


# --------------------------------------------------------------------------
# alinhamento
# --------------------------------------------------------------------------


def test_alinhar_casa_com_o_indice_dos_retornos():
    indice = pd.date_range("2024-01-31", periods=3, freq="ME")
    cdi = pd.Series([0.01, 0.011, 0.0105], index=indice)

    assert alinhar(cdi, indice).tolist() == pytest.approx([0.01, 0.011, 0.0105])


def test_mes_faltando_falha_em_vez_de_virar_zero():
    """Preencher com zero inflaria o excesso de retorno silenciosamente."""
    indice = pd.date_range("2024-01-31", periods=3, freq="ME")
    incompleto = pd.Series([0.01], index=indice[:1])

    with pytest.raises(CDIIndisponivel, match="Sem CDI"):
        alinhar(incompleto, indice)
