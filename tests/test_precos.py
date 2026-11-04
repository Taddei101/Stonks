"""Testes do cache de preços.

Sem rede: `_baixar` é substituído por uma função que devolve série sintética
e conta chamadas. O que se testa aqui é a política de cache, não o yfinance.

O caso importante é a separação entre série de mercado e série ajustada. As
duas são numericamente parecidas e indistinguíveis a olho no parquet — se o
cache as misturar, a análise de risco recebe preço de mercado sem erro
nenhum, e o resultado fica errado de um jeito que não dá para perceber.
"""

import pandas as pd
import pytest

from carteira import precos as mod_precos

TICKERS = ["AAAA3", "BBBB4"]
INICIO = "2024-01-01"
FIM = "2024-01-10"


@pytest.fixture
def baixar_falso(monkeypatch):
    """Substitui o download. Série ajustada vem com metade do valor.

    A diferença é exagerada de propósito: qualquer vazamento entre as duas
    séries aparece como fator 2, não como arredondamento.
    """
    chamadas = []

    def _falso(simbolos, ajustado=False, **kwargs):
        chamadas.append({"simbolos": list(simbolos), "ajustado": ajustado})
        datas = pd.date_range(INICIO, FIM, freq="B")
        base = 100.0 if not ajustado else 50.0
        return pd.DataFrame(
            {mod_precos._sem_sufixo(s): base for s in simbolos}, index=datas
        )

    monkeypatch.setattr(mod_precos, "_baixar", _falso)
    return chamadas


def test_cache_evita_segundo_download(tmp_path, baixar_falso):
    cache = tmp_path / "precos.parquet"

    mod_precos.historico(TICKERS, INICIO, FIM, cache=cache)
    mod_precos.historico(TICKERS, INICIO, FIM, cache=cache)

    assert len(baixar_falso) == 1


def test_serie_ajustada_nao_vem_do_cache_da_serie_de_mercado(tmp_path, baixar_falso):
    """O caso que a coluna `ajustado` existe para impedir."""
    cache = tmp_path / "precos.parquet"

    mercado = mod_precos.historico(TICKERS, INICIO, FIM, cache=cache)
    total = mod_precos.historico_total(TICKERS, INICIO, FIM, cache=cache)

    assert mercado["AAAA3"].iloc[0] == pytest.approx(100.0)
    assert total["AAAA3"].iloc[0] == pytest.approx(50.0)
    assert len(baixar_falso) == 2
    assert [c["ajustado"] for c in baixar_falso] == [False, True]


def test_as_duas_series_convivem_no_mesmo_cache(tmp_path, baixar_falso):
    cache = tmp_path / "precos.parquet"

    mod_precos.historico(TICKERS, INICIO, FIM, cache=cache)
    mod_precos.historico_total(TICKERS, INICIO, FIM, cache=cache)
    baixar_falso.clear()

    mercado = mod_precos.historico(TICKERS, INICIO, FIM, cache=cache)
    total = mod_precos.historico_total(TICKERS, INICIO, FIM, cache=cache)

    assert not baixar_falso, "as duas deveriam vir do cache"
    assert mercado["AAAA3"].iloc[0] == pytest.approx(100.0)
    assert total["AAAA3"].iloc[0] == pytest.approx(50.0)


def test_cache_antigo_sem_a_coluna_e_lido_como_serie_de_mercado(tmp_path, baixar_falso):
    """Compatibilidade: parquet gravado antes desta mudança não tem a coluna.

    Ele contém série de mercado, porque era a única que existia. Tratá-lo
    como tal evita invalidar o cache de quem já usa o projeto.
    """
    cache = tmp_path / "precos.parquet"
    datas = pd.date_range(INICIO, FIM, freq="B")
    antigo = pd.DataFrame(
        [
            {"data": data, "ticker": ticker, "close": 100.0}
            for data in datas
            for ticker in TICKERS
        ]
    )
    antigo.to_parquet(cache, index=False)

    mercado = mod_precos.historico(TICKERS, INICIO, FIM, cache=cache)

    assert not baixar_falso, "deveria ter usado o cache antigo"
    assert mercado["AAAA3"].iloc[0] == pytest.approx(100.0)


def test_historico_total_pede_serie_ajustada(tmp_path, baixar_falso):
    mod_precos.historico_total(TICKERS, INICIO, FIM, cache=tmp_path / "t.parquet")

    assert baixar_falso[0]["ajustado"] is True


def test_sem_tickers_nao_baixa_nada(baixar_falso):
    assert mod_precos.historico([], INICIO).empty
    assert not baixar_falso


def test_cache_com_ticker_de_historico_curto_nao_e_aceito(tmp_path, baixar_falso):
    """O bug que truncou 7 anos de histórico em silêncio.

    Com um ticker no cache desde 2016 e outro só desde 2024, o mínimo
    GLOBAL das datas dava 2016 e a verificação passava. O cache devolvia o
    segundo ticker vazio no começo, e quem consumisse a série cortaria o
    período comum sem erro nenhum.
    """
    cache = tmp_path / "precos.parquet"
    antigo = pd.DataFrame(
        [
            {"data": d, "ticker": "VELHO", "close": 100.0, "ajustado": False}
            for d in pd.bdate_range("2016-01-01", "2026-01-01")
        ]
        + [
            {"data": d, "ticker": "NOVO", "close": 50.0, "ajustado": False}
            for d in pd.bdate_range("2024-01-01", "2026-01-01")
        ]
    )
    antigo.to_parquet(cache, index=False)

    mod_precos.historico(["VELHO", "NOVO"], "2016-01-01", "2026-01-01", cache=cache)

    assert len(baixar_falso) == 1, "deveria ter rebaixado em vez de usar o cache"


def test_cache_com_todos_os_tickers_cobrindo_e_aceito(tmp_path, baixar_falso):
    cache = tmp_path / "precos.parquet"
    pd.DataFrame(
        [
            {"data": d, "ticker": t, "close": 100.0, "ajustado": False}
            for d in pd.bdate_range("2016-01-01", "2026-01-01")
            for t in ("A", "B")
        ]
    ).to_parquet(cache, index=False)

    mod_precos.historico(["A", "B"], "2016-01-01", "2026-01-01", cache=cache)

    assert not baixar_falso
