"""Testes do etl.py.

NOTA: este script faz o mesmo trabalho de `carteira/io_extrato.py`, de forma
mais simples e sem deduplicação. O pipeline de verdade usa o módulo, não
este arquivo — ele sobrevive como ferramenta de inspeção rápida. Se um dia
deixar de ter uso, remova os dois juntos.

Os testes antigos chamavam `get_folder()` sem argumento e dependiam da pasta
`extratos/`, que está no .gitignore. Num clone limpo os dois falhavam. Agora
as funções aceitam a pasta como parâmetro e os testes usam o extrato de
exemplo.
"""

import pytest

from etl import VALID_HEADERS, data_frame_pattern, get_folder


def test_encontra_planilhas(pasta_extratos):
    assert len(get_folder(pasta_extratos)) == 1


def test_pasta_vazia_devolve_lista_vazia(tmp_path):
    assert get_folder(tmp_path) == []


def test_dataframe_tem_o_layout_da_b3(pasta_extratos):
    df = data_frame_pattern(pasta_extratos)

    assert not df.empty
    assert df.columns.tolist() == VALID_HEADERS


def test_pasta_sem_planilha_encerra(tmp_path):
    """Sem arquivo não há o que validar — encerra em vez de estourar IndexError."""
    with pytest.raises(SystemExit):
        data_frame_pattern(tmp_path)
