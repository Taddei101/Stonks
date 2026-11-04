"""Testes da decomposição em componentes principais.

Os casos extremos têm resposta analítica conhecida, e é por eles que o teste
começa: ativos independentes dão N apostas, ativos idênticos dão 1. Se a
implementação errar escala, normalização ou ordem, um dos dois quebra.
"""

import numpy as np
import pandas as pd
import pytest

from carteira.componentes import (
    apostas_efetivas,
    cargas,
    componentes_para_explicar,
    decomposicao,
    pares_mais_correlacionados,
)


def _correlacao(matriz, nomes):
    return pd.DataFrame(matriz, index=nomes, columns=nomes)


def identidade(n=5):
    """Ativos perfeitamente independentes."""
    return _correlacao(np.eye(n), [f"A{i}" for i in range(n)])


def tudo_junto(n=5, rho=0.999):
    """Ativos que se movem praticamente como um só."""
    m = np.full((n, n), rho)
    np.fill_diagonal(m, 1.0)
    return _correlacao(m, [f"A{i}" for i in range(n)])


def dois_blocos(rho_dentro=0.9, rho_fora=0.1):
    """Dois grupos coesos e quase independentes entre si — o caso de interesse."""
    nomes = ["X1", "X2", "X3", "Y1", "Y2", "Y3"]
    m = np.full((6, 6), rho_fora)
    m[:3, :3] = rho_dentro
    m[3:, 3:] = rho_dentro
    np.fill_diagonal(m, 1.0)
    return _correlacao(m, nomes)


# --------------------------------------------------------------------------
# apostas efetivas: os extremos têm resposta conhecida
# --------------------------------------------------------------------------


def test_ativos_independentes_dao_n_apostas():
    assert apostas_efetivas(identidade(5)) == pytest.approx(5.0)
    assert apostas_efetivas(identidade(7)) == pytest.approx(7.0)


def test_ativos_identicos_dao_uma_aposta():
    assert apostas_efetivas(tudo_junto(5)) == pytest.approx(1.0, abs=0.02)


def test_dois_blocos_dao_cerca_de_duas_apostas():
    """Seis papéis, dois grupos coesos: a carteira se comporta como ~2."""
    efetivas = apostas_efetivas(dois_blocos())

    assert 1.5 < efetivas < 3.0


def test_mais_correlacao_significa_menos_apostas():
    """Monotonicidade: é o teste que pega erro de sinal na entropia."""
    anterior = None
    for rho in (0.0, 0.3, 0.6, 0.9):
        m = np.full((5, 5), rho)
        np.fill_diagonal(m, 1.0)
        atual = apostas_efetivas(_correlacao(m, [f"A{i}" for i in range(5)]))
        if anterior is not None:
            assert atual < anterior
        anterior = atual


def test_apostas_efetivas_nunca_passa_do_numero_de_ativos():
    for n in (2, 4, 7):
        assert apostas_efetivas(identidade(n)) <= n + 1e-9


# --------------------------------------------------------------------------
# decomposição
# --------------------------------------------------------------------------


def test_autovalores_somam_o_numero_de_ativos():
    """Propriedade da matriz de correlação: traço igual a N."""
    tabela = decomposicao(dois_blocos())

    assert tabela["autovalor"].sum() == pytest.approx(6.0)


def test_autovalores_vem_em_ordem_decrescente():
    autovalores = decomposicao(dois_blocos())["autovalor"].to_numpy()

    assert np.all(np.diff(autovalores) <= 1e-12)


def test_acumulada_termina_em_um():
    assert decomposicao(dois_blocos())["acumulada"].iloc[-1] == pytest.approx(1.0)


def test_independentes_precisam_de_todos_os_componentes():
    """Sem estrutura comum não há o que resumir."""
    assert componentes_para_explicar(identidade(5), alvo=0.90) == 5


def test_correlacionados_precisam_de_um_componente_so():
    assert componentes_para_explicar(tudo_junto(5), alvo=0.90) == 1


def test_alvo_invalido():
    with pytest.raises(ValueError, match="alvo"):
        componentes_para_explicar(identidade(3), alvo=1.5)


# --------------------------------------------------------------------------
# cargas
# --------------------------------------------------------------------------


def test_primeiro_componente_pega_todo_mundo_junto():
    """Fator de mercado: cargas de mesmo sinal e magnitude parecida."""
    cp1 = cargas(tudo_junto(5), quantos=1)["CP1"]

    assert (cp1 > 0).all()
    assert cp1.std() < 0.05


def test_segundo_componente_separa_os_blocos():
    """É onde se descobre o eixo que divide a carteira."""
    cp2 = cargas(dois_blocos(), quantos=2)["CP2"]

    assert np.sign(cp2[["X1", "X2", "X3"]]).nunique() == 1
    assert np.sign(cp2[["Y1", "Y2", "Y3"]]).nunique() == 1
    assert np.sign(cp2["X1"]) != np.sign(cp2["Y1"])


def test_orientacao_das_cargas_e_estavel():
    """O sinal do autovetor é arbitrário; a convenção o torna reprodutível."""
    correlacao = dois_blocos()

    assert cargas(correlacao).equals(cargas(correlacao))
    for coluna in cargas(correlacao).items():
        serie = coluna[1]
        assert serie[serie.abs().idxmax()] > 0


def test_cargas_sao_vetores_unitarios():
    norma = np.linalg.norm(cargas(dois_blocos(), quantos=3).to_numpy(), axis=0)

    assert np.allclose(norma, 1.0)


def test_pede_mais_componentes_do_que_existe():
    assert cargas(identidade(3), quantos=10).shape[1] == 3


# --------------------------------------------------------------------------
# pares
# --------------------------------------------------------------------------


def test_pares_acima_do_limite():
    pares = pares_mais_correlacionados(dois_blocos(), limite=0.7)

    assert len(pares) == 6  # 3 pares dentro de cada bloco
    assert set(pares["a"]) <= {"X1", "X2", "Y1", "Y2"}


def test_pares_vem_ordenados():
    nomes = ["A", "B", "C"]
    m = np.array([[1.0, 0.95, 0.75], [0.95, 1.0, 0.72], [0.75, 0.72, 1.0]])

    pares = pares_mais_correlacionados(_correlacao(m, nomes), limite=0.7)

    assert pares["correlacao"].tolist() == pytest.approx([0.95, 0.75, 0.72])


def test_sem_par_acima_do_limite_devolve_tabela_vazia():
    pares = pares_mais_correlacionados(identidade(4), limite=0.7)

    assert pares.empty
    assert list(pares.columns) == ["a", "b", "correlacao"]
