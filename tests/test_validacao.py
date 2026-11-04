"""Testes da validação deslizante.

O teste que importa mais é o de vazamento: provar que nenhuma regra enxerga
dado posterior ao mês que está sendo avaliado. Vazamento de futuro não
levanta exceção — produz resultado excelente e falso, e é o erro mais comum
em backtest.
"""

import numpy as np
import pandas as pd
import pytest

from carteira.validacao import (
    JanelaInsuficiente,
    deslizante,
    formatar,
    resumo,
)

ATIVOS = ["A", "B", "C"]


@pytest.fixture
def retornos():
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        rng.normal(0.01, 0.05, (60, 3)),
        columns=ATIVOS,
        index=pd.date_range("2020-01-31", periods=60, freq="ME"),
    )


def iguais(dados: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / dados.shape[1], index=dados.columns)


def so_o_primeiro(dados: pd.DataFrame) -> pd.Series:
    pesos = pd.Series(0.0, index=dados.columns)
    pesos.iloc[0] = 1.0
    return pesos


# --------------------------------------------------------------------------
# vazamento de futuro
# --------------------------------------------------------------------------


def test_regra_nunca_ve_o_mes_avaliado(retornos):
    """O teste central. Cada regra registra a última data que recebeu."""
    vistos = []

    def espia(dados):
        vistos.append(dados.index.max())
        return iguais(dados)

    resultado = deslizante(retornos, {"espia": espia}, treino=24)

    avaliados = resultado["retornos"].index
    assert len(vistos) == len(avaliados)
    for visto, avaliado in zip(vistos, avaliados):
        assert visto < avaliado, "a regra viu o mês que seria avaliado"


def test_janela_rolante_tem_tamanho_fixo(retornos):
    tamanhos = []

    def espia(dados):
        tamanhos.append(len(dados))
        return iguais(dados)

    deslizante(retornos, {"espia": espia}, treino=24, modo="rolante")

    assert set(tamanhos) == {24}


def test_janela_ancorada_cresce(retornos):
    tamanhos = []

    def espia(dados):
        tamanhos.append(len(dados))
        return iguais(dados)

    deslizante(retornos, {"espia": espia}, treino=24, modo="ancorada")

    assert tamanhos == sorted(tamanhos)
    assert tamanhos[0] == 24
    assert tamanhos[-1] == len(retornos) - 1


def test_modo_invalido(retornos):
    with pytest.raises(ValueError, match="rolante"):
        deslizante(retornos, {"iguais": iguais}, treino=24, modo="torta")


# --------------------------------------------------------------------------
# contabilidade
# --------------------------------------------------------------------------


def test_retorno_fora_da_amostra_e_o_peso_vezes_o_mes_seguinte(retornos):
    resultado = deslizante(retornos, {"iguais": iguais}, treino=24)
    serie = resultado["retornos"]["iguais"]

    esperado = retornos.iloc[24:].mean(axis=1)

    assert serie.to_numpy() == pytest.approx(esperado.to_numpy())


def test_quantidade_avaliada_e_o_que_sobra_do_treino(retornos):
    resultado = deslizante(retornos, {"iguais": iguais}, treino=24)

    assert len(resultado["retornos"]) == len(retornos) - 24
    assert resultado["descartados"] == 24


def test_dados_de_menos_falha_com_explicacao(retornos):
    with pytest.raises(JanelaInsuficiente, match="não sobra"):
        deslizante(retornos, {"iguais": iguais}, treino=60)


def test_pesos_sempre_somam_um(retornos):
    def desnormalizada(dados):
        return pd.Series(7.0, index=dados.columns)

    resultado = deslizante(retornos, {"x": desnormalizada}, treino=24)

    assert resultado["pesos"]["x"].sum(axis=1).to_numpy() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# giro
# --------------------------------------------------------------------------


def test_regra_sem_parametro_nao_gira(retornos):
    """1/N nunca muda, então não custa nada em corretagem."""
    resultado = deslizante(retornos, {"iguais": iguais}, treino=24)

    assert resultado["giro"]["iguais"].sum() == pytest.approx(0.0)


def test_regra_que_alterna_gira_muito(retornos):
    """Giro é a soma das mudanças de peso, então trocar tudo dá 2,0."""
    estado = {"n": 0}

    def alterna(dados):
        estado["n"] += 1
        pesos = pd.Series(0.0, index=dados.columns)
        pesos.iloc[estado["n"] % 2] = 1.0
        return pesos

    resultado = deslizante(retornos, {"alterna": alterna}, treino=24)
    giro = resultado["giro"]["alterna"]

    assert giro.iloc[1:].to_numpy() == pytest.approx(2.0)


# --------------------------------------------------------------------------
# regra que falha
# --------------------------------------------------------------------------


def test_regra_que_quebra_mantem_a_carteira_anterior(retornos):
    """Não convergir num mês não pode derrubar a validação inteira."""
    estado = {"n": 0}

    def instavel(dados):
        estado["n"] += 1
        if estado["n"] % 3 == 0:
            raise RuntimeError("não convergiu")
        return so_o_primeiro(dados)

    resultado = deslizante(retornos, {"instavel": instavel}, treino=24)

    assert resultado["falhas"]["instavel"] > 0
    assert resultado["retornos"]["instavel"].notna().all()
    assert resultado["pesos"]["instavel"].sum(axis=1).to_numpy() == pytest.approx(1.0)


def test_regra_que_quebra_logo_no_inicio_usa_pesos_iguais(retornos):
    def sempre_quebra(dados):
        raise RuntimeError("nunca funciona")

    resultado = deslizante(retornos, {"ruim": sempre_quebra}, treino=24)
    pesos = resultado["pesos"]["ruim"]

    assert resultado["falhas"]["ruim"] == len(resultado["retornos"])
    assert pesos.iloc[0].to_numpy() == pytest.approx([1 / 3] * 3)


# --------------------------------------------------------------------------
# relatório
# --------------------------------------------------------------------------


def test_resumo_traz_giro_e_falhas(retornos):
    resultado = deslizante(
        retornos, {"iguais": iguais, "primeiro": so_o_primeiro}, treino=24
    )
    tabela = resumo(resultado)

    assert "giro_anual" in tabela.columns
    assert tabela.loc["iguais", "giro_anual"] == pytest.approx(0.0)
    assert tabela.loc["iguais", "falhas"] == 0


def test_formatar_avisa_sobre_os_meses_descartados(retornos):
    resultado = deslizante(retornos, {"iguais": iguais}, treino=24)
    texto = formatar(resultado, resumo(resultado))

    assert "VALIDAÇÃO DESLIZANTE" in texto
    assert "não são avaliados" in texto
    assert "GIRO" in texto


def test_formatar_lista_as_regras_que_falharam(retornos):
    def sempre_quebra(dados):
        raise RuntimeError("nunca")

    resultado = deslizante(
        retornos, {"iguais": iguais, "ruim": sempre_quebra}, treino=24
    )
    texto = formatar(resultado, resumo(resultado))

    assert "FALHARAM" in texto


# --------------------------------------------------------------------------
# o resultado que motiva tudo
# --------------------------------------------------------------------------


def test_otimizar_dentro_da_amostra_nao_garante_nada_fora():
    """Regra que escolhe o melhor ativo passado não vence o 1/N no futuro.

    Com retornos independentes entre meses, o vencedor passado não carrega
    informação — e a regra que persegue vencedor passado só adiciona giro.
    """
    rng = np.random.default_rng(42)
    retornos = pd.DataFrame(
        rng.normal(0.01, 0.06, (120, 4)),
        columns=list("ABCD"),
        index=pd.date_range("2016-01-31", periods=120, freq="ME"),
    )

    def persegue_vencedor(dados):
        pesos = pd.Series(0.0, index=dados.columns)
        pesos[dados.mean().idxmax()] = 1.0
        return pesos

    resultado = deslizante(
        retornos, {"iguais": iguais, "vencedor": persegue_vencedor}, treino=36
    )
    tabela = resumo(resultado)

    assert tabela.loc["vencedor", "giro_anual"] > 0.1
    assert (
        tabela.loc["vencedor", "volatilidade_anual"]
        > tabela.loc["iguais", "volatilidade_anual"]
    )
