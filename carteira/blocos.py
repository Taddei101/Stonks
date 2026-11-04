"""Agrupamento de ativos por fator de risco.

Diversificação se mede por fator compartilhado, não por contagem de papéis.
Sete ações em três setores com um risco político comum a vários deles não
são sete apostas — e a matriz de correlação de uma janela curta não revela
isso, porque o fator fica quieto por anos e se manifesta de uma vez.

Este módulo torna o agrupamento explícito e versionado, para que a análise
possa somar risco por bloco e não só por ativo.

Um ativo pertence a várias dimensões ao mesmo tempo (setor, controle, moeda,
origem da receita), e é essa sobreposição que interessa: ITUB4 e BBAS3
dividem o setor; BBAS3 e PETR4 dividem o controle; VALE3 e PETR4 dividem a
exposição a dólar pela receita exportadora.
"""

import json
from pathlib import Path

import pandas as pd


class BlocosInvalidos(ValueError):
    """O arquivo de blocos não cobre todos os ativos pedidos."""


def carregar_blocos(caminho: Path) -> dict[str, pd.Series]:
    """Lê o mapeamento e devolve {dimensão: Series(ticker -> categoria)}.

    Chaves começando com `_` são comentário e saem da leitura — JSON não tem
    sintaxe de comentário, e a alternativa seria manter a explicação fora do
    arquivo que ela explica.
    """
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))

    dimensoes = {}
    for dimensao, mapa in dados.items():
        if dimensao.startswith("_"):
            continue
        if not isinstance(mapa, dict):
            raise BlocosInvalidos(
                f"Dimensão {dimensao!r} deveria ser um objeto ticker -> categoria."
            )
        dimensoes[dimensao] = pd.Series(mapa, dtype="object").sort_index()

    if not dimensoes:
        raise BlocosInvalidos(f"Nenhuma dimensão em {caminho}.")

    return dimensoes


def validar_cobertura(
    dimensoes: dict[str, pd.Series], tickers: list[str]
) -> None:
    """Falha se algum ticker não estiver mapeado em alguma dimensão.

    Falhar é melhor que assumir: ticker sem bloco viraria uma categoria
    "outros" que esconderia justamente a concentração que se quer medir.
    """
    faltando = {}
    for dimensao, mapa in dimensoes.items():
        ausentes = sorted(set(tickers) - set(mapa.index))
        if ausentes:
            faltando[dimensao] = ausentes

    if faltando:
        detalhe = "\n".join(
            f"  {dimensao}: {', '.join(ausentes)}"
            for dimensao, ausentes in sorted(faltando.items())
        )
        raise BlocosInvalidos(
            "Ticker sem bloco definido em config/blocos.json:\n"
            + detalhe
            + "\nAcrescente o mapeamento — sem ele a concentração por fator "
            "fica subestimada."
        )


def agrupar(valores: pd.Series, mapa: pd.Series) -> pd.Series:
    """Soma uma série indexada por ticker dentro de cada categoria.

    Serve tanto para peso de capital quanto para contribuição de risco, que
    é justamente a comparação que interessa.
    """
    categorias = mapa.reindex(valores.index)
    if categorias.isna().any():
        ausentes = sorted(categorias[categorias.isna()].index)
        raise BlocosInvalidos(f"Sem categoria para: {ausentes}")

    return valores.groupby(categorias).sum().sort_values(ascending=False)


def resumo_por_dimensao(
    valores: pd.Series, dimensoes: dict[str, pd.Series]
) -> dict[str, pd.Series]:
    """Aplica `agrupar` em todas as dimensões de uma vez."""
    validar_cobertura(dimensoes, list(valores.index))
    return {
        dimensao: agrupar(valores, mapa) for dimensao, mapa in dimensoes.items()
    }
