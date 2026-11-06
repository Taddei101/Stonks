"""Leitura do relatório de Posição da B3 (Minha carteira → Posição).

O xlsx traz uma aba por tipo de ativo, e só as abas do que a pessoa tem.
Aqui entram as de renda variável; o resto (Renda Fixa, Tesouro…) é
devolvido pelo nome em `abas_ignoradas`, para a tela avisar o cliente.

Saída: uma linha por ticker, somando o mesmo ticker em corretoras
diferentes. Quem consome (rebalanceamento, comparação de métodos) não
precisa saber de qual aba o ativo veio.
"""

from typing import NamedTuple

import pandas as pd

# Nome da aba exatamente como a B3 exporta: "Acoes" vem sem acento.
CLASSES = {
    "Acoes": "Ação",
    "ETF": "ETF",
    "Fundo de Investimento": "FII",
}

COLUNAS = ["Código de Negociação", "Quantidade", "Preço de Fechamento", "Valor Atualizado"]
SAIDA = ["ticker", "classe", "quantidade", "preco", "valor"]


class LayoutInesperado(ValueError):
    """Uma aba de renda variável não tem as colunas esperadas da B3."""


class Posicao(NamedTuple):
    ativos: pd.DataFrame
    direitos: pd.DataFrame
    abas_ignoradas: list[str]


def _ler_aba(df: pd.DataFrame, aba: str) -> pd.DataFrame:
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise LayoutInesperado(f"Aba {aba!r} sem as colunas {faltando}")

    codigo = df["Código de Negociação"].astype("string").str.strip()
    # Rodapé da B3 (linha vazia, "Total", valor do total) não tem código.
    df = df[codigo.notna() & (codigo != "")]

    numeros = df[COLUNAS[1:]].replace("-", pd.NA).apply(pd.to_numeric)
    return pd.DataFrame(
        {
            "ticker": codigo[df.index],
            "classe": CLASSES[aba],
            "quantidade": numeros["Quantidade"],
            "preco": numeros["Preço de Fechamento"],
            "valor": numeros["Valor Atualizado"],
            "direito": df.get("Tipo", pd.Series("", index=df.index)).eq("Direito"),
        }
    )


def _somar_por_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SAIDA)
    return df.groupby(["ticker", "classe"], as_index=False, sort=True).agg(
        quantidade=("quantidade", "sum"),
        preco=("preco", "first"),
        valor=("valor", "sum"),
    )


def ler_posicao(arquivo) -> Posicao:
    """Lê o xlsx de Posição. Aceita caminho ou arquivo aberto (upload do Flask)."""
    abas = pd.read_excel(arquivo, sheet_name=None, dtype=object)

    partes = [_ler_aba(df, nome) for nome, df in abas.items() if nome in CLASSES]
    tudo = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=SAIDA + ["direito"])
    direito = tudo["direito"].astype(bool)

    return Posicao(
        ativos=_somar_por_ticker(tudo[~direito]),
        direitos=_somar_por_ticker(tudo[direito]),
        abas_ignoradas=[nome for nome in abas if nome not in CLASSES],
    )
