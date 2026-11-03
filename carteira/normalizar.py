"""Conversão do extrato bruto para dados tipados, e a taxonomia de movimentações.

A taxonomia é o coração deste módulo. O extrato da B3 mistura três coisas
bem diferentes na mesma coluna "Movimentação", e tratá-las igual produz erro
silencioso e grande.
"""

import pandas as pd

RENOMEAR = {
    "Entrada/Saída": "entrada_saida",
    "Data": "data",
    "Movimentação": "movimentacao",
    "Produto": "produto",
    "Instituição": "instituicao",
    "Quantidade": "quantidade",
    "Preço unitário": "preco_unitario",
    "Valor da Operação": "valor_operacao",
}

# Mexem na custódia. O sinal vem sempre de Entrada/Saída:
#   Transferência - Liquidação  Credito = compra, Debito = venda
#   Bonificação em Ativos       Credito = ações recebidas (pode ser fracionária)
#   Fração em Ativos            Debito  = a fração é retirada para ir a leilão
MOV_QUANTIDADE = frozenset(
    {
        "Transferência - Liquidação",
        "Bonificação em Ativos",
        "Fração em Ativos",
    }
)

# Entram ou saem dinheiro, nunca quantidade.
#   Leilão de Fração é o produto da venda da fração retirada acima.
MOV_CAIXA = frozenset(
    {
        "Dividendo",
        "Juros Sobre Capital Próprio",
        "Rendimento",
        "Resgate",
        "Leilão de Fração",
    }
)

# NÃO é movimentação: é uma foto da posição do papel naquela data.
# Somar estas linhas infla a carteira. Elas servem como conferência — ver
# `posicao.conferir_atualizacoes`.
MOV_SNAPSHOT = frozenset({"Atualização"})

SINAL = {"Credito": 1, "Debito": -1}


class MovimentacaoDesconhecida(ValueError):
    """O extrato trouxe um tipo de movimentação que ainda não foi classificado."""


def _para_float(valor) -> float:
    """Converte número do extrato, tolerando `-` e formato brasileiro.

    O extrato usa `-` para "não se aplica" (bonificação e atualização não têm
    preço). O tratamento de vírgula é defensivo: hoje o export vem com ponto
    decimal, mas é um arquivo de origem web e isso pode mudar.
    """
    if valor is None:
        return float("nan")

    texto = str(valor).strip()
    if texto in {"", "-", "nan", "None", "NaN"}:
        return float("nan")

    if "." in texto and "," in texto:
        # 1.855,56 -> separador de milhar é o ponto
        texto = texto.replace(".", "").replace(",", ".")
    elif "," in texto:
        texto = texto.replace(",", ".")

    return float(texto)


def _classificar(movimentacao: pd.Series) -> pd.Series:
    desconhecidas = set(movimentacao.unique()) - (
        MOV_QUANTIDADE | MOV_CAIXA | MOV_SNAPSHOT
    )
    if desconhecidas:
        raise MovimentacaoDesconhecida(
            "Tipo(s) de movimentação não classificado(s): "
            f"{sorted(desconhecidas)}.\n"
            "Classifique em MOV_QUANTIDADE, MOV_CAIXA ou MOV_SNAPSHOT antes "
            "de seguir — ignorar silenciosamente corromperia a posição."
        )

    def classe(m: str) -> str:
        if m in MOV_QUANTIDADE:
            return "quantidade"
        if m in MOV_CAIXA:
            return "caixa"
        return "snapshot"

    return movimentacao.map(classe)


def normalizar(bruto: pd.DataFrame) -> pd.DataFrame:
    """Devolve o extrato tipado, com ticker extraído e movimentação classificada.

    Falha se aparecer um tipo de movimentação desconhecido: é o único jeito de
    descobrir que a B3 passou a registrar um evento novo (desdobramento,
    incorporação) antes que ele estrague a posição em silêncio.
    """
    df = bruto.rename(columns=RENOMEAR).copy()

    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")

    # O nome do produto varia para o mesmo papel ("ITAU UNIBANCO HOLDING S.A"
    # e "S/A" convivem no mesmo arquivo), então só o ticker serve como chave.
    df["ticker"] = df["produto"].str.split(" - ").str[0].str.strip()

    for coluna in ("quantidade", "preco_unitario", "valor_operacao"):
        df[coluna] = df[coluna].map(_para_float)

    df["classe"] = _classificar(df["movimentacao"])

    sinal = df["entrada_saida"].map(SINAL)
    if sinal.isna().any():
        invalidos = sorted(df.loc[sinal.isna(), "entrada_saida"].unique())
        raise ValueError(f"Entrada/Saída inesperada: {invalidos}")
    df["sinal"] = sinal.astype(int)

    colunas = [
        "data",
        "movimentacao",
        "classe",
        "entrada_saida",
        "sinal",
        "ticker",
        "quantidade",
        "preco_unitario",
        "valor_operacao",
        "instituicao",
        "produto",
    ]
    if "_arquivo" in df.columns:
        colunas.append("_arquivo")

    return (
        df[colunas]
        .sort_values(["data", "ticker", "movimentacao"])
        .reset_index(drop=True)
    )