"""Gráficos em PNG. Única parte do projeto que depende de matplotlib.

Paleta validada para separação em visão normal e nas três formas de
daltonismo (azul #2a78d6, laranja #eb6834, aqua #1baf7a). A aqua fica abaixo
de 3:1 de contraste contra o fundo claro, então a série que a usa carrega
rótulo direto — a cor não é o único jeito de identificá-la.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem display; só grava arquivo
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_FRACA = "#52514e"
GRADE = "#e3e2de"

SERIE_1 = "#2a78d6"  # azul
SERIE_2 = "#eb6834"  # laranja
SERIE_3 = "#1baf7a"  # aqua

LARGURA_LINHA = 1.8


def _reais(valor, _pos=None) -> str:
    inteiro = f"{valor:,.0f}".replace(",", ".")
    return f"R$ {inteiro}"


def _pct(valor, _pos=None) -> str:
    return f"{valor * 100:,.0f}%".replace(",", ".")


def _limpar(eixo) -> None:
    eixo.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eixo.spines[lado].set_color(GRADE)
    eixo.grid(axis="y", color=GRADE, linewidth=0.8, zorder=0)
    eixo.set_axisbelow(True)
    eixo.tick_params(colors=TINTA_FRACA, labelsize=9, length=0)


def _rotular_fim(eixo, serie, cor, texto=None) -> None:
    """Rótulo direto no último ponto — identidade sem depender da cor."""
    if not len(serie):
        return
    x, y = serie.index[-1], serie.iloc[-1]
    eixo.scatter([x], [y], s=28, color=cor, zorder=5,
                 edgecolor=SUPERFICIE, linewidth=1.5)
    eixo.annotate(
        texto if texto is not None else _reais(y),
        xy=(x, y),
        xytext=(8, 0),
        textcoords="offset points",
        color=TINTA,
        fontsize=9,
        va="center",
    )


def grafico_proventos_mensais(
    proventos_por_linha,
    saida: Path,
    titulo: str = "Renda passiva",
    janela: int = 12,
) -> Path:
    """Barras de provento recebido por mês, com média móvel de 12 meses.

    Recebe as linhas de provento já filtradas (data, valor_operacao).

    Meses sem provento aparecem como barra zero em vez de sumirem — um eixo
    que pula meses vazios mente sobre a regularidade da renda, que é
    exatamente o que se quer enxergar numa carteira de dividendos.

    Barras e média móvel dividem o mesmo eixo: a média é da mesma grandeza
    (R$ por mês), não um segundo eixo.
    """
    import pandas as pd

    serie = (
        proventos_por_linha.set_index("data")["valor_operacao"]
        .resample("MS")
        .sum()
    )
    if len(serie):
        meses = pd.date_range(serie.index.min(), serie.index.max(), freq="MS")
        serie = serie.reindex(meses).fillna(0.0)

    media_movel = serie.rolling(janela, min_periods=janela).mean()

    figura, eixo = plt.subplots(figsize=(11, 5), facecolor=SUPERFICIE)
    _limpar(eixo)

    eixo.bar(
        serie.index, serie.values,
        width=22,  # menos que os ~30 dias do mês: deixa o vão entre barras
        color=SERIE_3, zorder=3, linewidth=0,
    )
    if media_movel.notna().any():
        eixo.plot(
            media_movel.index, media_movel.values,
            color=TINTA_FRACA, linewidth=1.6, zorder=4,
            label=f"Média móvel {janela} meses",
        )
        eixo.legend(loc="upper left", frameon=False, fontsize=9,
                    labelcolor=TINTA_FRACA)

    # Rótulo só nos três maiores meses: a aqua fica abaixo de 3:1 contra o
    # fundo, então a série precisa de identificação que não seja a cor — e
    # rotular toda barra viraria ruído.
    for data in serie.nlargest(3).index:
        valor = serie[data]
        if valor <= 0:
            continue
        eixo.annotate(
            _reais(valor), xy=(data, valor), xytext=(0, 5),
            textcoords="offset points", ha="center",
            color=TINTA, fontsize=8.5,
        )

    eixo.yaxis.set_major_formatter(FuncFormatter(_reais))
    eixo.set_title(
        f"{titulo} — proventos recebidos por mês",
        color=TINTA, fontsize=13, fontweight="bold", loc="left", pad=28,
    )

    ultimos = serie.tail(janela)
    resumo = (
        f"Últimos {len(ultimos)} meses: {_reais(ultimos.sum())}  ·  "
        f"média {_reais(ultimos.mean())}/mês  ·  "
        f"total histórico {_reais(serie.sum())}"
    )
    eixo.annotate(
        resumo, xy=(0, 1.015), xycoords="axes fraction",
        color=TINTA_FRACA, fontsize=9.5, va="bottom",
    )

    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(saida, dpi=150, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(figura)
    return saida


def grafico_retorno(
    valores,
    dados,
    proventos_acumulados,
    saida: Path,
    titulo: str = "Carteira",
) -> Path:
    """Três painéis: retorno TWR, valor vs capital aportado, proventos.

    Painéis separados em vez de eixo duplo: proventos na casa das centenas e
    carteira na casa das dezenas de milhares no mesmo eixo tornaria uma das
    duas invisível.
    """
    figura, eixos = plt.subplots(
        3, 1, figsize=(11, 12), sharex=True, facecolor=SUPERFICIE
    )
    figura.subplots_adjust(hspace=0.28, right=0.88)

    # --- 1. retorno time-weighted --------------------------------------------
    eixo = eixos[0]
    _limpar(eixo)
    serie_twr = dados["serie_twr"]
    eixo.plot(serie_twr.index, serie_twr.values,
              color=SERIE_1, linewidth=LARGURA_LINHA, zorder=3)
    eixo.axhline(0, color=GRADE, linewidth=1, zorder=1)
    eixo.yaxis.set_major_formatter(FuncFormatter(_pct))
    _rotular_fim(eixo, serie_twr, SERIE_1, f"{serie_twr.iloc[-1]:+.1%}")

    eixo.set_title(
        "Retorno time-weighted acumulado",
        color=TINTA, fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    legenda = (
        f"TWR {dados['twr_periodo']:+.1%} no período  ·  "
        f"{dados['twr_anual']:+.1%} ao ano\n"
        f"MWR {dados['mwr_periodo']:+.1%} no período  ·  "
        f"{dados['mwr_anual']:+.1%} ao ano"
    )
    # Dentro da área de plotagem, no canto superior esquerdo: a curva de
    # retorno sobe da esquerda para a direita, então esse canto está livre.
    eixo.annotate(
        legenda,
        xy=(0.015, 0.96), xycoords="axes fraction",
        color=TINTA_FRACA, fontsize=9.5, va="top", linespacing=1.5,
    )

    # --- 2. valor da carteira contra capital aportado ------------------------
    eixo = eixos[1]
    _limpar(eixo)
    total = valores["total"]
    aportado = valores["aporte_externo"].cumsum()

    eixo.plot(total.index, total.values, color=SERIE_1,
              linewidth=LARGURA_LINHA, zorder=3, label="Valor da carteira")
    eixo.plot(aportado.index, aportado.values, color=SERIE_2,
              linewidth=LARGURA_LINHA, zorder=3, label="Capital aportado")
    eixo.fill_between(
        total.index, aportado.values, total.values,
        where=(total.values >= aportado.values),
        color=SERIE_1, alpha=0.08, zorder=2, linewidth=0,
    )
    eixo.yaxis.set_major_formatter(FuncFormatter(_reais))
    _rotular_fim(eixo, total, SERIE_1)
    _rotular_fim(eixo, aportado, SERIE_2)

    eixo.set_title(
        "Valor da carteira e capital aportado",
        color=TINTA, fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    moldura = eixo.legend(
        loc="upper left", frameon=False, fontsize=9, labelcolor=TINTA_FRACA
    )
    moldura.set_zorder(6)

    # --- 3. proventos acumulados --------------------------------------------
    eixo = eixos[2]
    _limpar(eixo)
    eixo.plot(
        proventos_acumulados.index, proventos_acumulados.values,
        color=SERIE_3, linewidth=LARGURA_LINHA, zorder=3,
    )
    eixo.fill_between(
        proventos_acumulados.index, 0, proventos_acumulados.values,
        color=SERIE_3, alpha=0.10, zorder=2, linewidth=0,
    )
    eixo.yaxis.set_major_formatter(FuncFormatter(_reais))
    _rotular_fim(eixo, proventos_acumulados, SERIE_3)

    eixo.set_title(
        "Proventos acumulados (dividendos, JCP e rendimentos)",
        color=TINTA, fontsize=12, fontweight="bold", loc="left", pad=12,
    )
    eixo.tick_params(axis="x", labelsize=9)

    figura.suptitle(
        titulo, color=TINTA, fontsize=15, fontweight="bold",
        x=0.09, ha="left", y=0.975,
    )

    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(saida, dpi=150, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(figura)
    return saida
