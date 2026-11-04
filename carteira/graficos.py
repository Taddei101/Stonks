"""Gráficos em PNG. Única parte do projeto que depende de matplotlib.

Paleta validada para separação em visão normal e nas três formas de
daltonismo (azul #2a78d6, laranja #eb6834, aqua #1baf7a). A aqua fica abaixo
de 3:1 de contraste contra o fundo claro, então a série que a usa carrega
rótulo direto — a cor não é o único jeito de identificá-la.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem display; só grava arquivo

# O cifrão de "R$" delimita fórmula no mathtext do matplotlib: uma string com
# número PAR de $ é lida como matemática e sai em itálico e embaralhada
# ("R$ 283 · média R$ 26" vira "R283·médiaR 26"). Como todo rótulo aqui é em
# reais, desligar a interpretação é mais seguro que escapar caso a caso.
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
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


def _eixo_mensal(eixo, inicio, fim, alvo_ticks: int = 10) -> None:
    """Rótulos de data em mm/aa, com espaçamento em múltiplo redondo de mês.

    O formatador padrão do matplotlib escolhe o passo sozinho e escreve
    2024-07, que é longo e não é o formato que se lê aqui. O intervalo sai de
    um alvo de ~10 marcas, arredondado para 1, 2, 3, 6 ou 12 meses — passo
    quebrado (5 em 5 meses) atrapalha a leitura.
    """
    meses = max(1, (fim.year - inicio.year) * 12 + fim.month - inicio.month)
    bruto = max(1, round(meses / alvo_ticks))
    intervalo = next((c for c in (1, 2, 3, 6, 12) if c >= bruto), 12)

    eixo.xaxis.set_major_locator(mdates.MonthLocator(interval=intervalo))
    eixo.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))


def _salvar(figura, saida: Path) -> Path:
    """Grava o PNG de forma atômica, com erro legível quando o SO recusa.

    Escreve num arquivo temporário ao lado e só então renomeia. Além de não
    deixar PNG pela metade se algo falhar no meio, isso separa dois problemas
    que o traceback cru confunde: falha ao *criar* arquivo novo e falha ao
    *sobrescrever* o arquivo final.
    """
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    temporario = saida.with_name(f".{saida.stem}.tmp{saida.suffix}")

    try:
        figura.savefig(
            temporario, dpi=150, facecolor=SUPERFICIE, bbox_inches="tight"
        )
        os.replace(temporario, saida)
    except OSError as erro:
        temporario.unlink(missing_ok=True)
        raise OSError(
            f"Não consegui gravar {saida}.\n"
            f"  erro do sistema: {erro}\n"
            "  Causas prováveis, nesta ordem:\n"
            "   1. O arquivo está aberto num visualizador de imagem. "
            "Feche e rode de novo.\n"
            "   2. A pasta está sincronizada pelo OneDrive (Documentos "
            "costuma estar). A sincronização pode recusar a escrita; "
            "pause o OneDrive ou mova o projeto para fora de Documentos.\n"
            "   3. Antivírus bloqueando escrita de arquivo novo."
        ) from erro
    finally:
        plt.close(figura)

    return saida


MESES_ABREV = ["jan", "fev", "mar", "abr", "mai", "jun",
               "jul", "ago", "set", "out", "nov", "dez"]


def _serie_mensal(proventos_por_linha) -> "pd.Series":
    """Soma por mês, com mês vazio preenchido com zero em vez de sumir.

    Um eixo que pula mês sem provento mente sobre a regularidade da renda,
    que é justamente o que se quer enxergar numa carteira de dividendos.
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
    return serie


def grafico_proventos_ano(
    proventos_por_linha,
    saida: Path,
    ano: int | None = None,
    hoje=None,
    titulo: str = "Renda passiva",
) -> Path:
    """Doze barras, janeiro a dezembro, do ano escolhido.

    Os doze meses aparecem sempre, mesmo sem pagamento — num gráfico de renda
    passiva o buraco é informação.

    Mês futuro do ano corrente é desenhado esmaecido: "não recebi em março" e
    "dezembro ainda não chegou" são coisas diferentes, e as duas valeriam
    barra zero se nada as distinguisse. A média de referência também só conta
    os meses já decorridos, senão dezembro vazio puxaria a média para baixo.
    """
    import pandas as pd

    hoje = pd.Timestamp(hoje) if hoje is not None else pd.Timestamp.today()
    ano = int(ano) if ano is not None else hoje.year

    serie = _serie_mensal(proventos_por_linha)
    calendario = pd.date_range(f"{ano}-01-01", f"{ano}-12-01", freq="MS")
    do_ano = serie.reindex(calendario).fillna(0.0)

    # O mês corrente não é nenhum dos dois: ainda pode receber pagamento, mas
    # não pode entrar na média como se estivesse fechado.
    if ano < hoje.year:
        fechados, indice_corrente = 12, None
    elif ano > hoje.year:
        fechados, indice_corrente = 0, None
    else:
        fechados, indice_corrente = hoje.month - 1, hoje.month - 1

    futuro = [
        (indice_corrente is not None and i > indice_corrente) or ano > hoje.year
        for i in range(12)
    ]

    figura, eixo = plt.subplots(figsize=(11, 5), facecolor=SUPERFICIE)
    _limpar(eixo)

    posicoes = range(12)
    eixo.bar(
        posicoes, do_ano.values, width=0.62, zorder=3, linewidth=0,
        # alpha por barra não é aceito pelo bar(); vai embutido na cor
        color=[to_rgba(SERIE_3, 0.25 if f else 1.0) for f in futuro],
    )

    realizados = do_ano.values[:fechados]
    if fechados:
        plural = "mês fechado" if fechados == 1 else "meses fechados"
        eixo.axhline(
            realizados.mean(), color=TINTA_FRACA, linewidth=1.4, zorder=4,
            label=f"Média dos {fechados} {plural}",
        )
        eixo.legend(loc="upper left", frameon=False, fontsize=9,
                    labelcolor=TINTA_FRACA)

    # Rótulo nos três maiores: a aqua fica abaixo de 3:1 de contraste contra
    # o fundo, então a série precisa de identificação que não seja a cor.
    for i in do_ano.reset_index(drop=True).nlargest(3).index:
        valor = do_ano.values[i]
        if valor <= 0:
            continue
        eixo.annotate(
            _reais(valor), xy=(i, valor), xytext=(0, 5),
            textcoords="offset points", ha="center",
            color=TINTA, fontsize=8.5,
        )

    eixo.margins(y=0.18)  # folga no topo para legenda e rótulos
    eixo.set_ylim(bottom=0)
    eixo.set_xticks(list(posicoes))
    eixo.set_xticklabels(MESES_ABREV)
    eixo.yaxis.set_major_formatter(FuncFormatter(_reais))
    eixo.set_title(
        f"{titulo} — proventos recebidos em {ano}",
        color=TINTA, fontsize=13, fontweight="bold", loc="left", pad=28,
    )

    partes = [f"Total {ano}: {_reais(do_ano.sum())}"]
    if fechados:
        partes.append(f"média {_reais(realizados.mean())}/mês")
    if indice_corrente is not None:
        partes.append(f"{MESES_ABREV[indice_corrente]} em curso")
    restantes = sum(futuro)
    if restantes:
        plural = "mês" if restantes == 1 else "meses"
        partes.append(f"{restantes} {plural} por vir (esmaecido)")
    eixo.annotate(
        "  ·  ".join(partes), xy=(0, 1.015), xycoords="axes fraction",
        color=TINTA_FRACA, fontsize=9.5, va="bottom",
    )

    return _salvar(figura, saida)


def grafico_proventos_mensais(
    proventos_por_linha,
    saida: Path,
    titulo: str = "Renda passiva",
    janela: int = 12,
) -> Path:
    """Histórico completo mês a mês, com média móvel de 12 meses.

    Visão de série longa. Para o ano corrente use `grafico_proventos_ano`.
    """
    serie = _serie_mensal(proventos_por_linha)
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

    if len(serie):
        _eixo_mensal(eixo, serie.index.min(), serie.index.max())

    return _salvar(figura, saida)


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
    _eixo_mensal(eixo, valores.index.min(), valores.index.max())

    figura.suptitle(
        titulo, color=TINTA, fontsize=15, fontweight="bold",
        x=0.09, ha="left", y=0.975,
    )

    return _salvar(figura, saida)
