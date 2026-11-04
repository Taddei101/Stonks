"""Validação deslizante: estima com o passado, mede no futuro.

O PROBLEMA

Comparar regras de alocação na mesma janela em que elas foram estimadas não
responde nada. A mínima variância vai ter a menor volatilidade — ela
minimizou exatamente essa quantidade nesses dados. Dizer que ela "venceu" é
dizer que a otimização funcionou.

O CUSTO DA HONESTIDADE

Com 119 meses e treino de 36, sobram 83 avaliações fora da amostra. Os 36
primeiros nunca são testados: servem só para a primeira estimativa. Isso não
é defeito corrigível — é o preço de fazer a pergunta certa.

Note a assimetria que isso expõe: 1/N não tem parâmetro nenhum, então não
paga esse custo e não sofre com erro de estimativa. É por isso que ela é uma
régua tão difícil, e por que regras otimizadas que brilham dentro da amostra
costumam perder fora dela.

DUAS VARIANTES, COMPROMISSOS OPOSTOS

  rolante    sempre os últimos N meses. Descarta dado antigo, mas é como
             alguém usaria de verdade, e adapta a mudança de regime.
  ancorada   tudo desde o início até t, crescendo. Aproveita mais dados, mas
             as primeiras estimativas são ruidosas e o método muda de
             caráter ao longo do tempo.

Rodar as duas é informativo por si: conclusão que sobrevive às duas é
robusta; divergência entre elas diz quanta estrutura antiga ainda vale.

GIRO

Cada reestimativa muda os pesos, e mudar peso custa corretagem e imposto.
O giro medido aqui é o custo que a regra impõe antes de qualquer benefício —
e regras que dependem de estimar retorno costumam girar muito.
"""

import numpy as np
import pandas as pd

MESES_TREINO = 36


class JanelaInsuficiente(ValueError):
    """Dados de menos para treinar e ainda sobrar período de teste."""


def _janela(retornos: pd.DataFrame, t: int, treino: int, modo: str) -> pd.DataFrame:
    if modo == "rolante":
        return retornos.iloc[t - treino : t]
    if modo == "ancorada":
        return retornos.iloc[:t]
    raise ValueError(f"modo deve ser 'rolante' ou 'ancorada', veio {modo!r}")


def deslizante(
    retornos: pd.DataFrame,
    regras: dict,
    treino: int = MESES_TREINO,
    modo: str = "rolante",
) -> dict:
    """Roda a validação e devolve retornos fora da amostra, pesos e giro.

    `regras` é {nome: função(retornos) -> pesos}. Toda regra recebe APENAS
    os dados anteriores ao mês avaliado — é isso que torna o resultado
    informativo sobre o futuro.

    Devolve:
      retornos   DataFrame mês x regra, fora da amostra
      pesos      {regra: DataFrame mês x ativo}
      giro       DataFrame mês x regra, soma das mudanças de peso
      falhas     {regra: número de meses em que a regra não pôde ser
                 calculada; nesses meses ela mantém os pesos anteriores}
    """
    if len(retornos) <= treino:
        raise JanelaInsuficiente(
            f"{len(retornos)} meses de dados e treino de {treino}: não sobra "
            "nada para avaliar. Use janela maior ou treino menor."
        )

    datas = retornos.index[treino:]
    fora = {nome: [] for nome in regras}
    historico_pesos = {nome: [] for nome in regras}
    giros = {nome: [] for nome in regras}
    falhas = {nome: 0 for nome in regras}
    anteriores = {nome: None for nome in regras}

    for t in range(treino, len(retornos)):
        passado = _janela(retornos, t, treino, modo)
        futuro = retornos.iloc[t]

        for nome, regra in regras.items():
            try:
                pesos = regra(passado).reindex(retornos.columns).fillna(0.0)
                soma = float(pesos.sum())
                pesos = pesos / soma if soma else pesos
            except Exception:
                # Regra que não converge num mês não derruba a validação
                # inteira. Mantém a carteira anterior, que é o que alguém
                # faria na prática, e o contador registra o ocorrido.
                falhas[nome] += 1
                pesos = anteriores[nome]
                if pesos is None:
                    pesos = pd.Series(
                        1.0 / len(retornos.columns), index=retornos.columns
                    )

            anterior = anteriores[nome]
            giros[nome].append(
                0.0 if anterior is None else float(np.abs(pesos - anterior).sum())
            )
            anteriores[nome] = pesos

            historico_pesos[nome].append(pesos)
            fora[nome].append(float((futuro * pesos).sum()))

    return {
        "retornos": pd.DataFrame(fora, index=datas),
        "pesos": {
            nome: pd.DataFrame(lista, index=datas)
            for nome, lista in historico_pesos.items()
        },
        "giro": pd.DataFrame(giros, index=datas),
        "falhas": falhas,
        "modo": modo,
        "treino": treino,
        "descartados": treino,
    }


def resumo(resultado: dict, cdi: pd.Series | None = None) -> pd.DataFrame:
    """Painel das regras sobre o período fora da amostra."""
    from carteira.comparar import comparar

    retornos = resultado["retornos"]
    series = {nome: retornos[nome] for nome in retornos.columns}

    referencia = "iguais" if "iguais" in series else list(series)[0]
    tabela = comparar(series, cdi, referencia=referencia)

    tabela["giro_anual"] = resultado["giro"].mean() * 12
    tabela["falhas"] = pd.Series(resultado["falhas"])
    return tabela


def formatar(resultado: dict, tabela: pd.DataFrame) -> str:
    """Renderiza a validação para o terminal."""
    linhas = []
    retornos = resultado["retornos"]
    referencia = "iguais" if "iguais" in tabela.index else tabela.index[0]

    linhas.append("=" * 78)
    linhas.append(f"VALIDAÇÃO DESLIZANTE — janela {resultado['modo'].upper()}")
    linhas.append("=" * 78)
    linhas.append(
        f"Treino: {resultado['treino']} meses   |   "
        f"Avaliado fora da amostra: {len(retornos)} meses "
        f"({retornos.index.min():%m/%Y} a {retornos.index.max():%m/%Y})"
    )
    linhas.append(
        f"Os {resultado['descartados']} primeiros meses não são avaliados — "
        "servem só para a primeira estimativa."
    )
    linhas.append("")

    linhas.append(
        "  " + "regra".ljust(20) + "retorno".rjust(9) + "vol".rjust(8)
        + "Sharpe".rjust(8) + "  perda máx".rjust(11) + "  giro/ano".rjust(10)
    )
    linhas.append("  " + "-" * 74)
    for nome, linha in tabela.iterrows():
        marca = "  (ref)" if nome == referencia else ""
        linhas.append(
            f"  {str(nome).ljust(20)}"
            f"{linha['retorno_anual']:+8.2%} "
            f"{linha['volatilidade_anual']:7.2%} "
            f"{linha['sharpe']:7.2f} "
            f"{linha['perda_maxima']:10.1%} "
            f"{linha['giro_anual']:9.1%}{marca}"
        )

    linhas.append("")
    linhas.append(f"PROBABILIDADE DE SUPERAR {referencia.upper()}, FORA DA AMOSTRA")
    for nome, linha in tabela.iterrows():
        if nome == referencia or not np.isfinite(linha["prob_superar_ref"]):
            continue
        p = linha["prob_superar_ref"]
        veredito = (
            "diferença real" if p > 0.90 or p < 0.10
            else "empate" if 0.35 <= p <= 0.65
            else "sugestivo, não conclusivo"
        )
        linhas.append(f"  {str(nome).ljust(20)}{p:6.1%}   {veredito}")

    problemas = {n: q for n, q in resultado["falhas"].items() if q}
    if problemas:
        linhas.append("")
        linhas.append("REGRAS QUE FALHARAM EM ALGUM MÊS (mantiveram a carteira anterior)")
        for nome, quantas in sorted(problemas.items()):
            linhas.append(f"  {str(nome).ljust(20)}{quantas} mês(es)")

    linhas.append("")
    linhas.append(
        "GIRO é a soma das mudanças de peso a cada reestimativa, anualizada.\n"
        "  100% ao ano significa trocar o equivalente à carteira inteira por\n"
        "  ano — custo em corretagem e imposto que o retorno acima NÃO desconta."
    )
    largura = tabela["sharpe_max"] - tabela["sharpe_min"]
    linhas.append("")
    linhas.append(
        f"Largura típica do intervalo de Sharpe: {largura.median():.2f}. "
        "Diferenças menores\nque isso não são distinguíveis."
    )
    linhas.append("=" * 78)
    return "\n".join(linhas)
