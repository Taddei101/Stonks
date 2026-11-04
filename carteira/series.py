"""Séries de retorno a partir de preços ajustados.

Este módulo é a ponte entre preço e risco. Tudo aqui é função pura: recebe
DataFrame de preço, devolve DataFrame de retorno. Sem rede, sem matplotlib.

ENTRADA ESPERADA: preço AJUSTADO por provento e desdobramento, vindo de
`precos.historico_total`. Passar a série de valor de mercado produz números
que parecem certos e estão errados — a ação que paga dividendo aparece com
retorno sistematicamente menor e com um degrau na data-ex.

DUAS CONVENÇÕES DE RETORNO, e a escolha não é estética:

  simples   R = P_t/P_{t-1} - 1
            Agrega entre ATIVOS: o retorno de uma carteira é a média
            ponderada dos retornos simples dos componentes. Com log isso é
            falso. Por isso é o que se usa para covariância e carteira.

  log       r = ln(P_t/P_{t-1})
            Agrega no TEMPO: a soma dos log-retornos é o log-retorno do
            período. Útil para acumular e para estatística de cauda.

FREQUÊNCIA: a análise de risco roda em frequência mensal. Retorno diário na
B3 tem ruído de microestrutura e assincronia de negociação — papel menos
líquido como TAEE11 negocia em horários diferentes de ITUB4, o que puxa a
correlação medida para baixo de forma artificial. Mensal elimina quase todo
esse efeito ao custo de menos observações.
"""

import numpy as np
import pandas as pd

MESES_NO_ANO = 12
DIAS_UTEIS_NO_ANO = 252


class HistoricoInsuficiente(ValueError):
    """Não há período comum suficiente entre os ativos pedidos."""


def periodo_comum(precos: pd.DataFrame, minimo: int = 2) -> pd.DataFrame:
    """Recorta para a janela em que TODOS os ativos têm preço.

    Por que cortar em vez de tratar ativo a ativo: correlação e covariância
    só fazem sentido sobre observações emparelhadas. Estimar cada par na sua
    própria janela produz uma matriz que pode nem ser positiva definida, e o
    erro aparece bem longe daqui, como otimização que devolve peso negativo
    sem motivo.

    O custo é visível e deve ser: se um ativo tem histórico curto, a janela
    de todo mundo encolhe. É melhor saber disso do que descobrir depois.
    """
    if precos is None or not len(precos):
        raise HistoricoInsuficiente("Série de preços vazia.")

    recorte = precos.dropna(how="any")
    if len(recorte) < minimo:
        por_ativo = precos.notna().sum().sort_values()
        raise HistoricoInsuficiente(
            f"Só {len(recorte)} observação(ões) com todos os ativos presentes, "
            f"mínimo é {minimo}.\n"
            "Observações por ativo:\n"
            + por_ativo.to_string()
            + "\nO ativo com menos histórico está limitando a janela."
        )
    return recorte


def retornos_diarios(
    precos: pd.DataFrame, metodo: str = "simples"
) -> pd.DataFrame:
    """Retorno dia a dia por ativo.

    `metodo` em {"simples", "log"} — ver a nota no topo do módulo sobre
    quando cada um vale.
    """
    recorte = periodo_comum(precos)

    if metodo == "simples":
        retornos = recorte.pct_change()
    elif metodo == "log":
        retornos = np.log(recorte / recorte.shift(1))
    else:
        raise ValueError(f"metodo deve ser 'simples' ou 'log', veio {metodo!r}")

    return retornos.dropna(how="all")


def retornos_mensais(
    precos: pd.DataFrame, metodo: str = "simples"
) -> pd.DataFrame:
    """Retorno mês a mês por ativo, do último preço de cada mês.

    Esta é a frequência padrão da análise de risco deste projeto.
    """
    recorte = periodo_comum(precos)
    fim_do_mes = _descartar_mes_incompleto(recorte.resample("ME").last(), recorte)

    if metodo == "simples":
        retornos = fim_do_mes.pct_change()
    elif metodo == "log":
        retornos = np.log(fim_do_mes / fim_do_mes.shift(1))
    else:
        raise ValueError(f"metodo deve ser 'simples' ou 'log', veio {metodo!r}")

    return retornos.dropna(how="all")


FRACAO_MINIMA_DO_MES = 0.6


def _descartar_mes_incompleto(
    fim_do_mes: pd.DataFrame, diario: pd.DataFrame
) -> pd.DataFrame:
    """Remove o último mês quando ele ainda está em andamento.

    `resample("ME")` rotula todo balde com o último dia do mês, inclusive o
    do mês corrente, que pode conter dois ou três pregões. O `pct_change`
    seguinte trata esse pedaço como se fosse um mês inteiro: um retorno de
    três dias entra na amostra com o mesmo peso de um de vinte e um, vindo
    de uma distribuição completamente diferente.

    O critério é a CONTAGEM de pregões do último balde contra a mediana dos
    anteriores. Comparar o rótulo do balde com a última data de preço não
    funciona: o rótulo é sempre o último dia do calendário, e quando o mês
    termina em sábado ou feriado — março de 2024 fechou na sexta-feira
    santa — um mês completo pareceria incompleto. A contagem é imune a
    feriado, a calendário de bolsa e a semana curta.
    """
    if len(fim_do_mes) < 2:
        return fim_do_mes

    por_mes = diario.resample("ME").size()
    tipico = float(por_mes.iloc[:-1].median())
    if tipico <= 0:
        return fim_do_mes

    if por_mes.iloc[-1] < tipico * FRACAO_MINIMA_DO_MES:
        return fim_do_mes.iloc[:-1]
    return fim_do_mes


def retorno_carteira(retornos: pd.DataFrame, pesos: pd.Series) -> pd.Series:
    """Série de retorno de uma carteira a pesos FIXOS.

    Peso fixo embute uma premissa que precisa ficar explícita: a carteira é
    rebalanceada ao alvo a cada período. Não é o que acontece na prática —
    entre rebalanceamentos os pesos derivam com o preço. Para comparar
    ALTERNATIVAS DE ALOCAÇÃO isso é adequado e é o padrão da literatura,
    porque isola a decisão de peso do efeito da política de rebalanceamento.

    Para medir o efeito da política de rebalanceamento em si, esta função não
    serve: aí é preciso simular a deriva dos pesos e os custos de negociar.

    Exige retorno SIMPLES. Com log a média ponderada não é o retorno da
    carteira, e o erro cresce com a volatilidade.
    """
    pesos = pesos.astype(float)

    faltando = sorted(set(pesos.index) - set(retornos.columns))
    if faltando:
        raise KeyError(f"Sem série de retorno para: {faltando}")

    soma = float(pesos.sum())
    if not np.isclose(soma, 1.0, atol=1e-6):
        pesos = pesos / soma

    alinhados = retornos[pesos.index]
    return (alinhados * pesos).sum(axis=1).rename("carteira")


def acumulado(retornos) -> pd.Series | pd.DataFrame:
    """Retorno acumulado a partir de retornos simples período a período."""
    return (1 + retornos).cumprod() - 1


def anualizar_retorno(retornos: pd.Series, periodos_no_ano: int = MESES_NO_ANO) -> float:
    """Retorno geométrico anualizado.

    Geométrico, não aritmético: a média aritmética de retornos superestima o
    que se ganha de fato, e a diferença cresce com a volatilidade. Com
    carteira concentrada e volátil, que é o caso aqui, a diferença não é
    acadêmica.
    """
    limpos = pd.Series(retornos).dropna()
    if not len(limpos):
        return float("nan")

    total = float((1 + limpos).prod())
    if total <= 0:
        return float("nan")
    return total ** (periodos_no_ano / len(limpos)) - 1


def anualizar_volatilidade(
    retornos: pd.Series, periodos_no_ano: int = MESES_NO_ANO
) -> float:
    """Desvio padrão anualizado pela raiz do número de períodos.

    A regra da raiz supõe retornos independentes entre períodos. Em série
    real existe autocorrelação e agrupamento de volatilidade, então este
    número subestima um pouco o risco de horizonte longo. É a convenção
    padrão e serve para comparar alternativas entre si, que é o uso aqui.
    """
    limpos = pd.Series(retornos).dropna()
    if len(limpos) < 2:
        return float("nan")
    return float(limpos.std(ddof=1) * np.sqrt(periodos_no_ano))
