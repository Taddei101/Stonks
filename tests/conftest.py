"""Dados de teste.

O extrato de exemplo é inventado — nenhum dado real de carteira entra no
repositório. Ele vive como CSV (versionável, legível no diff) e vira .xlsx
em tempo de teste, porque é .xlsx que a B3 exporta e é .xlsx que o
`io_extrato` sabe ler. Assim o `.gitignore` segue barrando `*.xlsx` sem
precisar de exceção, e ninguém commita planilha binária por acidente.

Os números conferem de propósito: a posição derivada de AAAA3 ao fim de
2024 é 100 + 10 - 30 - 0,5 = 79,5, exatamente o que a linha de `Atualização`
informa. Isso faz o teste de `conferir_atualizacoes` ter significado.
"""

from pathlib import Path

import pandas as pd
import pytest

PASTA_DADOS = Path(__file__).parent / "dados"
EXEMPLO_CSV = PASTA_DADOS / "movimentacao-exemplo.csv"


def _escrever_xlsx(origem_csv: Path, destino_xlsx: Path) -> Path:
    """Converte o CSV de exemplo em .xlsx no layout exato da B3.

    `dtype=str` em todo o caminho: a coluna de preço contém `-` literal nas
    linhas de bonificação e atualização, e deixar o pandas inferir tipo aqui
    esconderia justamente o caso que o `normalizar._para_float` trata.
    """
    df = pd.read_csv(origem_csv, dtype=str)
    destino_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(destino_xlsx, index=False)
    return destino_xlsx


@pytest.fixture
def pasta_extratos(tmp_path: Path) -> Path:
    """Pasta com um extrato de exemplo, no formato que a B3 emite.

    O nome segue o padrão real (`movimentacao-<timestamp>.xlsx`), porque o
    `encontrar_extratos` depende de ordem alfabética ser ordem cronológica.
    """
    pasta = tmp_path / "extratos"
    _escrever_xlsx(EXEMPLO_CSV, pasta / "movimentacao-2024-12-31-10-00-00.xlsx")
    return pasta


@pytest.fixture
def pasta_extratos_duplicados(tmp_path: Path) -> Path:
    """Dois extratos sobrepostos, como acontece em downloads sucessivos.

    Cada export novo da B3 repete todo o histórico anterior. A deduplicação
    precisa remover a repetição sem apagar as duas linhas de JCP legítimas
    que existem no exemplo (mesmo papel, mesma data, mesmo valor).
    """
    pasta = tmp_path / "extratos"
    _escrever_xlsx(EXEMPLO_CSV, pasta / "movimentacao-2024-06-30-10-00-00.xlsx")
    _escrever_xlsx(EXEMPLO_CSV, pasta / "movimentacao-2024-12-31-10-00-00.xlsx")
    return pasta


@pytest.fixture
def movimentacoes(pasta_extratos: Path) -> pd.DataFrame:
    """Extrato de exemplo já lido e normalizado."""
    from carteira.io_extrato import carregar_movimentacoes
    from carteira.normalizar import normalizar

    return normalizar(carregar_movimentacoes(pasta_extratos))


@pytest.fixture
def ledger(movimentacoes: pd.DataFrame) -> pd.DataFrame:
    """Livro de deltas de quantidade do extrato de exemplo, sem ajustes."""
    from carteira.posicao import construir_ledger

    return construir_ledger(movimentacoes)
