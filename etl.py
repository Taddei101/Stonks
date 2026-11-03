import pandas as pd
from pathlib import Path
import sys


VALID_HEADERS = ['Entrada/Saída', 'Data', 'Movimentação', 'Produto', 'Instituição', 'Quantidade', 'Preço unitário', 'Valor da Operação']

def get_folder():
    try:
        folder = Path(__file__).parent / "extratos"
        excel_files = sorted(f for f in folder.glob("*.xlsx") if not f.name.startswith("~$"))
        return excel_files

    except FileNotFoundError:
        print("No Excel files on folder")
        return []

def data_frame_pattern():
    excel_files = get_folder()
    df = pd.read_excel(excel_files[0], dtype=str)
    headers = df.columns.tolist()

    if headers != VALID_HEADERS:
        print("invalid XLSX file")
        sys.exit(1)

    else:
        print(df.head())
        return df



def main():
    data_frame_pattern()


if __name__ == "__main__":
    main()
