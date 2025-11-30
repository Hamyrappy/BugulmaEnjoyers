from pathlib import Path

import pandas as pd

REQUIRED_COL = "tat_toxic"

COL_NOT_FOUND = "Column '{column}' not found in the input file."


def read_input(path: str | Path, column: str = REQUIRED_COL) -> str:
    path = Path(path)
    try:
        df = pd.read_csv(str(path.resolve().absolute()), sep="\t", index_col="ID")
    except ValueError:
        df = pd.read_csv(str(path.resolve().absolute()), sep="\t")
    if column not in df.columns:
        raise ValueError(COL_NOT_FOUND.format(column=column))
    return df[column].to_list()
