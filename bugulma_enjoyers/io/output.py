from pathlib import Path

import pandas as pd


def write_output(results: list[str], inputs: list[str], path: str | Path) -> None:
    path = Path(path)
    df = pd.DataFrame({"ID": range(len(inputs)), "tat_toxic": inputs, "tat_detox1": results})

    df.to_csv(str(path.absolute().resolve()), sep="\t", index_label="ID", encoding="utf-8")
    