import json
from pathlib import Path

from bugulma_enjoyers.io import read_input, write_output


def test_input():
    inp = read_input(Path("tests") / "resources" / "sample_input.tsv")
    assert len(inp) == 37
    assert inp[0] == "Это пример токсичного комментария."
    for comment in inp:
        assert isinstance(comment, str)


def test_output():
    inp = read_input(Path("tests") / "resources" / "sample_input.tsv")
    out = ["Это пример детоксированного комментария."] * len(inp)
    respath = Path("tests") / "temp" / "sample_out.tsv"
    if respath.exists():
        respath.unlink()
    respath.parent.mkdir(parents=True, exist_ok=True)
    write_output(out, inp, respath)
    assert respath.exists()
    read = read_input(respath, column="tat_detox1")
    assert read == out

