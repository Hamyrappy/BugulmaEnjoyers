"""Main entrypoint for BugulmaEnjoyers."""

from pathlib import Path

import click

# from bugulma_enjoyers.detoxifiers import TheOneAndSuperDetoxifierWeFinallySelected # noqa: ERA001


@click.option("--file", "-f", help="File to read.", default="data.txt")
@click.option("--output", "-o", help="File to write.", default="result.txt")
@click.command()
def main(file: str = "data.txt", output: str = "result.txt") -> None:
    """Main entrypoint for BugulmaEnjoyers."""
    fp = Path(file)
    with fp.open() as f:
        texts = [line.strip() for line in f]
    results = texts  # TheOneAndSuperDetoxifierWeFinallySelected.detoxify_batch(texts)
    with Path(output).open("w") as f:
        f.write("\n".join(results))


if __name__ == "__main__":
    main()
