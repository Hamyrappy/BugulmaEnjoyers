"""Main entrypoint for BugulmaEnjoyers."""

import click

# from bugulma_enjoyers.detoxifiers import TheOneAndSuperDetoxifierWeFinallySelected # noqa: ERA001
from bugulma_enjoyers.detoxifiers import PipelineConfig, StandaloneDetoxifier
from bugulma_enjoyers.io import read_input, write_output
from bugulma_enjoyers.setup_logging import setup_logging


@click.option("--verbose", "-v", count=True, default=False)
@click.option("--quiet", "-q", count=True, default=False)
@click.option("--file", "-f", help="File to read.", default="dev_inputs.tsv")
@click.option("--output", "-o", help="File to write.", default="dev_outputs.tsv")
@click.command()
def main(
    file: str = "dev_inputs.tsv",
    output: str = "dev_outputs.tsv",
    verbose: int = 0,
    quiet: int = 0,
) -> None:
    """Main entrypoint for BugulmaEnjoyers."""
    verbosity = verbose - quiet +1
    setup_logging(verbosity)
    texts = read_input(file)
    config = PipelineConfig()
    detox = StandaloneDetoxifier(config)
    results = detox.detoxify_batch(texts, ["tt"] * len(texts))
    write_output(results, texts, output)


if __name__ == "__main__":
    main()
