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
@click.option("--batch-size-1", help="Batch size for first detoxifier.", default=8)
@click.option("--batch-size-2", help="Batch size for second detoxifier.", default=8)
@click.command()
def main(
    file: str = "dev_inputs.tsv",
    output: str = "dev_outputs.tsv",
    verbose: int = 0,
    quiet: int = 0,
    batch_size_1: int = 8,
    batch_size_2: int = 8,
) -> None:
    """Main entrypoint for BugulmaEnjoyers."""
    verbosity = verbose - quiet + 1
    setup_logging(verbosity)
    texts = read_input(file)
    config = PipelineConfig(batch_size=batch_size_1)
    detox = StandaloneDetoxifier(config)
    config2 = PipelineConfig(
        detoxifier_model_name="google/models/gemini-2.5-pro", batch_size=batch_size_2,
    )
    detox2 = StandaloneDetoxifier(config2)
    results = detox2.detoxify_batch(
        detox.detoxify_batch(texts, ["tt"] * len(texts)),
        ["tt"] * len(texts),
    )
    write_output(results, texts, output)


if __name__ == "__main__":
    main()
