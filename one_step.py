"""Single-step detoxification experiments.

This module is used for experimenting with individual detoxifiers.
For two-step detoxification, use main.py.
"""

import click

from bugulma_enjoyers.detoxifiers import PipelineConfig, StandaloneDetoxifier
from bugulma_enjoyers.io import read_input, write_output
from bugulma_enjoyers.setup_logging import setup_logging


@click.option("--verbose", "-v", count=True, default=False)
@click.option("--quiet", "-q", count=True, default=False)
@click.option("--file", "-f", help="File to read.", default="dev_inputs.tsv")
@click.option("--output", "-o", help="File to write.", default="dev_outputs.tsv")
@click.option(
    "--detoxifier", help="Detoxifier model name.", default="hf/s-nlp/mt0-xl-detox-orpo"
)
@click.option("--batch-size", help="Batch size for detoxifier.", default=8)
@click.option("--language","-l", help="Language of prompts: tt for tatar, ru for russian, en for english", default='tt')
@click.command()
def main(
    file: str = "dev_inputs.tsv",
    output: str = "dev_outputs.tsv",
    verbose: int = 0,
    quiet: int = 0,
    detoxifier: str = "hf/s-nlp/mt0-xl-detox-orpo",
    batch_size: int = 8,
    language: str = 'tt',
) -> None:
    """Run single-step detoxification for experiments."""
    verbosity = verbose - quiet + 1
    setup_logging(verbosity)
    texts = read_input(file)
    config = PipelineConfig(detoxifier_model_name=detoxifier, batch_size=batch_size)
    detox = StandaloneDetoxifier(config)
    results = detox.detoxify_batch(texts, [language] * len(texts))
    write_output(results, texts, output)


if __name__ == "__main__":
    main()
