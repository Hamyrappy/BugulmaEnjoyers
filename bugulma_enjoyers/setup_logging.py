"""Utility functions for setting up logging."""

import logging


def setup_logging(verbosity: int) -> None:
    """
    Set up logging with a level based on the verbosity parameter.

    Args:
        verbosity (int): The verbosity level, from -1 (only critical messages) to 2 (all messages).

    Returns:
        None

    """
    logging_level = max(min(logging.CRITICAL, logging.CRITICAL - verbosity * 10), logging.DEBUG)
    logging.basicConfig(
        level=logging_level,
        format="[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d] [%H:%M:%S",
        force=True,
    )
