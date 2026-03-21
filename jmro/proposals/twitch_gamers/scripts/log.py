"""
Logging utilities for the Twitch Gamers exploration pipeline.

Usage:
    from log import log, section, step_timer

    section(1, "LOADING DATA")
    with step_timer("Loading edges"):
        ...
    log.info("some detail")
"""

import logging
import sys
import time

_start_time = time.time()


class _ElapsedFormatter(logging.Formatter):
    """Shows [MM:SS.ss] elapsed since script start alongside the standard fields."""

    def format(self, record):
        elapsed = time.time() - _start_time
        mins, secs = divmod(elapsed, 60)
        record.elapsed = f"{int(mins):02d}:{secs:05.2f}"
        return super().format(record)


def _build_logger() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ElapsedFormatter(
            fmt="%(asctime)s [%(elapsed)s] %(levelname)-5s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger = logging.getLogger("twitch_exploration")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False
    return logger


log = _build_logger()


def section(num: int, title: str) -> None:
    """Print a prominent section header."""
    log.info("=" * 60)
    log.info(f"{num}. {title}")
    log.info("=" * 60)


class step_timer:
    """Context manager that logs start / done with wall-clock time for a step.

    Example:
        with step_timer("Loading edges"):
            edges = pd.read_csv(...)
        # prints:  Loading edges... -> Loading edges done (1.2s)
    """

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        log.info(f"  {self.label}...")
        self._t0 = time.time()
        return self

    def __exit__(self, *_):
        dt = time.time() - self._t0
        log.info(f"  {self.label} done ({dt:.1f}s)")


def total_elapsed() -> str:
    """Return a human-readable string of total time since import."""
    t = time.time() - _start_time
    mins, secs = divmod(t, 60)
    return f"{int(mins)}m {secs:.1f}s"
