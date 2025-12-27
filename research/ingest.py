"""Module entrypoint for research ingestion service.

Allows running the service as `python -m research.ingest` while keeping the
legacy script entrypoint `python research/ingest_service.py` working.
"""
from __future__ import annotations

import sys
from typing import Iterable

from research import ingest_service


def main(argv: Iterable[str] | None = None) -> None:
    """Invoke the research ingest CLI, optionally overriding argv."""
    if argv is None:
        ingest_service.main()
        return

    original_argv = sys.argv
    sys.argv = [original_argv[0], *argv]
    try:
        ingest_service.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":  # pragma: no cover - CLI shim
    main()
