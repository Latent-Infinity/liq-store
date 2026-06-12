"""Module entry point so ``python -m liq.store`` invokes the CLI."""

from __future__ import annotations

import sys

from liq.store.cli import main

if __name__ == "__main__":
    sys.exit(main())
