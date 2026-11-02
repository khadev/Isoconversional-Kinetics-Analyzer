"""Allows running the package as ``python -m isokinetics``."""

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
