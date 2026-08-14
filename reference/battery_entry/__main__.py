"""__main__.py — makes ``python -m battery_entry <subcommand>`` work."""
from __future__ import annotations

import sys

from battery_entry.cli import main

if __name__ == "__main__":
    sys.exit(main())
