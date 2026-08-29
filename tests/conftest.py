"""Put ``src/`` on the import path once, for every test file.

Each test module already does this insert itself, which works when the whole
suite runs because an alphabetically earlier file happens to execute first.
It does not work when a single file is run on its own -
``pytest tests/test_packaging.py`` failed with ModuleNotFoundError purely
because of collection order. pytest imports conftest before any test module,
so doing it here makes every file independently runnable and removes the
ordering dependency.

The insert is deliberately *not* an installed-package lookup: tests should
exercise the working tree, so that a source change is covered without a
reinstall.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
