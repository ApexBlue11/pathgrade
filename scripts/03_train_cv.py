#!/usr/bin/env python
"""Step 3: cross-validated training. Never reads the locked test set."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathgrade.train import main

if __name__ == "__main__":
    raise SystemExit(main())
