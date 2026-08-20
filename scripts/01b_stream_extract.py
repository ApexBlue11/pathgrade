#!/usr/bin/env python
"""Step 1b: stream slides from GDC, encode, discard. Use when you have no local WSIs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathgrade.preprocessing.stream_extract import main

if __name__ == "__main__":
    raise SystemExit(main())
