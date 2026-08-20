#!/usr/bin/env python
"""Tail a running Kaggle kernel, printing only lines that are new.

`kaggle kernels logs` returns the whole log as a JSON array every call, which
is unreadable to poll by hand. This diffs against what it has already shown, so
a long run reads like `tail -f` rather than a wall of repeats.

Useful because the pipeline flushes as it goes: a mistake an hour into
extraction is visible immediately instead of at the end of the session.

    python kaggle/watch.py apexblue/pathgrade-pipeline
    python kaggle/watch.py apexblue/pathgrade-pipeline --interval 120 --grep "slides/h"
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

TERMINAL = ("COMPLETE", "ERROR", "CANCEL")


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return (proc.stdout or "") + (proc.stderr or "")


def status(ref: str) -> str:
    out = run(["kaggle", "kernels", "status", ref])
    m = re.search(r"KernelWorkerStatus\.([A-Z]+)", out)
    return m.group(1) if m else "UNKNOWN"


def fetch(ref: str) -> list[dict]:
    out = run(["kaggle", "kernels", "logs", ref]).strip()
    start = out.find("[")
    if start < 0:
        return []
    try:
        return json.loads(out[start:])
    except json.JSONDecodeError:
        return []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref", help="owner/kernel-slug")
    p.add_argument("--interval", type=float, default=60.0, help="seconds between polls")
    p.add_argument("--grep", default=None, help="only print lines matching this regex")
    p.add_argument("--max-hours", type=float, default=10.0)
    args = p.parse_args()

    pattern = re.compile(args.grep) if args.grep else None
    seen = 0
    last_status = None
    deadline = time.time() + args.max_hours * 3600

    print(f"watching {args.ref} (every {args.interval:.0f}s)\n", flush=True)
    while time.time() < deadline:
        state = status(args.ref)
        if state != last_status:
            print(f"--- {time.strftime('%H:%M:%S')}  {state} ---", flush=True)
            last_status = state

        entries = fetch(args.ref)
        for entry in entries[seen:]:
            text = entry.get("data", "").rstrip("\n")
            if not text:
                continue
            if pattern and not pattern.search(text):
                continue
            mark = "!" if entry.get("stream_name") == "stderr" else " "
            print(f"{entry.get('time', 0):8.1f}s {mark} {text}", flush=True)
        seen = max(seen, len(entries))

        if state in TERMINAL:
            print(f"\n{args.ref} finished: {state}", flush=True)
            return 0 if state == "COMPLETE" else 1
        time.sleep(args.interval)

    print("watch timed out", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
