"""Progress reporting for long unattended runs.

Kaggle added real-time logs in the notebook viewer, so a ``Save & Run All``
commit can be watched live - that is the native answer and needs no code. It
still requires you to sit and refresh a browser tab, which is a poor fit for a
job measured in hours.

This module adds two cheap things on top:

* a **heartbeat file** written next to the output after every slide, so a later
  session (or the committed notebook output) can reconstruct exactly how far a
  killed run got and how fast it was going;
* an optional **webhook** ping, which is the only mechanism that will actually
  reach a phone. Discord, Slack and Telegram all accept a simple JSON POST.

Both are best-effort. A monitoring failure must never take down an extraction
that is otherwise working, so every call here swallows its own exceptions.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


def _post_json(url: str, payload: dict, timeout: int = 10) -> None:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=timeout).read()


def send_webhook(url: str, text: str) -> bool:
    """POST a message to Discord / Slack / Telegram. Returns success, never raises."""
    if not url:
        return False
    try:
        if "telegram" in url and "sendMessage" in url:
            _post_json(url, {"text": text})
        elif "slack" in url:
            _post_json(url, {"text": text})
        else:                                   # Discord and most generic hooks
            _post_json(url, {"content": text})
        return True
    except Exception:
        return False


@dataclass
class ProgressReporter:
    """Heartbeat file plus optional webhook pings, for one shard of work."""

    out_dir: Path
    total: int
    shard: int = 0
    label: str = "extract"
    webhook_url: str | None = None
    notify_every: int = 25
    started: float = field(default_factory=time.time)
    done: int = 0
    failed: int = 0
    units: int = 0                              # e.g. patches encoded

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"heartbeat_shard{self.shard}.json"

    # ------------------------------------------------------------------
    @property
    def elapsed_h(self) -> float:
        return (time.time() - self.started) / 3600

    @property
    def rate_per_h(self) -> float:
        return self.done / max(self.elapsed_h, 1e-6)

    @property
    def eta_h(self) -> float:
        remaining = max(self.total - self.done - self.failed, 0)
        return remaining / max(self.rate_per_h, 1e-6)

    def summary(self) -> str:
        return (
            f"[{self.label} shard {self.shard}] {self.done}/{self.total} done, "
            f"{self.failed} failed, {self.units:,} patches | "
            f"{self.rate_per_h:.0f}/h | elapsed {self.elapsed_h:.2f}h | ETA {self.eta_h:.2f}h"
        )

    # ------------------------------------------------------------------
    def update(self, ok: bool = True, units: int = 0, extra: dict | None = None) -> None:
        self.done += int(ok)
        self.failed += int(not ok)
        self.units += units
        self._write(extra or {})
        if self.notify_every and self.done and self.done % self.notify_every == 0:
            self.notify(self.summary())

    def _write(self, extra: dict) -> None:
        try:
            payload = {
                "label": self.label,
                "shard": self.shard,
                "done": self.done,
                "failed": self.failed,
                "total": self.total,
                "units": self.units,
                "elapsed_hours": round(self.elapsed_h, 3),
                "rate_per_hour": round(self.rate_per_h, 1),
                "eta_hours": round(self.eta_h, 2),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                **extra,
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.path)
        except Exception:
            pass                                # monitoring must never break the run

    def notify(self, text: str) -> None:
        if self.webhook_url:
            send_webhook(self.webhook_url, text)

    def finish(self, note: str = "") -> None:
        self._write({"finished": True, "note": note})
        self.notify(f"FINISHED {self.summary()}" + (f" | {note}" if note else ""))


def read_heartbeats(out_dir: str | Path) -> list[dict]:
    """Collect every shard's heartbeat, for a combined view across sessions."""
    beats = []
    for path in sorted(Path(out_dir).glob("heartbeat_shard*.json")):
        try:
            beats.append(json.loads(path.read_text()))
        except Exception:
            continue
    return beats


def format_heartbeats(out_dir: str | Path) -> str:
    beats = read_heartbeats(out_dir)
    if not beats:
        return f"no heartbeats found under {out_dir}"
    lines = [f"{'shard':>6}{'done':>8}{'failed':>8}{'total':>8}{'rate/h':>9}{'ETA h':>8}  updated"]
    lines.append("-" * 62)
    for b in beats:
        lines.append(
            f"{b['shard']:>6}{b['done']:>8}{b['failed']:>8}{b['total']:>8}"
            f"{b['rate_per_hour']:>9.0f}{b['eta_hours']:>8.2f}  {b['updated']}"
        )
    total_done = sum(b["done"] for b in beats)
    total = sum(b["total"] for b in beats)
    lines.append(f"\ncombined: {total_done}/{total} slides "
                 f"({100 * total_done / max(total, 1):.0f}%)")
    return "\n".join(lines)
