"""GDC query and streaming download for TCGA slides.

Downloading ~470 diagnostic slides means moving roughly half a terabyte, which
does not fit on a Kaggle worker. The pipeline therefore never holds more than a
couple of slides at once: fetch one, encode it, delete it, move on.

Nothing here needs authentication - TCGA diagnostic slide images are open
access. Controlled-access projects would need a token, which this module does
not handle on purpose.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

# TCGA barcodes look like TCGA-BA-4078-01Z-00-DX1; the patient is the first
# three fields, which is the granularity the grade labels use.
def patient_id_from_barcode(barcode: str) -> str:
    return "-".join(barcode.split("-")[:3])


@dataclass
class SlideRecord:
    file_id: str
    file_name: str
    file_size: int
    patient_id: str
    md5: str | None = None

    @property
    def slide_barcode(self) -> str:
        return self.file_name.split(".")[0]

    @property
    def is_diagnostic(self) -> bool:
        """DX = formalin-fixed diagnostic slide; TS/BS = frozen section.

        Grading must use diagnostic slides - frozen sections have freezing
        artefact that changes nuclear morphology, which is the very thing the
        grade depends on.
        """
        return "-DX" in self.slide_barcode.upper()


def query_slides(
    project_id: str = "TCGA-HNSC",
    diagnostic_only: bool = True,
    size: int = 5000,
    timeout: int = 120,
) -> list[SlideRecord]:
    """Query the GDC files endpoint for slide images in a project."""
    content = [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
        {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
    ]
    if diagnostic_only:
        content.append(
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}}
        )

    payload = {
        "filters": {"op": "and", "content": content},
        "fields": "file_id,file_name,file_size,md5sum",
        "format": "JSON",
        "size": str(size),
    }
    req = urllib.request.Request(
        GDC_FILES_ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        hits = json.loads(resp.read().decode())["data"]["hits"]

    records = [
        SlideRecord(
            file_id=h["file_id"],
            file_name=h["file_name"],
            file_size=int(h.get("file_size", 0)),
            patient_id=patient_id_from_barcode(h["file_name"]),
            md5=h.get("md5sum"),
        )
        for h in hits
    ]
    if diagnostic_only:
        records = [r for r in records if r.is_diagnostic]
    return sorted(records, key=lambda r: (r.patient_id, r.file_name))


def one_slide_per_patient(records: list[SlideRecord]) -> list[SlideRecord]:
    """Keep the smallest diagnostic slide per patient.

    Smallest rather than first: slide size is dominated by scanned area, and the
    extra area in the largest slide is rarely extra tumour. This is the single
    cheapest way to cut total download volume roughly in half.
    """
    best: dict[str, SlideRecord] = {}
    for r in records:
        cur = best.get(r.patient_id)
        if cur is None or r.file_size < cur.file_size:
            best[r.patient_id] = r
    return [best[k] for k in sorted(best)]


def shard(records: list, index: int, count: int) -> list:
    """Deterministic interleaved shard, so parallel sessions cover disjoint slides.

    Interleaving rather than contiguous blocks keeps each shard's total bytes
    similar even when file sizes are ordered.
    """
    if count <= 1:
        return records
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} out of range for {count} shards")
    return records[index::count]


def download_slide(
    record: SlideRecord,
    dest_dir: str | Path,
    chunk_mb: int = 8,
    retries: int = 3,
    verify_md5: bool = False,
    timeout: int = 300,
) -> Path:
    """Stream one slide to disk. Returns the local path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / record.file_name
    partial = dest.with_suffix(dest.suffix + ".partial")
    url = f"{GDC_DATA_ENDPOINT}/{record.file_id}"

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            digest = hashlib.md5() if verify_md5 else None
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(partial, "wb") as fh:
                while True:
                    chunk = resp.read(chunk_mb * 1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    if digest is not None:
                        digest.update(chunk)

            if record.file_size and partial.stat().st_size != record.file_size:
                raise IOError(
                    f"size mismatch: got {partial.stat().st_size}, expected {record.file_size}"
                )
            if digest is not None and record.md5 and digest.hexdigest() != record.md5:
                raise IOError("md5 mismatch")

            partial.replace(dest)
            return dest
        except Exception as e:                       # network flakiness is expected
            last_error = e
            partial.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))

    raise IOError(f"failed to download {record.file_name} after {retries} attempts: {last_error}")


def free_disk_gb(path: str | Path = ".") -> float:
    usage = os.statvfs(path) if hasattr(os, "statvfs") else None
    if usage is not None:
        return usage.f_bavail * usage.f_frsize / 1e9
    import shutil

    return shutil.disk_usage(path).free / 1e9


def summarise(records: list[SlideRecord]) -> str:
    total_gb = sum(r.file_size for r in records) / 1e9
    patients = len({r.patient_id for r in records})
    mean_gb = total_gb / max(len(records), 1)
    return (
        f"{len(records)} slides / {patients} patients / {total_gb:.0f} GB "
        f"(mean {mean_gb:.2f} GB per slide)"
    )
