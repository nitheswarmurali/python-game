"""Small utilities for files, timestamps, metrics, and report exports."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

LOGGER = logging.getLogger(__name__)


def timestamp() -> str:
    """Return a filesystem-friendly local timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directories(root: Path) -> None:
    """Create application data directories if they do not exist."""
    for name in ("images", "videos", "output", "models", "reports", "screenshots"):
        (root / name).mkdir(parents=True, exist_ok=True)


def export_report(path: Path, history: Iterable[Mapping[str, object]]) -> None:
    """Export detection history as a portable CSV report."""
    rows = list(history)
    if not rows:
        raise ValueError("There is no detection history to export.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "source", "faces", "fps"]
    with path.open("w", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def export_json(path: Path, payload: object) -> None:
    """Write analytics as readable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as report:
        json.dump(payload, report, indent=2, default=str)
