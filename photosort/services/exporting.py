"""Use case: dump the index in a format something else can read.

Building the records and serialising them are separate steps, so a future
consumer (an HTTP endpoint, a different format) reuses the first without
inheriting a file-writing side effect.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

from .context import AppContext

FORMATS = ("json", "csv")

#: Each CSV column and how to read it out of a record — most are a plain
#: lookup, `detections`/`metadata` need flattening to a single cell. One list
#: instead of a header row and a row-builder kept in sync by hand.
_CSV_FIELDS: tuple[tuple[str, Any], ...] = (
    ("id", lambda r: r["id"]),
    ("path", lambda r: r["path"]),
    ("category", lambda r: r["category"]),
    ("needs_review", lambda r: r["needs_review"]),
    ("width", lambda r: r["width"]),
    ("height", lambda r: r["height"]),
    ("taken_at", lambda r: r["taken_at"]),
    ("detections", lambda r: ";".join(
        f"{d['class']}:{d['confidence']:.2f}" for d in r["detections"]
    )),
    ("metadata", lambda r: json.dumps(r["metadata"], ensure_ascii=False) if r["metadata"] else ""),
)
CSV_COLUMNS = [name for name, _ in _CSV_FIELDS]


def records(ctx: AppContext, category: str | None = None) -> Iterator[dict[str, Any]]:
    """One dict per indexed image, metadata parsed and detections attached."""
    for row in ctx.storage.images.iter_export():
        if category and row["category"] != category:
            continue
        record = dict(row)
        record["metadata"] = json.loads(record["metadata"]) if record["metadata"] else None
        record["detections"] = [
            dict(d) for d in ctx.storage.detections.for_image(row["id"])
        ]
        yield record


def export(ctx: AppContext, output: Path, fmt: str = "json",
           category: str | None = None) -> int:
    """Write every record to `output`. Returns how many were written."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    rows = list(records(ctx, category))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        _write_json(rows, output)
    else:
        _write_csv(rows, output)
    return len(rows)


def _write_json(rows: list[dict[str, Any]], output: Path) -> None:
    """`rows` as one pretty-printed JSON array."""
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    """`rows` as CSV, one column per `_CSV_FIELDS` entry."""
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for record in rows:
            writer.writerow([read(record) for _name, read in _CSV_FIELDS])
