#!/usr/bin/env python3
"""Build a deterministic ZIP with embedded SHA-256 manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_NAMES = {".DS_Store", "Thumbs.db"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
GENERATED_DIRS = {".git", ".pytest_cache", "__pycache__", "build", "dist"}
FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def distributable_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if GENERATED_DIRS.intersection(relative.parts) or relative.parts[:2] == (
            "results", "reproduced"
        ):
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        if path.name in {"MANIFEST_SHA256.csv", "MANIFEST_SHA256.json"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = (args.output or (root.parent / f"{root.name}.zip")).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("Archive output must be outside the package root.")

    prefix = root.name
    payloads: list[tuple[str, bytes]] = []
    rows: list[dict[str, str | int]] = []
    for path in distributable_files(root):
        archive_name = f"{prefix}/{path.relative_to(root).as_posix()}"
        payload = path.read_bytes()
        payloads.append((archive_name, payload))
        rows.append(
            {
                "relative_path": archive_name,
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer, fieldnames=("relative_path", "size_bytes", "sha256")
    )
    writer.writeheader()
    writer.writerows(rows)
    csv_payload = csv_buffer.getvalue().encode("utf-8")
    json_payload = json.dumps(
        {
            "archive_format_version": 1,
            "package_root": prefix,
            "hash_algorithm": "SHA-256",
            "file_count": len(rows),
            "manifest_files_excluded_from_count": [
                f"{prefix}/MANIFEST_SHA256.csv",
                f"{prefix}/MANIFEST_SHA256.json",
            ],
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True
    ) as archive:
        for name, payload in payloads:
            archive.writestr(zip_info(name), payload)
        archive.writestr(zip_info(f"{prefix}/MANIFEST_SHA256.csv"), csv_payload)
        archive.writestr(zip_info(f"{prefix}/MANIFEST_SHA256.json"), json_payload)
    os.replace(temporary, output)

    archive_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Created {output}")
    print(f"Files listed in embedded manifest: {len(rows)}")
    print(f"Archive bytes: {output.stat().st_size}")
    print(f"Archive SHA-256: {archive_hash}")


if __name__ == "__main__":
    main()
