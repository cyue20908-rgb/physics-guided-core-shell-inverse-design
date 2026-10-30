# -*- coding: utf-8 -*-
"""Verify a reproducibility ZIP against its embedded SHA-256 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import PurePosixPath, Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a deterministic reproducibility archive")
    parser.add_argument("archive", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    with zipfile.ZipFile(args.archive, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Archive contains duplicate member names.")
        unsafe = [name for name in names if not safe_member(name)]
        if unsafe:
            raise RuntimeError(f"Unsafe archive members: {unsafe}")
        csv_candidates = [name for name in names if name.endswith("/MANIFEST_SHA256.csv") or name == "MANIFEST_SHA256.csv"]
        json_candidates = [name for name in names if name.endswith("/MANIFEST_SHA256.json") or name == "MANIFEST_SHA256.json"]
        if len(csv_candidates) != 1 or len(json_candidates) != 1:
            raise RuntimeError("Exactly one embedded CSV and JSON manifest are required.")
        csv_name = csv_candidates[0]
        json_name = json_candidates[0]

        manifest_json = json.loads(archive.read(json_name).decode("utf-8-sig"))
        reader = csv.DictReader(io.StringIO(archive.read(csv_name).decode("utf-8-sig")))
        rows = list(reader)
        if int(manifest_json.get("file_count", -1)) != len(rows):
            raise RuntimeError("CSV and JSON manifest counts disagree.")
        expected_fields = {"relative_path", "size_bytes", "sha256"}
        if set(reader.fieldnames or ()) != expected_fields:
            raise RuntimeError(f"Unexpected CSV manifest columns: {reader.fieldnames}")

        failures = []
        listed = set()
        for row in rows:
            name = row["relative_path"]
            if name in listed:
                failures.append(f"duplicate manifest row: {name}")
                continue
            listed.add(name)
            if not safe_member(name):
                failures.append(f"unsafe manifest path: {name}")
                continue
            if name not in names:
                failures.append(f"missing: {name}")
                continue
            payload = archive.read(name)
            if len(payload) != int(row["size_bytes"]):
                failures.append(f"size mismatch: {name}")
            if sha256_bytes(payload) != row["sha256"]:
                failures.append(f"sha256 mismatch: {name}")
        unexpected = set(names) - listed - {csv_name, json_name}
        if unexpected:
            failures.extend(f"unexpected archive member: {name}" for name in sorted(unexpected))
        corrupt = archive.testzip()
        if corrupt is not None:
            failures.append(f"CRC failure: {corrupt}")
        if failures:
            raise RuntimeError("Archive verification failed:\n" + "\n".join(failures))

    print("Reproducibility archive verification PASSED")
    print(f"Archive: {args.archive}")
    print(f"Manifest files verified: {len(rows)}")


if __name__ == "__main__":
    main()
