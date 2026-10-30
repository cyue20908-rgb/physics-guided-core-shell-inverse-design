# -*- coding: utf-8 -*-
"""Verify a custom or GitHub/Zenodo reproducibility ZIP against SHA-256 manifests."""

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
        file_names = [info.filename for info in archive.infolist() if not info.is_dir()]
        embedded_csv = [
            name for name in file_names
            if name.endswith("/MANIFEST_SHA256.csv") or name == "MANIFEST_SHA256.csv"
        ]
        embedded_json = [
            name for name in file_names
            if name.endswith("/MANIFEST_SHA256.json") or name == "MANIFEST_SHA256.json"
        ]

        failures = []
        listed = set()
        ignored = set()
        if embedded_csv or embedded_json:
            if len(embedded_csv) != 1 or len(embedded_json) != 1:
                raise RuntimeError("Exactly one embedded CSV and JSON manifest are required.")
            csv_name = embedded_csv[0]
            json_name = embedded_json[0]
            manifest_json = json.loads(archive.read(json_name).decode("utf-8-sig"))
            reader = csv.DictReader(io.StringIO(archive.read(csv_name).decode("utf-8-sig")))
            rows = list(reader)
            if int(manifest_json.get("file_count", -1)) != len(rows):
                raise RuntimeError("CSV and JSON manifest counts disagree.")
            if set(reader.fieldnames or ()) != {"relative_path", "size_bytes", "sha256"}:
                raise RuntimeError(f"Unexpected CSV manifest columns: {reader.fieldnames}")
            checks = [
                (row["relative_path"], int(row["size_bytes"]), row["sha256"])
                for row in rows
            ]
            ignored = {csv_name, json_name}
            manifest_kind = "embedded archive manifest"
        else:
            package_candidates = [
                name for name in file_names
                if name.endswith("/reproducibility/manifests/package_checksums.csv")
                or name == "reproducibility/manifests/package_checksums.csv"
            ]
            if len(package_candidates) != 1:
                raise RuntimeError("Exactly one package checksum manifest is required.")
            csv_name = package_candidates[0]
            suffix = "reproducibility/manifests/package_checksums.csv"
            prefix = csv_name[:-len(suffix)]
            reader = csv.DictReader(io.StringIO(archive.read(csv_name).decode("utf-8-sig")))
            rows = list(reader)
            if set(reader.fieldnames or ()) != {"path", "bytes", "sha256"}:
                raise RuntimeError(f"Unexpected package checksum columns: {reader.fieldnames}")
            checks = [
                (f"{prefix}{row['path']}", int(row["bytes"]), row["sha256"])
                for row in rows
            ]
            ignored = {csv_name}
            manifest_kind = "package checksum manifest"

        for name, expected_size, expected_hash in checks:
            if name in listed:
                failures.append(f"duplicate manifest row: {name}")
                continue
            listed.add(name)
            if not safe_member(name):
                failures.append(f"unsafe manifest path: {name}")
                continue
            if name not in file_names:
                failures.append(f"missing: {name}")
                continue
            payload = archive.read(name)
            if len(payload) != expected_size:
                failures.append(f"size mismatch: {name}")
            if sha256_bytes(payload) != expected_hash:
                failures.append(f"sha256 mismatch: {name}")
        unexpected = set(file_names) - listed - ignored
        if unexpected:
            failures.extend(f"unexpected archive member: {name}" for name in sorted(unexpected))
        corrupt = archive.testzip()
        if corrupt is not None:
            failures.append(f"CRC failure: {corrupt}")
        if failures:
            raise RuntimeError("Archive verification failed:\n" + "\n".join(failures))

    print("Reproducibility archive verification PASSED")
    print(f"Archive: {args.archive}")
    print(f"Manifest type: {manifest_kind}")
    print(f"Manifest files verified: {len(rows)}")


if __name__ == "__main__":
    main()
