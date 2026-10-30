#!/usr/bin/env python3
"""Regenerate the package-level SHA-256 checksum manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reproducibility" / "manifests" / "package_checksums.csv"
FORBIDDEN_NAMES = {".DS_Store", "Thumbs.db"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
GENERATED_DIRS = {".git", ".pytest_cache", "__pycache__", "build", "dist"}


def is_generated(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return bool(GENERATED_DIRS.intersection(relative.parts)) or relative.parts[:2] == (
        "results", "reproduced"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included_files(root: Path, output: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path == output:
            continue
        if is_generated(path, root):
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def included_git_paths(root: Path, output: Path, git_ref: str) -> list[str]:
    output_relative = output.relative_to(root).as_posix()
    raw = subprocess.check_output(
        ["git", "-C", str(root), "ls-tree", "-r", "-z", "--name-only", git_ref]
    )
    paths = []
    for value in raw.split(b"\0"):
        if not value:
            continue
        relative = value.decode("utf-8")
        parts = Path(relative).parts
        if relative == output_relative:
            continue
        if GENERATED_DIRS.intersection(parts) or parts[:2] == ("results", "reproduced"):
            continue
        path = Path(relative)
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        paths.append(path.as_posix())
    return sorted(paths)


def git_blob(root: Path, git_ref: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), "show", f"{git_ref}:{relative}"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--git-ref",
        default=None,
        help="Hash committed Git blob bytes from this ref instead of platform-dependent working-tree bytes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output or (root / "reproducibility" / "manifests" / "package_checksums.csv")).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError("The checksum manifest must be located inside the package root.") from exc

    rows = []
    if args.git_ref:
        for relative in included_git_paths(root, output, args.git_ref):
            payload = git_blob(root, args.git_ref, relative)
            rows.append(
                {
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        source = f"Git ref {args.git_ref}"
    else:
        for path in included_files(root, output):
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        source = "working tree"

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    print(f"Wrote {len(rows)} checksums from {source} to {output}")


if __name__ == "__main__":
    main()
