from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = ROOT / "reproducibility" / "manifests" / "package_checksums.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_required_release_files_exist() -> None:
    required = (
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "pyproject.toml",
        "reproducibility/environment/environment.yml",
        "reproducibility/environment/requirements.txt",
        "data/dataset/multilayer_mie_dataset.npz",
        "scripts/verification/build_reproducibility_archive.py",
        "scripts/verification/verify_reproducibility_archive.py",
    )
    missing = [relative for relative in required if not (ROOT / relative).is_file()]
    assert not missing, f"Missing required release files: {missing}"


def test_checksum_manifest_excludes_generated_python_files() -> None:
    # Python may create bytecode during a local preflight. It must not be part
    # of the tracked/released file manifest, irrespective of local cache state.
    with CHECKSUM_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        paths = [row["path"] for row in csv.DictReader(handle)]
    forbidden = [
        path for path in paths
        if "__pycache__" in Path(path).parts or Path(path).suffix.lower() in {".pyc", ".pyo"}
    ]
    assert not forbidden, f"Generated Python files in release manifest: {forbidden}"


def test_public_configs_do_not_contain_absolute_local_paths() -> None:
    absolute_windows_path = re.compile(r"^[A-Za-z]:[\\/]")

    def values(value: object):
        if isinstance(value, dict):
            for item in value.values():
                yield from values(item)
        elif isinstance(value, list):
            for item in value:
                yield from values(item)
        else:
            yield value

    failures = []
    for path in sorted((ROOT / "configs").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for value in values(payload):
            if isinstance(value, str) and absolute_windows_path.match(value):
                failures.append(f"{path.relative_to(ROOT).as_posix()}: {value}")
    assert not failures, f"Public configs contain local absolute paths: {failures}"


def test_package_checksums() -> None:
    assert CHECKSUM_FILE.is_file()
    with CHECKSUM_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "Package checksum manifest is empty."

    listed = set()
    for row in rows:
        relative = Path(row["path"])
        normalized = relative.as_posix()
        assert normalized not in listed, f"Duplicate manifest row: {normalized}"
        listed.add(normalized)
        path = ROOT / relative
        assert path.is_file(), f"Manifest file is missing: {normalized}"
        assert path.stat().st_size == int(row["bytes"]), f"Size mismatch: {normalized}"
        assert sha256_file(path) == row["sha256"], f"SHA-256 mismatch: {normalized}"

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != CHECKSUM_FILE
        and path.suffix.lower() not in {".pyc", ".pyo"}
        and not {".git", ".pytest_cache", "__pycache__", "build", "dist"}.intersection(
            path.relative_to(ROOT).parts
        )
        and path.relative_to(ROOT).parts[:2] != ("results", "reproduced")
    }
    assert listed == actual, (
        f"Manifest differs from package files; missing={sorted(actual - listed)}, "
        f"unexpected={sorted(listed - actual)}"
    )
