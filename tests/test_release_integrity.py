from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = ROOT / "reproducibility" / "manifests" / "package_checksums.csv"


GENERATED_DIRS = {".git", ".pytest_cache", "__pycache__", "build", "dist"}


def included(relative: Path) -> bool:
    return (
        relative.as_posix() != "reproducibility/manifests/package_checksums.csv"
        and relative.suffix.lower() not in {".pyc", ".pyo"}
        and not GENERATED_DIRS.intersection(relative.parts)
        and relative.parts[:2] != ("results", "reproduced")
    )


def git_snapshot() -> dict[str, bytes] | None:
    try:
        inside = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if inside != "true":
        return None
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-tree", "-r", "-z", "--name-only", "HEAD"]
    )
    snapshot = {}
    for value in raw.split(b"\0"):
        if not value:
            continue
        relative = Path(value.decode("utf-8"))
        if not included(relative):
            continue
        snapshot[relative.as_posix()] = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"HEAD:{relative.as_posix()}"]
        )
    return snapshot


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

    snapshot = git_snapshot()
    if snapshot is None:
        snapshot = {
            path.relative_to(ROOT).as_posix(): path.read_bytes()
            for path in ROOT.rglob("*")
            if path.is_file() and included(path.relative_to(ROOT))
        }

    listed = set()
    for row in rows:
        normalized = Path(row["path"]).as_posix()
        assert normalized not in listed, f"Duplicate manifest row: {normalized}"
        listed.add(normalized)
        assert normalized in snapshot, f"Manifest file is missing: {normalized}"
        payload = snapshot[normalized]
        assert len(payload) == int(row["bytes"]), f"Size mismatch: {normalized}"
        assert hashlib.sha256(payload).hexdigest() == row["sha256"], (
            f"SHA-256 mismatch: {normalized}"
        )

    actual = set(snapshot)
    assert listed == actual, (
        f"Manifest differs from package files; missing={sorted(actual - listed)}, "
        f"unexpected={sorted(listed - actual)}"
    )


def test_release_versions_match() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = project["project"]["version"]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    package_init = (
        ROOT / "src" / "core_shell_optical_design" / "__init__.py"
    ).read_text(encoding="utf-8")
    citation_match = re.search(r"^version:\s*([^\s]+)\s*$", citation, re.MULTILINE)
    package_match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", package_init, re.MULTILINE
    )
    assert citation_match is not None
    assert package_match is not None
    assert citation_match.group(1) == project_version
    assert package_match.group(1) == project_version
