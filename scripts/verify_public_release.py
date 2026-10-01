#!/usr/bin/env python3
"""Verify an explicit public-file manifest and inspect nested archives.

--candidate verifies an unpublished proposal. --release additionally requires
explicit per-file approval in PUBLICATION_MANIFEST.json.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PUBLICATION_MANIFEST.json"
PERSONAL_PATH = re.compile(rb"/(?:Users|home)/[^/\s\"'<>]+", re.I)
WINDOWS_PATH = re.compile(rb"\b[A-Z]:\\Users\\[^\\\s\"']+", re.I)
SECRET_PATTERNS = {
    "provider_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "cloud_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_literal": re.compile(
        rb"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[\"'][^\"']{12,}[\"']"
    ),
}
MANIFEST_NAME = "PUBLICATION_MANIFEST.json"
OVERLAP = 512
DIRECT_LINK_FIELDS = {
    "sample_order",
    "source_file",
    "source_row",
    "raw_fingerprint",
}
OPAQUE_FLOW_ID = re.compile(r"rid_[0-9a-f]{32}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_stream(stream, label: str, findings: set[tuple[str, str]]) -> None:
    tail = b""
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            break
        data = tail + block
        for rule, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                findings.add((label, rule))
        if PERSONAL_PATH.search(data):
            findings.add((label, "personal_absolute_path"))
        if WINDOWS_PATH.search(data):
            findings.add((label, "windows_personal_path"))
        tail = data[-OVERLAP:]


def scan_member(name: str, payload: bytes, findings: set[tuple[str, str]], depth: int = 0) -> None:
    if depth > 5:
        findings.add((name, "nested_archive_depth_limit"))
        return
    stream = io.BytesIO(payload)
    if zipfile.is_zipfile(stream):
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                with archive.open(info) as inner:
                    data = inner.read()
                scan_member(f"{name}!{info.filename}", data, findings, depth + 1)
        return
    stream.seek(0)
    try:
        with tarfile.open(fileobj=stream, mode="r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                inner = archive.extractfile(member)
                if inner is None:
                    continue
                data = inner.read()
                scan_member(f"{name}!{member.name}", data, findings, depth + 1)
        return
    except (tarfile.TarError, OSError):
        pass
    stream.seek(0)
    if name.lower().endswith(".gz"):
        try:
            with gzip.GzipFile(fileobj=stream) as uncompressed:
                scan_stream(uncompressed, name + "!decompressed", findings)
            return
        except (OSError, EOFError):
            pass
    stream.seek(0)
    scan_stream(stream, name, findings)


def scan_file(path: Path, rel: str, findings: set[tuple[str, str]]) -> None:
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if not info.is_dir():
                        with archive.open(info) as stream:
                            scan_member(f"{rel}!{info.filename}", stream.read(), findings)
            return
        try:
            with tarfile.open(path, mode="r:*") as archive:
                for member in archive.getmembers():
                    if member.isfile():
                        stream = archive.extractfile(member)
                        if stream is not None:
                            scan_member(f"{rel}!{member.name}", stream.read(), findings)
            return
        except (tarfile.TarError, OSError):
            pass
        if path.name.lower().endswith(".gz"):
            try:
                with gzip.open(path, "rb") as stream:
                    scan_stream(stream, rel + "!decompressed", findings)
                return
            except (OSError, EOFError):
                pass
        with path.open("rb") as stream:
            scan_stream(stream, rel, findings)
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as error:
        findings.add((rel, "unreadable_or_malformed_archive"))


def csv_linkability_findings(path: Path, rel: str) -> set[tuple[str, str]]:
    """Reject retained row locators/fingerprints and legacy flow identifiers."""
    if not (rel.lower().endswith(".csv") or rel.lower().endswith(".csv.gz")):
        return set()
    try:
        return csv_payload_linkability_findings(path.read_bytes(), rel)
    except OSError:
        return {(rel, "unreadable_csv_for_linkability_check")}


def csv_payload_linkability_findings(payload: bytes, rel: str) -> set[tuple[str, str]]:
    """Inspect a CSV blob, including compressed historical Git objects."""
    findings: set[tuple[str, str]] = set()
    try:
        binary_stream = io.BytesIO(payload)
        if rel.lower().endswith(".gz"):
            binary_stream = gzip.GzipFile(fileobj=binary_stream)
        with io.TextIOWrapper(binary_stream, encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            normalized = [column.strip().lower() for column in header]
            for column in normalized:
                if column in DIRECT_LINK_FIELDS or column.startswith("tensor_fingerprint"):
                    findings.add((rel, "direct_linkage_field:" + column))
            if "flow_id" in normalized:
                position = normalized.index("flow_id")
                for row in reader:
                    if position < len(row) and row[position]:
                        if not OPAQUE_FLOW_ID.fullmatch(row[position]):
                            findings.add((rel, "nonopaque_flow_id"))
                            break
    except (OSError, EOFError, UnicodeError, csv.Error, StopIteration):
        findings.add((rel, "unreadable_csv_for_linkability_check"))
    return findings


def tracked_history_csv_findings() -> tuple[int, set[tuple[str, str]]]:
    """Inspect every reachable CSV blob so removed files cannot hide in history."""
    listing = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    objects: dict[str, set[str]] = {}
    for line in listing:
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        oid, name = parts
        if name.lower().endswith(".csv") or name.lower().endswith(".csv.gz"):
            objects.setdefault(oid, set()).add(name)
    findings: set[tuple[str, str]] = set()
    for oid, names in objects.items():
        payload = subprocess.run(
            ["git", "cat-file", "blob", oid],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        for name in names:
            findings.update(csv_payload_linkability_findings(payload, name))
    return len(objects), findings


def tracked_history_findings() -> tuple[int, set[str]]:
    commits = subprocess.run(
        ["git", "rev-list", "--all"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.splitlines()
    regex = (
        r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"AKIA[0-9A-Z]{16}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|"
        r"/(Users|home)/[^/[:space:]]+"
    )
    result = subprocess.run(
        ["git", "grep", "-I", "-l", "-E", regex, *commits],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("Git history scan failed")
    files = set()
    for line in result.stdout.splitlines():
        # Git prefixes matches with the commit id; return only a repository path.
        pieces = line.split(":", 1)
        files.add(pieces[-1])
    return len(commits), files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--candidate", action="store_true", help="verify an unpublished proposal")
    mode.add_argument("--release", action="store_true", help="require explicit release approvals")
    parser.add_argument("--history", action="store_true", help="scan every reachable Git commit")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.release:
        if not manifest.get("release_authorized", False):
            raise SystemExit("Blocked: this manifest has not been authorized for release")
        gates = manifest.get("release_gates", {})
        uncleared = sorted(name for name in (
            "data_rights_approved",
            "linkability_review_cleared",
            "history_treatment_approved",
            "license_scope_confirmed",
        ) if not gates.get(name, False))
        if uncleared:
            raise SystemExit("Blocked: unresolved release gates: " + ", ".join(uncleared))
        pending = [row["path"] for row in manifest["files"] if row.get("status") != "approved"]
        if pending:
            raise SystemExit(f"Blocked: {len(pending)} manifest files are not approved")

    expected = {row["path"]: row for row in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise SystemExit("Duplicate file paths in publication manifest")
    control = {MANIFEST_NAME}
    actual = {}
    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        if path.is_symlink():
            raise SystemExit(f"Symlink is not allowed in the publication tree: {path.relative_to(ROOT)}")
        if path.is_file():
            actual[path.relative_to(ROOT).as_posix()] = path
    unexpected = sorted(set(actual) - set(expected) - control)
    missing = sorted(set(expected) - set(actual))
    if unexpected or missing:
        for name in unexpected[:50]:
            print("UNLISTED", name)
        for name in missing[:50]:
            print("MISSING", name)
        raise SystemExit(f"Manifest mismatch: {len(unexpected)} unlisted, {len(missing)} missing")

    findings: set[tuple[str, str]] = set()
    for rel, row in expected.items():
        path = actual[rel]
        size = path.stat().st_size
        if size != row["bytes"]:
            raise SystemExit(f"Size mismatch: {rel}")
        if sha256(path) != row["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {rel}")
        findings.update(csv_linkability_findings(path, rel))
        scan_file(path, rel, findings)
    scan_file(MANIFEST, MANIFEST_NAME, findings)
    if findings:
        for path, rule in sorted(findings):
            print("REVIEW", rule, path)
        raise SystemExit(f"Content scan found {len(findings)} review items")

    history_count = None
    if args.history:
        history_count, history_files = tracked_history_findings()
        if history_files:
            for name in sorted(history_files):
                print("HISTORY_REVIEW", name)
            raise SystemExit(f"History scan found matches in {len(history_files)} paths")
        history_csv_count, history_csv_findings = tracked_history_csv_findings()
        if history_csv_findings:
            for name, finding in sorted(history_csv_findings):
                print("HISTORY_LINKABILITY_REVIEW", finding, name)
            raise SystemExit(
                f"History CSV scan found direct linkage in {len(history_csv_findings)} path/rule pairs"
            )
    print(f"Manifest verified: {len(expected)} files; high-confidence content scan clear")
    if args.history:
        print(
            f"Reachable Git history text scan: {history_count} commits; no high-confidence matches "
            "(binary/archive blobs are outside this text-pattern check)"
        )
        print(
            f"Reachable Git CSV scan: {history_csv_count} unique blobs; no direct link fields "
            "or nonopaque flow IDs"
        )
    blockers = []
    if not manifest.get("release_authorized", False):
        blockers.append("release_authorized=false")
    gates = manifest.get("release_gates", {})
    blockers.extend(
        name + "=false"
        for name in (
            "data_rights_approved",
            "linkability_review_cleared",
            "history_treatment_approved",
            "license_scope_confirmed",
        )
        if not gates.get(name, False)
    )
    pending_count = sum(row.get("status") != "approved" for row in manifest["files"])
    if pending_count:
        blockers.append(f"{pending_count} file approval(s) pending")
    if blockers:
        print("Publication remains blocked: " + "; ".join(blockers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
