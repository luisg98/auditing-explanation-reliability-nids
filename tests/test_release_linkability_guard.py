from pathlib import Path

from scripts.verify_public_release import csv_linkability_findings


def test_csv_guard_rejects_source_locators_and_fingerprints(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    path.write_text(
        "flow_id,sample_order,tensor_fingerprint_exact\n"
        "rid_0123456789abcdef0123456789abcdef,12,deadbeef\n",
        encoding="utf-8",
    )

    findings = csv_linkability_findings(path, "records.csv")

    assert ("records.csv", "direct_linkage_field:sample_order") in findings
    assert ("records.csv", "direct_linkage_field:tensor_fingerprint_exact") in findings


def test_csv_guard_rejects_legacy_flow_ids(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    path.write_text("flow_id\ncase-000001\n", encoding="utf-8")

    assert csv_linkability_findings(path, "records.csv") == {
        ("records.csv", "nonopaque_flow_id")
    }


def test_csv_guard_accepts_frozen_random_ids(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    path.write_text(
        "flow_id,metric\nrid_0123456789abcdef0123456789abcdef,0.25\n",
        encoding="utf-8",
    )

    assert csv_linkability_findings(path, "records.csv") == set()
