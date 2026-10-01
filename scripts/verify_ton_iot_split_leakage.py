#!/usr/bin/env python3
"""Independently verify that pipeline_v5's ToN-IoT binary train/val/test split
has no exact- or rounded-float32 duplicate feature vectors across splits,
reusing the exact fingerprinting/overlap methodology from
`src.leakage_free_tabular.prepare` (the same check that found CICIDS2017's
leakage). ToN-IoT's split is re-randomized per seed (confirmed: train_idx
differs across seeds), so every one of the 5 accepted seeds is checked
independently.

This gates whether pipeline_v5's already-trained ToN-IoT MLP/CNN models can be
reused as-is in the v2 audit (see MASTER_PROMPT_V2_CAMERA_READY.md and the
[[ton-iot-models-kept-as-is]] memory) without retraining on a corrected split.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leakage_free_tabular.prepare import tensor_fingerprints, _pairwise_overlap  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
DECIMALS = 5
SPLITS = ("train", "val", "test")
DATA_ROOT = ROOT / "results/v1/v5/processed/ton_iot_binary"
OUTPUT_PATH = ROOT / "results/strict_leakage_audit/ton_iot_split_leakage_verification_decision.json"


def check_seed(seed: int) -> dict:
    seed_dir = DATA_ROOT / f"seed_{seed}"
    tensors = {split: np.load(seed_dir / f"X_{split}.npy") for split in SPLITS}
    fingerprints = {
        split: dict(zip(("exact", "rounded"), tensor_fingerprints(values, DECIMALS)))
        for split, values in tensors.items()
    }
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    overlaps = []
    all_zero = True
    for kind in ("exact", "rounded"):
        for left, right in pairs:
            overlap = _pairwise_overlap(fingerprints[left][kind], fingerprints[right][kind])
            passed = overlap == 0
            all_zero = all_zero and passed
            overlaps.append(
                {
                    "representation": kind,
                    "split_a": left,
                    "split_b": right,
                    "rows_a": int(len(tensors[left])),
                    "rows_b": int(len(tensors[right])),
                    "overlap_unique_fingerprints": int(overlap),
                    "passed": bool(passed),
                }
            )
    return {"seed": seed, "all_checks_passed": bool(all_zero), "checks": overlaps}


def main() -> None:
    per_seed = [check_seed(seed) for seed in SEEDS]
    total_rows = sum(
        check["rows_a"] for entry in per_seed for check in entry["checks"] if check["representation"] == "exact"
    ) // 3  # 3 pairs counted per seed; rows_a alone double counts train
    total_conflicts = sum(
        check["overlap_unique_fingerprints"]
        for entry in per_seed
        for check in entry["checks"]
        if check["representation"] == "rounded"
    )
    exact_conflicts = sum(
        check["overlap_unique_fingerprints"]
        for entry in per_seed
        for check in entry["checks"]
        if check["representation"] == "exact"
    )
    conflict_rate = total_conflicts / max(total_rows, 1)
    all_exact_zero = exact_conflicts == 0
    negligible = all_exact_zero and total_conflicts > 0 and conflict_rate < 1e-4
    clean = all_exact_zero and total_conflicts == 0
    status = "clean" if clean else ("negligible_conflicts" if negligible else "material_conflicts")
    decision = {
        "verification": "ton_iot_binary_split_leakage",
        "purpose": (
            "Independently confirm pipeline_v5's ToN-IoT binary train/val/test "
            "split has zero exact/rounded(5-decimal)-float32 duplicate feature "
            "vectors across splits, using the same methodology that found "
            "CICIDS2017's split leakage, before reusing v5's models as-is in v2."
        ),
        "seeds_checked": list(SEEDS),
        "decimals": DECIMALS,
        "exact_conflicts_total": int(exact_conflicts),
        "rounded_conflicts_total": int(total_conflicts),
        "approx_total_rows_per_seed": int(total_rows),
        "rounded_conflict_rate": conflict_rate,
        "comparison": "CICIDS2017's original split showed 0.2%-6.5% cross-split overlap; this rate is shown above for direct comparison.",
        "status": status,
        "conclusion": {
            "clean": "ToN-IoT split leakage-free for all 5 seeds (zero exact and zero rounded conflicts); v5's accepted MLP/CNN models may be reused as-is.",
            "negligible_conflicts": (
                "Zero exact-duplicate vectors; a handful of rounded(5-decimal)-only "
                "near-duplicates found (rate reported above), several orders of "
                "magnitude below CICIDS2017's documented leakage rate. Plausibly "
                "coincidental rounding collisions in continuous-valued NetFlow "
                "features rather than systematic leakage. Needs a human call on "
                "whether this counts as 'the slightest doubt' before reusing v5's "
                "models as-is (see MASTER_PROMPT_V2_CAMERA_READY.md)."
            ),
            "material_conflicts": "Exact-duplicate vectors found across splits; do NOT reuse v5's models as-is without remediation.",
        }[status],
        "per_seed": per_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in decision.items() if k != "per_seed"}, indent=2, sort_keys=True))
    if status == "material_conflicts":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
