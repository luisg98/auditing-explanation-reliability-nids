"""Contracts for binary scope, pairing, support and interval provenance."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from scripts.build_isdfs_binary import binary, paired_contrast, verify_pairing, VALUE
from scripts.revision_statistics import crossed_ci

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper"
TASKS = {"cicids2017_binary", "ton_iot_binary"}


def test_binary_scope_does_not_accept_global_or_multiclass_rows():
    f = pd.DataFrame({"task_id": ["ALL", "cicids2017_multiclass", *sorted(TASKS)], "value": [999, 999, 1, 2]})
    assert binary(f).value.tolist() == [1, 2]


def test_equal_counts_are_not_evidence_of_pairing():
    f = pd.DataFrame({"seed": [42, 42], "flow_id": ["a", "b"], "target_class_id": [1, 1], "method": ["a", "b"]})
    with pytest.raises(AssertionError, match="identical flow"):
        verify_pairing(f, ["method"], "test")
    f.flow_id = "a"
    assert verify_pairing(f, ["method"], "test") == 1
    f.loc[1, "target_class_id"] = 0
    with pytest.raises(AssertionError, match="identical flow"):
        verify_pairing(f, ["method"], "test")


def test_operator_contrast_preserves_targets_and_source_pairing():
    rows = []
    for seed in range(42, 47):
        for source in ["a", "b"]:
            value = (seed - 42) / 5 + (source == "a")
            for operator, effect in [("training_median", value), ("joint_donor", value + .25)]:
                rows.append(dict(seed=seed, flow_id=source, target_class_id=1, operator=operator, **{VALUE: effect}))
    f = pd.DataFrame(rows)
    paired = paired_contrast(f, "joint_donor")
    ci = crossed_ci(paired, n_resamples=100, seed=1)
    assert ci["n_unique_sources"] == 2
    assert ci["n_seed_flow_cells"] == 10
    assert ci["estimate"] == pytest.approx(.25)
    assert ci["ci_low"] == pytest.approx(ci["ci_high"])
    f.loc[f.index[-1], "target_class_id"] = 0
    with pytest.raises(ValueError, match="Unpaired sources/targets"):
        paired_contrast(f, "joint_donor")


def test_published_inventory_is_exclusively_binary():
    for name, count in [("rq1_repeatability", 32), ("rq2_operator_effects", 120),
                        ("rq2_paired_operator_contrasts", 80), ("rq3_outcome_effects", 96)]:
        f = pd.read_csv(OUT / (name + ".csv"))
        assert len(f) == count
        assert set(f.task_id) == TASKS
        assert f.n_seeds.eq(5).all()
        assert (f.ci_low <= f.ci_high).all()
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert all(r["passed"] for r in manifest["checks"])
    assert manifest["resamples"] == 5000
    assert not manifest["new_training"] and not manifest["new_attributions"]


def test_lime_keeps_shared_run_intervals_and_shap_coverage_is_metric_specific():
    f = pd.read_csv(OUT / "rq1_repeatability.csv")
    lime = f[f.method.eq("lime")]
    assert lime.coverage.eq(1).all()
    assert lime.status.eq("available_verified").all()
    assert lime.uncertainty_scope.str.contains("shared-run jackknife").all()
    shap = f[f.method.eq("shap") & f.task_id.eq("ton_iot_binary") & f.model.eq("mlp")].set_index("metric")
    assert shap.loc["spearman", "coverage"] == 1
    assert shap.loc["jaccard_10", "coverage"] == pytest.approx(.45)
    assert shap.loc["jaccard_10", "estimate"] > .98
    assert shap.loc["jaccard_10", "available_seed_flow_cells"] == 80
    assert shap.loc["jaccard_10", "n_seed_flow_cells"] == 36


def test_deterministic_agreement_does_not_imply_complete_coverage():
    f = pd.read_csv(OUT / "rq1_repeatability.csv")
    f = f[f.method.eq("occlusion")]
    assert np.allclose(f.estimate, 1)
    assert f.coverage.min() < .47


def test_low_support_false_negatives_are_retained_but_not_promoted():
    f = pd.read_csv(OUT / "rq3_outcome_effects.csv")
    assert set(f.method) == {"gradient_x_input", "integrated_gradients", "occlusion"}
    low = f[f.task_id.eq("cicids2017_binary") & f.error_type.eq("false_negative")]
    assert len(low) == 12 and not low.inference_eligible.any()
    assert low[low.model.eq("mlp")].n_unique_sources.eq(13).all()
    assert low[low.model.eq("cnn")].n_unique_sources.eq(8).all()
    s = pd.read_csv(OUT / "rq3_support.csv")
    assert s.n_seed_flow_cells.sum() == 9017


def test_headline_operator_change_is_paired_not_an_averaged_interval():
    f = pd.read_csv(OUT / "rq2_paired_operator_contrasts.csv")
    selected = f[f.task_id.eq("ton_iot_binary") & f.method.eq("integrated_gradients")
                 & f.operator.eq("conditional_gaussian_mean") & f.functional_check.eq("insertion")].set_index("model")
    assert round(selected.loc["mlp", "estimate"], 3) == -.222
    assert round(selected.loc["cnn", "estimate"], 3) == -.152
    assert selected.ci_high.lt(0).all()


def test_unavailable_evidence_is_explicit_not_filled_from_aggregate_means():
    f = pd.read_csv(OUT / "traceability.csv")
    gaps = f[f.status.eq("unavailable")]
    assert len(gaps) == 4
    assert gaps.estimate.isna().all()
    assert set(gaps[gaps.rq.eq("RQ3")].method) == {"lime", "shap"}
