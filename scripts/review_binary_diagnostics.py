#!/usr/bin/env python3
"""Bounded RQ2 tie/IG checks on the frozen 80-source binary panels.

No training or historical writes. Controls retain the historical seed streams
and are re-matched whenever a selected set changes. Model inference is batched;
every baseline AOPC must reproduce its stored per-source counterpart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tensorflow as tf
import numpy as np
import pandas as pd
from src.leakage_free_tabular import epistemic_audit as audit
from src.leakage_free_tabular_ext import study2_dependency_perturbations as ops
from src.leakage_free_tabular_ext.common import load_extension_context
from scripts.revision_reproduce import check_environment
from scripts.revision_statistics import crossed_ci, stable_seed as statistical_seed

OUT = ROOT / "results/reliability_diagnostics"
TASKS = ["cicids2017_binary", "ton_iot_binary"]
METHODS = ["gradient_x_input", "integrated_gradients", "occlusion", "lime", "shap"]
OPERATORS = ["training_median", "conditional_gaussian_mean", "joint_donor"]
FRACTIONS = np.array([.05, .1, .2, .3, .5])
INPUTS = {}


def record(path):
    path = Path(path)
    key = str(path.relative_to(ROOT))
    if key not in INPUTS:
        INPUTS[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def load_probe(context, task, family, seed, spec):
    directory = audit._probe_directory(context.paths, task, family, seed, spec)
    meta = json.loads(record(directory / "cache_metadata.json").read_text())
    for name in ["attributions.npz", "per_flow_index.csv", "probe_decision.json"]:
        path = record(directory / name)
        expected = next(r["sha256"] for r in meta["outputs"] if Path(r["path"]).name == name)
        assert INPUTS[str(path.relative_to(ROOT))] == expected, path
    index = pd.read_csv(directory / "per_flow_index.csv")
    with np.load(directory / "attributions.npz", allow_pickle=False) as z:
        for key in ["sample_order", "target_class_id"]:
            np.testing.assert_array_equal(z[key], index[key])
        attrs = z["attributions"].copy()
    mask = index.in_stochastic_subset.astype(str).str.lower().isin(["true", "1"])
    return index.loc[mask].reset_index(drop=True), attrs[mask]


def account(context, task, family, seed, method, steps=None):
    specs = audit.probe_specs(context.audit_config, method)
    if steps is not None:
        specs = [s for s in specs if s.steps == steps and s.reference == "training_median"]
    elif method != "lime":
        specs = [s for s in specs if s.role == "primary"]
    arrays = []
    identity = None
    for spec in specs:
        index, a = load_probe(context, task, family, seed, spec)
        if identity is not None:
            pd.testing.assert_frame_equal(index[["flow_id", "sample_order", "target_class_id"]],
                                          identity[["flow_id", "sample_order", "target_class_id"]])
        identity = index
        arrays.append(a)
    assert len(identity) == 16
    return identity, np.mean(np.stack(arrays), axis=0) if len(arrays) > 1 else arrays[0]


def ranking(attributes, policy, identities):
    """Stable historical order, or reverse/seeded order inside epsilon-tie blocks.

    Blocks join consecutive magnitude gaps <= epsilon. This also examines
    practically constant accounts; no records are excluded.
    """
    a = np.abs(np.asarray(attributes, dtype=float))
    order = np.argsort(-a, axis=1, kind="stable")
    if policy == "stable":
        return order
    for i, row in enumerate(a):
        eps = max(1e-12, 1e-8 * row.max())
        cuts = np.r_[0, np.flatnonzero(-np.diff(row[order[i]]) > eps) + 1, len(row)]
        rng = np.random.default_rng(audit.stable_seed(20260929, policy, identities[i]))
        for start, stop in zip(cuts[:-1], cuts[1:]):
            block = order[i, start:stop].copy()
            if policy == "reverse":
                block = block[::-1]
            else:
                rng.shuffle(block)
            order[i, start:stop] = block
    return order


def boundaries(index, a, task, family, seed, method):
    absolute = np.abs(a.astype(float))
    sorted_a = -np.sort(-absolute, axis=1)
    eps = np.maximum(1e-12, 1e-8 * absolute.max(axis=1))
    rows = []
    for u in FRACTIONS:
        k = int(np.ceil(u * a.shape[1]))
        frame = index[["flow_id", "sample_order", "target_class_id"]].copy()
        frame["boundary_gap"] = sorted_a[:, k-1] - sorted_a[:, k]
        frame["epsilon"] = eps
        frame["nonidentifiable"] = frame.boundary_gap <= eps
        frame["constant_account"] = np.ptp(absolute, axis=1) <= eps
        frame["nonzero_coordinates"] = (absolute > eps[:, None]).sum(axis=1)
        for key, value in dict(task_id=task, model=family, seed=seed, method=method,
                               fraction=u, k=k).items():
            frame[key] = value
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def evaluate(model, oc, index, order, task, family, seed, method):
    """Reuse operator construction/RNG verbatim; batch only model predictions."""
    X = np.asarray(oc.data.X_test[index.sample_order.to_numpy(int)], dtype=np.float32)
    targets = index.target_class_id.to_numpy(int)
    true = index.true_class_id.to_numpy(int)
    flow_ids = index.flow_id.astype(str).tolist()
    n = len(X)
    clean = audit.target_probabilities(model, X, targets)
    frames = []
    for operator in OPERATORS:
        reference = ops.unconditional_reference(operator, oc, X, targets,
            rng=np.random.default_rng(audit.stable_seed(20260908, "reference", task, family, seed, method, operator)))
        reference_p = audit.target_probabilities(model, reference, targets)
        displacement = np.abs(X - reference)
        curves = []
        for u in FRACTIONS:
            selected = order[:, :int(np.ceil(u * X.shape[1]))]
            selections = [selected]
            rngs = [np.random.default_rng(audit.stable_seed(20260908, "attribution", task, family, seed, method, operator, float(u)))]
            for rep in range(50):
                control_seed = audit.stable_seed(20260908, "control", task, family, seed, method, operator, float(u), rep)
                selections.append(audit._displacement_bin_matched_feature_matrix(displacement, selected,
                    bins=5, seed=control_seed, flow_ids=flow_ids))
                rngs.append(np.random.default_rng(audit.stable_seed(control_seed, "matched", "values")))
            inputs = []
            for s, rng in zip(selections, rngs):
                deletion_values = ops.operator_values(operator, oc, X, targets, true, s, rng=rng)
                inputs.append(ops._replace(X, s, deletion_values))
                complement = ops._complement(s, X.shape[1])
                insertion_values = ops.operator_values(operator, oc, X, targets, true, complement, rng=rng)
                inputs.append(ops._replace(X, complement, insertion_values))
            predictions = audit.target_probabilities(model, np.concatenate(inputs), np.tile(targets, len(inputs)))
            predictions = predictions.reshape(51, 2, n)
            effects = np.stack([clean - predictions[:, 0], predictions[:, 1] - reference_p], axis=1)
            curves.append(effects[0] - effects[1:].mean(axis=0))
        values = np.stack(curves, axis=-1)  # arm, source, fraction
        for arm_index, arm in enumerate(["deletion", "insertion"]):
            frame = index[["flow_id", "sample_order", "target_class_id"]].copy()
            frame["value"] = ops._normalized_area(FRACTIONS, values[arm_index])
            for key, value in dict(task_id=task, model=family, seed=seed, method=method,
                                   operator=operator, functional_check=arm).items():
                frame[key] = value
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def batched_ig(model, X, targets, reference, steps):
    """Same float32 trapezoid rule, evaluating independent path points in batches."""
    alphas = np.linspace(0, 1, steps+1, dtype=np.float32)
    points = reference[None] + alphas[:, None, None] * (X-reference)[None]
    flat = points.reshape(-1, X.shape[1])
    tiled_targets = np.tile(targets, steps+1)
    gradients = []
    for start in range(0, len(flat), 1024):
        x = tf.convert_to_tensor(flat[start:start+1024])
        with tf.GradientTape() as tape:
            tape.watch(x)
            score = audit._target_tensor(model(x, training=False), tiled_targets[start:start+1024])
        gradients.append(tape.gradient(score, x).numpy())
    g = np.concatenate(gradients).reshape(steps+1, *X.shape)
    total = np.zeros_like(X)
    for i in range(steps):
        total += .5 * (g[i] + g[i+1])
    return (X-reference) * total / float(steps)


def run(args):
    packages = check_environment()
    frozen_manifest = OUT / "diagnostic_manifest.json"
    if frozen_manifest.exists():
        if (args.tasks != TASKS or args.models != ["mlp", "cnn"]
                or args.seeds != list(range(42,47)) or args.methods != METHODS):
            raise RuntimeError("A completed release cannot be replaced by a partial run; use a new output version.")
        frozen = json.loads(frozen_manifest.read_text())
        for item in frozen["scripts"] + frozen["inputs"] + frozen["outputs"]:
            path = ROOT / item["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError("Cached review inputs/implementation/outputs changed: " + item["path"]
                    + ". Preserve this release and use a new version/output directory for a changed experiment.")
    context = load_extension_context(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in [context.extension_config_path, context.paths.config_path,
                 context.paths.training_config_path, ROOT / "requirements-lock.txt"]:
        record(path)
    for task in args.tasks:
        oc = ops.build_operator_context(context, task)
        for p in (context.paths.processed_root / task).glob("*"):
            if p.is_file():
                record(p)
        record(context.tables_root / ops.STUDY / "feature_groups.csv")
        for family in args.models:
            for seed in args.seeds:
                start = time.monotonic()
                dest = OUT / "cells" / task / family / f"seed_{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                model_path = record(ROOT / "results/leakage_free_tabular/final" / task / family / f"seed_{seed}/model.keras")
                model = tf.keras.models.load_model(model_path, compile=False)
                boundary_frames, checks = [], []
                for method in args.methods:
                    index, a = account(context, task, family, seed, method)
                    boundary_frames.append(boundaries(index, a, task, family, seed, method))
                    historical_path = record(context.results_root / ops.STUDY / "operators" / task / family / f"seed_{seed}" / method / "operator_aopc_per_flow.csv")
                    historical = pd.read_csv(historical_path)
                    for policy in ["stable", "reverse", "random_0", "random_1", "random_2"]:
                        output = dest / f"{method}_{policy}.csv"
                        order = ranking(a, policy, index.flow_id)
                        if output.exists():
                            result = pd.read_csv(output)
                        else:
                            result = evaluate(model, oc, index, order, task, family, seed, method)
                            result["variant"] = policy
                            result.to_csv(output, index=False)
                        if policy == "stable":
                            keys = ["flow_id", "sample_order", "target_class_id", "operator", "functional_check"]
                            merged = result.merge(historical, on=keys, validate="one_to_one")
                            assert len(merged) == 96
                            difference = np.max(np.abs(merged.value - merged.attribution_minus_random_aopc))
                            np.testing.assert_allclose(merged.value, merged.attribution_minus_random_aopc, atol=3e-6, rtol=0)
                            checks.append(dict(method=method, baseline_max_absolute_difference=float(difference)))
                        print(task, family, seed, method, policy, round(time.monotonic()-start, 1), flush=True)
                    if method == "integrated_gradients":
                        X = np.asarray(oc.data.X_test[index.sample_order.to_numpy(int)], dtype=np.float32)
                        targets = index.target_class_id.to_numpy(int)
                        reference = np.broadcast_to(oc.training_median, X.shape)
                        score_delta = audit.target_probabilities(model, X, targets) - audit.target_probabilities(model, reference, targets)
                        residual_frames = []
                        for steps in [64, 128, 256, 512, 1024, 2048]:
                            if steps <= 128:
                                idx, attrs = account(context, task, family, seed, method, steps)
                                pd.testing.assert_series_equal(idx.flow_id, index.flow_id)
                            else:
                                npz = dest / f"ig_{steps}.npz"
                                if npz.exists():
                                    attrs = np.load(npz)["attributions"]
                                else:
                                    attrs = batched_ig(model, X, targets, reference, steps)
                                    np.savez_compressed(npz, attributions=attrs, sample_order=index.sample_order, target_class_id=targets)
                            residual = np.abs(attrs.sum(axis=1, dtype=np.float64) - score_delta)
                            r = index[["flow_id", "sample_order", "target_class_id"]].copy()
                            r["absolute_residual"], r["steps"] = residual, steps
                            residual_frames.append(r)
                            if steps != 64:
                                output = dest / f"integrated_gradients_steps_{steps}.csv"
                                if not output.exists():
                                    result = evaluate(model, oc, index, ranking(attrs, "stable", index.flow_id), task, family, seed, method)
                                    result["variant"] = f"steps_{steps}"
                                    result.to_csv(output, index=False)
                            print(task, family, seed, "IG", steps, "max residual", float(residual.max()), flush=True)
                            # All rows at one resolution per fitted model. No hidden exclusion.
                            if steps >= 256 and residual.max() <= .01:
                                break
                        residuals = pd.concat(residual_frames, ignore_index=True).assign(task_id=task, model=family, seed=seed)
                        residuals.to_csv(dest / "ig_residuals.csv", index=False)
                pd.concat(boundary_frames).to_csv(dest / "boundaries.csv", index=False)
                (dest / "checks.json").write_text(json.dumps(checks, indent=2)+"\n")
                (dest / "inputs.json").write_text(json.dumps(INPUTS, indent=2)+"\n")
                tf.keras.backend.clear_session()
    (OUT / "diagnostic_environment.json").write_text(json.dumps(dict(packages=packages, python=sys.version,
        command=sys.argv, ig_residual_tolerance=.01, ig_max_steps=2048,
        tie_policies=["stable", "reverse", "random_0", "random_1", "random_2"],
        operator_draws="historical seed streams", panel="80 records per task; no exclusions"), indent=2)+"\n")


def summarize():
    check_environment()
    cells = OUT / "cells"
    checks_paths = sorted(cells.glob("*/*/seed_*/checks.json"))
    assert len(checks_paths) == 20, "All 20 fitted binary conditions must complete"
    checks = [c for p in checks_paths for c in json.loads(p.read_text())]
    assert len(checks) == 100
    assert all(c["baseline_max_absolute_difference"] <= 3e-6 for c in checks)
    bounds = pd.concat([pd.read_csv(p) for p in cells.glob("*/*/seed_*/boundaries.csv")], ignore_index=True)
    assert len(bounds) == 8000, len(bounds)
    bounds.to_csv(OUT / "rq2_boundaries.csv", index=False)
    bounds.groupby(["task_id", "model", "method", "fraction", "k"]).agg(
        records=("flow_id", "size"), nonidentifiable=("nonidentifiable", "sum"),
        fraction_nonidentifiable=("nonidentifiable", "mean"), constant_accounts=("constant_account", "sum")
    ).to_csv(OUT / "rq2_boundary_summary.csv")
    frames = [pd.read_csv(p) for p in cells.glob("*/*/seed_*/*.csv") if p.name not in ["boundaries.csv", "ig_residuals.csv"]]
    # Keep every source when a condition stops at its declared numerical
    # tolerance: the final-resolution contrast still has the full panel.
    for p in cells.glob("*/*/seed_*/ig_residuals.csv"):
        last_steps = int(pd.read_csv(p).steps.max())
        final = pd.read_csv(p.parent / f"integrated_gradients_steps_{last_steps}.csv")
        frames.append(final.assign(variant="final_resolution"))
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(OUT / "rq2_sensitivity_per_flow.csv.gz", index=False)
    rows, changes, contrasts = [], [], []
    keys = ["task_id", "model", "method", "operator", "functional_check"]
    for identity, group in detail.groupby(keys):
        base = dict(zip(keys, identity))
        stable = group[group.variant.eq("stable")]
        for variant, g in group.groupby("variant"):
            if len(g) != 80:
                continue  # adaptive resolutions are not full-panel estimands
            ci = crossed_ci(g, "value", 5000, statistical_seed("review", *identity, variant))
            rows.append(base | dict(variant=variant, **ci))
            paired = g.merge(stable, on=["seed", "flow_id", "target_class_id"], suffixes=("", "_stable"), validate="one_to_one")
            paired["difference"] = paired.value - paired.value_stable
            change = crossed_ci(paired, "difference", 5000, statistical_seed("review_change", *identity, variant))
            changes.append(base | dict(variant=variant, **change, max_source_change=float(paired.difference.abs().max())))
    for identity, group in detail.groupby(["task_id", "model", "method", "functional_check", "variant"]):
        base = dict(zip(["task_id", "model", "method", "functional_check", "variant"], identity))
        med = group[group.operator.eq("training_median")]
        for operator in OPERATORS[1:]:
            g = group[group.operator.eq(operator)]
            if len(g) != 80:
                continue
            p = g.merge(med, on=["seed", "flow_id", "target_class_id"], suffixes=("", "_median"), validate="one_to_one")
            p["difference"] = p.value - p.value_median
            ci = crossed_ci(p, "difference", 5000, statistical_seed("review_contrast", *identity, operator))
            contrasts.append(base | dict(operator=operator, **ci))
    pd.DataFrame(rows).to_csv(OUT / "rq2_sensitivity_summary.csv", index=False)
    pd.DataFrame(changes).to_csv(OUT / "rq2_sensitivity_changes.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "rq2_sensitivity_contrasts.csv", index=False)
    residuals = pd.concat([pd.read_csv(p) for p in cells.glob("*/*/seed_*/ig_residuals.csv")], ignore_index=True)
    residuals.to_csv(OUT / "ig_rq2_residuals.csv", index=False)
    residuals.groupby(["task_id", "model", "steps"]).agg(records=("flow_id", "size"),
        median=("absolute_residual", "median"), p95=("absolute_residual", lambda x:x.quantile(.95)),
        maximum=("absolute_residual", "max"), above_tolerance=("absolute_residual", lambda x:int((x>.01).sum()))
    ).to_csv(OUT / "ig_rq2_residual_summary.csv")
    case_rows = []
    case_keys = ["task_id", "model", "seed", "flow_id", "target_class_id", "functional_check"]
    for operator in OPERATORS[1:]:
        g = detail[detail.method.eq("integrated_gradients") & detail.variant.isin(["stable", "steps_128", "final_resolution"])]
        paired = g[g.operator.eq(operator)].merge(g[g.operator.eq("training_median")],
            on=case_keys+["variant"], suffixes=("", "_median"), validate="one_to_one")
        paired["operator_contrast"] = paired.value-paired.value_median
        cases = paired.pivot(index=case_keys, columns="variant", values="operator_contrast").reset_index()
        cases["operator"] = operator
        cases["change_128_vs_64"] = cases.steps_128-cases.stable
        cases["change_final_vs_64"] = cases.final_resolution-cases.stable
        case_rows.append(cases)
    cases = pd.concat(case_rows,ignore_index=True)
    residual_keys = ["task_id", "model", "seed", "flow_id", "target_class_id"]
    initial = residuals[residuals.steps.eq(64)][residual_keys+["absolute_residual"]].rename(columns={"absolute_residual":"residual_64"})
    final = residuals.sort_values("steps").drop_duplicates(residual_keys,keep="last")[residual_keys+["steps","absolute_residual"]]
    cases = cases.merge(initial,on=residual_keys,validate="many_to_one").merge(final,on=residual_keys,validate="many_to_one")
    cases["high_initial_residual"] = cases.residual_64 > .01
    cases.to_csv(OUT / "ig_case_diagnostics.csv",index=False)
    inputs = {}
    for p in cells.glob("*/*/seed_*/inputs.json"):
        for path, digest in json.loads(p.read_text()).items():
            assert path not in inputs or inputs[path] == digest
            inputs[path] = digest
    scripts = [Path(__file__).resolve(), ROOT / "scripts/revision_statistics.py",
               ROOT / "src/leakage_free_tabular/epistemic_audit.py",
               ROOT / "src/leakage_free_tabular_ext/study2_dependency_perturbations.py"]
    outputs = [p for p in OUT.rglob("*") if p.is_file() and "baseline" not in p.parts
               and p.name != "diagnostic_manifest.json"]
    manifest = dict(status="completed_directed_recheck", new_training=False,
        baseline_reconstructions=checks, records_per_task=80, exclusions=0,
        scripts=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in scripts],
        inputs=[dict(path=p, sha256=h) for p,h in sorted(inputs.items())],
        outputs=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(outputs)])
    (OUT / "diagnostic_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Verified 100 baseline reconstructions; summarized 8,000 fraction-level boundaries and full-panel sensitivities.", flush=True)


def verify_quadrature():
    """Check batched path evaluation against the actual stored 64-step accounts."""
    check_environment()
    context = load_extension_context(ROOT)
    rows = []
    for task in TASKS:
        prep = context.paths.processed_root / task
        test = np.load(prep / "X_test.npy", mmap_mode="r")
        baseline = np.load(prep / "baseline.npy")
        for family in ["mlp", "cnn"]:
            for seed in range(42,47):
                index, original = account(context, task, family, seed, "integrated_gradients", 64)
                model_path = ROOT / "results/leakage_free_tabular/final" / task / family / f"seed_{seed}/model.keras"
                model = tf.keras.models.load_model(model_path, compile=False)
                X = test[index.sample_order.to_numpy(int)].astype(np.float32)
                actual = batched_ig(model, X, index.target_class_id.to_numpy(int), np.broadcast_to(baseline, X.shape), 64)
                error = float(np.max(np.abs(actual-original)))
                np.testing.assert_allclose(actual, original, atol=2e-5, rtol=2e-5)
                rows.append(dict(task_id=task, model=family, seed=seed, steps=64,
                                 max_absolute_attribution_difference=error, records=len(index)))
                tf.keras.backend.clear_session()
    pd.DataFrame(rows).to_csv(OUT / "ig_batched_equivalence.csv", index=False)
    print("Batched quadrature verified against all 320 stored 64-step accounts; max attribution difference", max(r["max_absolute_attribution_difference"] for r in rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "summarize", "verify-quadrature"])
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--models", nargs="+", default=["mlp", "cnn"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42,47)))
    parser.add_argument("--methods", nargs="+", default=METHODS)
    args = parser.parse_args()
    if args.stage == "run":
        run(args)
    elif args.stage == "summarize":
        summarize()
    else:
        verify_quadrature()
