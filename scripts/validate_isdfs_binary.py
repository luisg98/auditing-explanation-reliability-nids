#!/usr/bin/env python3
"""Check binary release provenance, reported numbers and the compiled IEEE PDF."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper"
CHECKS = []


def check(name, condition, detail=""):
    CHECKS.append(dict(check=name, passed=bool(condition), detail=detail))
    if not condition:
        raise AssertionError((name, detail))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, decimals=3):
    return f"{value:.{decimals}f}".replace("-0.", "-.").removeprefix("0")


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    check("all reconstruction checks", all(r["passed"] for r in manifest["checks"]), str(len(manifest["checks"])))
    check("no new training or attributions", not manifest["new_training"] and not manifest["new_attributions"])
    check("5000 new bootstrap draws", manifest["resamples"] == 5000)
    for item in manifest["inputs"] + manifest["scripts"]:
        check("input hash " + item["path"], sha(ROOT / item["path"]) == item["sha256"])

    sections = {p.stem: p.read_text() for p in (ROOT / "latex/sections").glob("*.tex")}
    body = "\n".join(sections[k] for k in ["abstract", "introduction", "related_work", "methods", "results", "discussion_conclusions"])
    result_text = sections["results"]
    locked = (ROOT / "latex/research_question.txt").read_text().strip()
    check("current agreed RQ is verbatim", locked in " ".join(sections["introduction"].split()))
    check("three tables and two figures", body.count(r"\begin{table*}") == 2
          and body.count(r"\begin{table}") == 1 and body.count(r"\begin{figure*}") == 2)
    check("main paper does not depend on supplement", "Supplement~" not in body and "supplementary" not in body.lower())
    check("record-based release explained", "version-matched repository package" in body
          and "per-source records" in body)
    check("reproduction command documented", "python scripts/revision_reproduce.py binary" in (ROOT / "README.md").read_text())
    check("no old global denominators", not any(t in body for t in ["7,348", "14,696", "18,952", "11,620"]))

    predictions = pd.read_csv(OUT / "predictive_context.csv")
    for r in predictions.itertuples():
        if r.metric in ["f1", "fpr"]:
            check(f"predictive text {r.task_id}/{r.model}/{r.metric}", number(r.mean, 4) in result_text)
        elif r.metric == "threshold":
            digits = 5 if r.task_id == "cicids2017_binary" else 3
            check(f"threshold text {r.task_id}/{r.model}", number(r.mean, digits) in result_text)

    rq1 = pd.read_csv(OUT / "rq1_repeatability.csv")
    table = (ROOT / "latex/assets/binary_repeatability.tex").read_text()
    for r in rq1.itertuples():
        digits = 4 if r.method == "integrated_gradients" and r.metric == "spearman" else 3
        rendered = f"{r.estimate:.{digits}f} [{r.ci_low:.{digits}f}, {r.ci_high:.{digits}f}]"
        check(f"RQ1 table {r.task_id}/{r.model}/{r.method}/{r.metric}", rendered in table, rendered)
        check("RQ1 coverage " + str(r.Index), f"{100*r.coverage:.1f}" in table)

    rq2 = pd.read_csv(OUT / "rq2_operator_effects.csv")
    for r in rq2[rq2.task_id.eq("ton_iot_binary") & rq2.method.eq("integrated_gradients")
                 & rq2.functional_check.eq("insertion") & rq2.operator.ne("joint_donor")].itertuples():
        for key in ["estimate", "ci_low", "ci_high"]:
            token = number(getattr(r, key))
            check(f"IG insertion {r.model}/{r.operator}/{key}", token in result_text, token)
    contrasts = pd.read_csv(OUT / "rq2_paired_operator_contrasts.csv")
    selected = contrasts[(contrasts.task_id.eq("ton_iot_binary") & contrasts.method.eq("integrated_gradients")
                          & contrasts.operator.eq("conditional_gaussian_mean"))
                         | (contrasts.task_id.eq("cicids2017_binary") & contrasts.model.eq("cnn")
                            & contrasts.method.eq("integrated_gradients") & contrasts.operator.eq("joint_donor")
                            & contrasts.functional_check.eq("deletion"))]
    for r in selected.itertuples():
        for key in ["estimate", "ci_low", "ci_high"]:
            token = number(getattr(r, key))
            check(f"paired contrast text {r.task_id}/{r.model}/{r.functional_check}/{key}", token in result_text, token)

    rq3 = pd.read_csv(OUT / "rq3_outcome_effects.csv")
    low = rq3[rq3.task_id.eq("cicids2017_binary") & rq3.error_type.eq("false_negative")]
    check("CICIDS FN descriptive only", len(low) == 12 and not low.inference_eligible.any())
    support = pd.read_csv(OUT / "rq3_support.csv")
    check("binary outcome denominator", support.n_seed_flow_cells.sum() == 9017)
    trace = pd.read_csv(OUT / "traceability.csv")
    check("traceability is 328 estimates + 4 gaps", len(trace) == 332 and trace.estimate.notna().sum() == 328)
    check("source paths for each reported estimate", trace.loc[trace.estimate.notna(), "source"].notna().all())
    cost = pd.read_csv(OUT / "cost_microbenchmark.csv")
    cost_table = (ROOT / "latex/assets/binary_cost.tex").read_text()
    for r in cost.itertuples():
        for key in ["median_ms", "min_ms", "max_ms"]:
            token = f"{getattr(r, key):,.1f}"
            check("measured cost " + r.method + "/" + key, token in cost_table, token)
    check("microbenchmark units explicit", "one input per observation" in sections["discussion_conclusions"])

    review = ROOT / "results/reliability_diagnostics"
    baseline_checks = pd.read_csv(review / "baseline_reconstruction_checks.csv")
    check("all 100 source-level baseline rechecks", len(baseline_checks) == 100
          and baseline_checks.baseline_max_absolute_difference.le(3e-6).all())
    boundaries = pd.read_csv(review / "rq2_boundary_summary.csv")
    check("boundary audit for 100 task/model/method/fraction cells", len(boundaries) == 100 and boundaries.records.eq(80).all())
    equivalence = pd.read_csv(review / "ig_batched_equivalence.csv")
    check("batched IG verified on 320 cached accounts", len(equivalence) == 20 and equivalence.records.sum() == 320
          and equivalence.max_absolute_attribution_difference.lt(2e-5).all())
    provenance = pd.read_csv(review / "shap_stage_provenance.csv")
    check("historical SHAP provenance not inferred from current environment", len(provenance) == 40
          and provenance.historical_effective_l1_reg.eq("UNKNOWN").all()
          and provenance.historical_shap_version.eq("UNKNOWN").all())
    review_text = (ROOT / "latex/assets/binary_review.tex").read_text()
    sensitivity = pd.read_csv(review / "rq2_sensitivity_contrasts.csv")
    headline = sensitivity[sensitivity.task_id.eq("ton_iot_binary") & sensitivity.method.eq("integrated_gradients")
                           & sensitivity.operator.eq("conditional_gaussian_mean") & sensitivity.functional_check.eq("insertion")]
    final = headline[headline.variant.eq("final_resolution")]
    for r in final.itertuples():
        for key in ["estimate", "ci_low", "ci_high"]:
            check("final-resolution IG text " + r.model + "/" + key, number(getattr(r,key)) in review_text)
    check("IG contrast persists in both models and all tie orders", len(headline[headline.variant.isin(["reverse","random_0","random_1","random_2"])]) == 8
          and headline.ci_high.lt(0).all())
    residuals = pd.read_csv(review / "ig_case_diagnostics.csv").drop_duplicates(["task_id","model","seed","flow_id"])
    check("all 320 IG cases meet final tolerance", len(residuals) == 320 and residuals.absolute_residual.le(.01).all())
    changes = pd.read_csv(review / "rq2_sensitivity_changes.csv")
    for model,method,arm in [("mlp","shap","deletion"),("cnn","occlusion","insertion")]:
        row = changes[changes.task_id.eq("cicids2017_binary") & changes.model.eq(model) & changes.method.eq(method)
                      & changes.operator.eq("training_median") & changes.functional_check.eq(arm) & changes.variant.eq("reverse")].iloc[0]
        check("reported tie-order change " + method, number(row.estimate) in review_text)

    pdf = ROOT / "latex/paper.pdf"
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    check("exactly six pages including references", re.search(r"Pages:\s+6\b", info) is not None)
    check("US Letter", "612 x 792 pts (letter)" in info)
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    pages = [p for p in text.split("\f") if p.strip()]
    # A heading in the right column can share a text-extraction line with
    # body text in the left column. Inspect each separated column fragment.
    refs = [i for i, p in enumerate(pages) if any(re.sub(r"\s+", "", column).lower() == "references"
            for line in p.splitlines() for column in re.split(r"\s{3,}", line))]
    figures = [i for i, p in enumerate(pages) if "Fig. 1." in p or "Fig. 2." in p]
    check("both figures before references", len(refs) == 1 and len(figures) == 2 and max(figures) <= refs[0])
    check("authors visible", all(n in pages[0] for n in ["Luís Gonçalves", "Joaquim P. Silva", "Paulo Teixeira", "Joaquim Gonçalves"]))
    check("no placeholders", not any(token in text for token in ["TODO", "TBD", "??", "No Author Given"]))
    log = (ROOT / "latex/paper.log").read_text()
    check("no overflowing boxes or unresolved references", not any(t in log for t in [r"Overfull \hbox", r"Overfull \vbox", "undefined references", "Citation `"]))
    fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
    check("all PDF fonts embedded", all(re.search(r"\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", l) for l in fonts.splitlines()[2:]))
    # Record the exact local release, including inputs required to rebuild it.
    paths = [ROOT / "latex/paper.tex", ROOT / "latex/authors.tex", ROOT / "latex/research_question.txt", ROOT / "latex/references.bib", pdf,
             *[ROOT / ("latex/sections/" + k + ".tex") for k in ["abstract", "introduction", "related_work", "methods", "results", "discussion_conclusions"]],
             *sorted((ROOT / "latex/assets").glob("binary_*")), *sorted(OUT.glob("*.csv")), OUT / "manifest.json",
             review / "rq2_sensitivity_per_flow.csv.gz", review / "rq2_boundaries.csv",
             review / "ig_rq2_residuals.csv", review / "baseline_reconstruction_checks.csv",
             ROOT / "REPRODUCIBILITY.md", ROOT / "ARTIFACT_PROVENANCE.md"]
    report = {"status": "verified_local_release", "checks": CHECKS, "pages": len(pages),
              "external_submission": False, "public_artifact": False,
              "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in paths]}
    (OUT / "release_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(CHECKS)} binary release checks; six-page PDF, source and evidence hashes recorded.")


if __name__ == "__main__":
    main()
