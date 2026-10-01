# Trust in Neural Intrusion Detection: Auditing Post-hoc Explanation Reliability

This GitHub repository is public and contains a working reproducibility snapshot. It is not yet an approved, immutable research release: `PUBLICATION_MANIFEST.json` keeps `release_authorized` false, file-level approvals are pending, and there is no version tag or DOI. Do not cite the moving `main` branch as a fixed version.

The package contains code, configurations, paper sources, and frozen numerical records used to rebuild Table II and Figures 1–2. The records retain reported rows, labels, targets, support counts, estimates, intervals, and limitations. Direct row locators (`source_file`, `source_row`, and `sample_order`) and raw/tensor fingerprints were removed. `flow_id` values were replaced with independent random 128-bit identifiers; the crosswalk was not retained. These changes do not prevent linkage through the remaining measurements.

A post-publication cross-check found an exact-value join between `results/source_records/statistics/repeatability_metric_fixed.csv.gz` and the historical `cross_probe_per_flow.csv.gz` table from commit [`7e926902a86de043c46ec95185b19c0c1e3ad3e3`](https://github.com/luisg98/auditing-explanation-reliability-nids/commit/7e926902a86de043c46ec95185b19c0c1e3ad3e3). It matched 4,246 rows across 436 current IDs; repeated matches narrowed two current IDs to a single historical locator each. This demonstrates linkage to legacy records, not identification of people. Treat these IDs as record keys, not anonymization. The linkability review remains unresolved, and the per-flow file should not be described as unlinkable.

The current `main` is a two-commit snapshot without the old history as an ancestor, but GitHub still serves the old commit and historical table by SHA without authentication. The old release and tag were removed; whether cached or cloned copies remain is unknown. This history treatment is unresolved.

The package contains derived research records, not the source datasets or their feature rows. The current data-rights review found no separate permission blocker for these outputs; upstream dataset terms and required citations are documented in `docs/data-access.md`. Public access to a benchmark dataset does not establish anonymity or grant reuse rights for the manuscript and research records.

No raw traffic captures, prepared feature matrices, trained model binaries, full attribution arrays, local environment snapshots, or historical manuscript archive is included. Reconstructing the reported tables and figures uses the included records and does not make model calls or retrain models. Fresh model inference requires source data and checkpoints obtained under their distributors' terms.

The MIT grant in `LICENSE` is limited to original software files in `src/`, `scripts/`, `tests/`, and `configs/`. It does not license datasets, derived records, tables, figures, or manuscript files. See `docs/data-access.md` for the dataset terms and package scope. The separate license for paper materials and project-generated records remains to be confirmed.

## Reproduce from included records

Use Python versions and packages from `requirements-lock.txt`; Tectonic is
needed to compile the manuscript PDF.

    python scripts/verify_public_release.py --candidate --history
    python scripts/revision_reproduce.py binary
    python scripts/reaggregate_review_diagnostics.py --check
    python -m pytest -q tests/test_isdfs_binary.py tests/test_manuscript_rq_lock.py tests/test_review_binary_diagnostics.py tests/test_release_linkability_guard.py

See `REPRODUCIBILITY.md` for the difference between record-based reconstruction
and fresh experimental reruns, and `ARTIFACT_PROVENANCE.md` for recorded,
reconstructed, and unknown settings.

Release mode requires explicit approval for every manifest entry and clearance
of linkability, Git history, and license scope. Dataset terms were reviewed for
the current derived-results-only package; adding benchmark source files would
require a new review.

## Data access

Raw source datasets are not redistributed. See `docs/data-access.md` and the
paper's cited dataset references before obtaining them from their distributors.
