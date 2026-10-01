# Trust in Neural Intrusion Detection: Auditing Post-hoc Explanation Reliability

This directory is a **local publication proposal** based on the repository tree
at `camera-ready-v1` for
[luisg98/auditing-explanation-reliability-nids](https://github.com/luisg98/auditing-explanation-reliability-nids).
The remote was confirmed private on 2026-10-01. This proposal has not been
approved or published. The exact versioned citation must be set only after the
release contents, data permissions, and Git history are cleared. The proposed
public snapshot uses an unrelated root commit with no inherited tags or commits;
the private source repository has not been rewritten.

The package contains the code, configurations, paper sources, and frozen
source-level numerical records used to rebuild Table II and Figures 1–2. The
records retain all rows, labels, targets, support counts, estimates, intervals,
and reported limitations. Direct row locators (`source_file`, `source_row`, and
`sample_order`) and raw/tensor fingerprints have been removed. `flow_id` values
are independent random 128-bit identifiers; the crosswalk was not retained.
These steps reduce direct linking but do not establish de-identification or
clear rights to redistribute row-level results. The per-flow records remain
blocked from release pending written data-rights clearance and review.

No raw traffic captures, prepared feature matrices, trained model binaries,
full attribution arrays, local environment snapshots, or historical manuscript
archive is included. Reconstructing the reported tables and figures uses the
included records and does not make model calls or retrain models. Fresh model
inference requires source data and checkpoints obtained under their
distributors' terms.

The MIT grant in `LICENSE` is limited to original software files in `src/`,
`scripts/`, `tests/`, and `configs/`. It does not license datasets, derived
records, tables, figures, or manuscript files. See `docs/data-access.md` for the
known dataset terms and unresolved permissions. The records cannot be published
until those permissions are explicit.

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
of data rights, linkability, Git history, and license scope. These are release
gates, not conclusions inferred from automated scans.

## Data access

Raw source datasets are not redistributed. See `docs/data-access.md` and the
paper's cited dataset references before obtaining them from their distributors.
