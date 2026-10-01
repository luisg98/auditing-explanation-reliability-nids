# Trust in Neural Intrusion Detection: Auditing Post-hoc Explanation Reliability

This is a **private publication candidate** based on the repository tree at
commit `7e926902a86de043c46ec95185b19c0c1e3ad3e3` for
[luisg98/auditing-explanation-reliability-nids](https://github.com/luisg98/auditing-explanation-reliability-nids).
The candidate is stored on the private remote's `main` branch and has not been
publicly released. It uses an unrelated root commit; the old release and tag
were removed from the remote, while the original workspace and its history were
left intact. GitHub still resolves the old commit by SHA, so the repository
must remain private until that residual exposure is assessed. Set the exact
versioned citation only after file approvals, linkability review, history
assessment, and license scope are settled.

The package contains the code, configurations, paper sources, and frozen
source-level numerical records used to rebuild Table II and Figures 1–2. The
records retain all rows, labels, targets, support counts, estimates, intervals,
and reported limitations. Direct row locators (`source_file`, `source_row`, and
`sample_order`) and raw/tensor fingerprints have been removed. `flow_id` values
are independent random 128-bit identifiers; the crosswalk was not retained.
These steps reduce direct linking but do not establish de-identification. The
package contains derived research records, not the source datasets or their
feature rows. The current rights review found no separate permission blocker
for those research outputs; upstream dataset terms and required citations are
documented in `docs/data-access.md`.

No raw traffic captures, prepared feature matrices, trained model binaries,
full attribution arrays, local environment snapshots, or historical manuscript
archive is included. Reconstructing the reported tables and figures uses the
included records and does not make model calls or retrain models. Fresh model
inference requires source data and checkpoints obtained under their
distributors' terms.

The MIT grant in `LICENSE` is limited to original software files in `src/`,
`scripts/`, `tests/`, and `configs/`. It does not license datasets, derived
records, tables, figures, or manuscript files. See `docs/data-access.md` for
the dataset terms and package scope. The separate license for paper materials
and project-generated records remains to be confirmed.

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
