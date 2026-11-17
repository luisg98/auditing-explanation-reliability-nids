# Auditing Black-Box Explanation Reliability in Neural Intrusion Detection

Reproducibility snapshot for the paper "Auditing Black-Box Explanation Reliability in Neural Intrusion Detection Models." Contains code, configurations, and frozen numerical records sufficient to rebuild Table II and Figures 1–2.

This is not an immutable release. `PUBLICATION_MANIFEST.json` holds `release_authorized: false`, per-file approvals are pending, and there is no version tag or DOI. Do not cite the moving `main` branch as a fixed version.

## Reproduce

```bash
python scripts/verify_public_release.py --candidate --history
python scripts/revision_reproduce.py binary
python -m pytest -q tests/test_isdfs_binary.py tests/test_manuscript_rq_lock.py
```

Reconstruction uses included records only — no model calls, no retraining. See `REPRODUCIBILITY.md` for the distinction between record-based reconstruction and fresh experimental reruns.

## Scope

Raw captures, feature matrices, model binaries, and full attribution arrays are not included. The MIT license covers source files in `src/`, `scripts/`, `tests/`, and `configs/` only — not datasets, derived records, figures, or manuscript files. See `docs/data-access.md` for dataset terms.
