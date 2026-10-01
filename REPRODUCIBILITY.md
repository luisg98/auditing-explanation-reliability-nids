# Reproducibility

## Reconstructing the paper from included records

The package includes the per-source/seed numerical inputs, analysis code, paper
source, figures, and the compiled PDF. In a matching environment, run:

    python scripts/revision_reproduce.py binary
    python scripts/reaggregate_review_diagnostics.py --check

The first command rebuilds the reported estimates, intervals, Table II,
Figures 1–2, and validation record from included numerical records. It does not
train a model or compute new attributions. The second recalculates the directed
RQ2 sensitivity summaries from the included source-level measurements and
checks them against the saved summaries.

## Re-running model inference or training

The package does not contain raw datasets, prepared feature matrices, model
checkpoints, or full attribution arrays. Re-running preprocessing, training,
or explainers therefore requires the matching source releases and substantial
compute. The relevant scripts and configurations are included, but a fresh
model run is a separate reproduction level and can produce a different
computational lineage. Do not describe the record-based reconstruction as a
fresh experimental rerun.

## Verification

    python scripts/verify_public_release.py --candidate --history
    python -m pytest -q tests/test_isdfs_binary.py tests/test_manuscript_rq_lock.py tests/test_review_binary_diagnostics.py

The public-release verifier checks the exact manifest, file sizes and SHA-256
digests, unexpected files, high-confidence secret patterns, personal absolute
paths, and nested compressed archives. It is an additional screening layer;
manual content and data-rights review remains required.

This proposal has no release tag or DOI. Before publication, approve each
conditional data record and resolve all release_gates in
PUBLICATION_MANIFEST.json. Then create the immutable repository version and
update CITATION.cff and the manuscript's artifact reference to that actual
version. `--release` refuses to proceed while any gate is false.
