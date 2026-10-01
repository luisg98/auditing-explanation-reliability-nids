# Proposed package contents

PUBLICATION_MANIFEST.json records the exact candidate files, byte sizes,
SHA-256 digests, and review status. Every new file remains proposed until
explicitly approved. A file's presence in this local candidate is not approval
to publish it.

Proposed content includes the current paper source/PDF and figures, the code
and configurations used by the binary analyses, focused checks, provenance
documentation, and the per-source numerical records required to reconstruct
the article's estimates, intervals, table, and figures.

The result records retain all reported numerical rows and support information.
The direct row locators source_file, source_row, and sample_order and the
raw/tensor fingerprints were removed. All 12,774 distinct flow_id values were
replaced by random 128-bit IDs; the mapping was discarded. Groupings are
preserved and the record-based rebuild checks numerical results. This reduces
direct linkage but is not a de-identification guarantee. The current package
contains derived research results, not benchmark source rows; dataset terms
and required attribution are documented in docs/data-access.md. Linkability
review remains a separate release condition. The private remote uses an
unrelated root commit; GitHub still resolves the old commit by SHA, so the
repository remains private pending history assessment.

The proposal excludes raw data, fitted model files, full attribution arrays,
local machine/environment dumps, work logs, superseded manuscripts, unrelated
pipeline generations, caches, and temporary files. Individual CSV/JSON/PDF
files are included only when listed in the manifest.
