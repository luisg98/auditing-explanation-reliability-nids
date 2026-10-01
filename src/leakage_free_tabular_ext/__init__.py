"""Research extensions to the frozen leakage-free epistemic audit.

Every module in this package treats
``configs/leakage_free_epistemic_audit.yaml`` and the frozen implementation in
``src/leakage_free_tabular`` as an immutable baseline.  Nothing here edits those
files: the baseline's content-addressed caches hash the bytes of
``epistemic_audit.py``, ``training.py``, ``cache.py``, ``src/models/*.py`` and
``src/utils/seed.py``, so a single edit there would invalidate every cached
attribution the manuscript rests on.
"""

EXTENSION_VERSION = "leakage_free_audit_extensions_v1"
