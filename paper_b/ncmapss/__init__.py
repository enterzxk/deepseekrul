"""N-CMAPSS DS02 cross-dataset line for Paper B.

The package enforces the observability boundary documented in
``docs/paper_b_ncmapss/DATASET_PROTOCOL.md``: only ``W`` and ``X_s`` may enter a
feature matrix; ``X_v`` (virtual sensors), ``T`` (health parameters theta),
``Y`` (RUL) and the auxiliary metadata are never features.
"""
