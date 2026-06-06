"""Compatibility exports for script path constants.

New application code should import from ``cartwise.paths``. This module remains
for existing CLI and pipeline scripts during the migration.
"""

from cartwise.paths import (  # noqa: F401
    AMAZON_ESCI_RAW_ROOT,
    AMAZON_REVIEWS_RAW_ROOT,
    ARTIFACT_INDEXES_ROOT,
    ARTIFACT_PREVIEWS_ROOT,
    ARTIFACT_REPORTS_ROOT,
    ARTIFACTS_ROOT,
    DATA_ROOT,
    DEV_PROCESSED_ROOT,
    EVIDENCE_DENSE_ARTIFACT_ROOT,
    METRICS_ROOT,
    MODELS_ROOT,
    PRODUCT_BM25_ARTIFACT_ROOT,
    PRODUCT_DENSE_ARTIFACT_ROOT,
    PROJECT_ROOT,
    PROCESSED_ROOT,
    PROCESSED_ROOTS,
    RAW_DATA_ROOT,
    REPORTS_ROOT,
    RETRIEVAL_AUDIT_ARTIFACT_ROOT,
)
