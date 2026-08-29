"""Evaluation utilities. Kept dependency-free so metrics can be tested without
the inference stack installed."""

from app.eval.metrics import (
    Labels,
    Metrics,
    RegressionVerdict,
    character_error_rate,
    classify,
    compare_to_baseline,
    evaluate,
    format_report,
    format_worst_offenders,
    levenshtein,
    load_labels,
    normalise_plate,
)

__all__ = [
    "Labels",
    "Metrics",
    "RegressionVerdict",
    "character_error_rate",
    "classify",
    "compare_to_baseline",
    "evaluate",
    "format_report",
    "format_worst_offenders",
    "levenshtein",
    "load_labels",
    "normalise_plate",
]
