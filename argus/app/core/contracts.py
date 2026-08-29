"""
Explicit runtime contracts.

Standard Python `assert` statements are stripped when running under `python -O`.
`require()` for preconditions and `ensure()` for postconditions always raise
`ContractViolation` regardless of optimization flags. They carry component-specific
diagnostic messages and are catchable at API boundaries.

Contracts are placed where an unexpected value would otherwise pass silently
into downstream processing:
  - boundaries between components (YOLO -> cropping -> OCR -> API)
  - values derived from model output or third-party responses
  - index, slice, and loop bounds
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class ContractViolation(RuntimeError):
    """
    A precondition or postcondition failed.

    Subclasses RuntimeError rather than ANPRServiceError on purpose: this is an
    internal invariant break, not a domain error the caller did something to
    cause. It should surface as a 500 and be investigated, never be quietly
    mapped to a 400 and blamed on the client.
    """


def require(condition: Any, message: str) -> None:
    """
    Precondition. Raises ContractViolation when `condition` is falsy.

    Use at the entry of a function that is about to trust a value it did not
    create — model output, a parsed response, a caller-supplied box.
    """
    if not condition:
        raise ContractViolation(f"precondition failed: {message}")


def ensure(condition: Any, message: str) -> None:
    """
    Postcondition. Raises ContractViolation when `condition` is falsy.

    Use at the exit of a function whose output something downstream will trust
    without re-checking.
    """
    if not condition:
        raise ContractViolation(f"postcondition failed: {message}")


def bounded(items: Sequence[Any] | None, limit: int, what: str) -> Sequence[Any]:
    """
    Enforce fixed upper bounds on item sequences at the data source.

    Returns at most `limit` items. Truncation is a warning-level event, not an
    error: the caller asked for work we are declining to do without bound, and
    that decision should be visible in the logs rather than silent.

    Bounding the sequence rather than counting inside the loop keeps the bound
    stated once, next to the data, instead of repeated in every loop body where
    it can drift.
    """
    require(limit > 0, f"{what} limit must be positive, got {limit}")

    if not items:
        return []
    if len(items) <= limit:
        return items

    # Imported here to keep this module importable by anything, including the
    # logging configuration itself.
    from app.core.logging import logger

    logger.warning(
        f"[bounds] {what}: {len(items)} exceeds cap of {limit}; "
        f"processing first {limit} and discarding {len(items) - limit}."
    )
    return items[:limit]
