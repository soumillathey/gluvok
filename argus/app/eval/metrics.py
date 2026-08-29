"""
Accuracy metrics for the ANPR eval corpus (issue #3).

Deliberately dependency-free — stdlib only. This module must be unit-testable
without torch, paddle or ultralytics installed, because the metric definitions
are the part most worth testing and the part least related to inference.

What changed and why
--------------------
The previous eval reported a "success rate": the share of images where the
pipeline emitted something plate-shaped. That is an EXTRACTION RATE. It counted
an Alamy watermark read as `BP2A4904` as a success, because nothing recorded
what the right answer was.

These metrics are measured against ground truth in data/labels.csv, so a
confident wrong read counts against you rather than for you.

Outcome taxonomy
----------------
Every evaluated image falls into exactly one bucket:

    EXACT            plate expected, returned, correct
    WRONG_READ       plate expected, returned, incorrect      <- dangerous
    MISS             plate expected, nothing returned         <- merely unhelpful
    FALSE_POSITIVE   no plate expected, something returned    <- dangerous
    TRUE_NEGATIVE    no plate expected, nothing returned
    UNLABELLED       no ground truth; excluded from all rates

The distinction between WRONG_READ / FALSE_POSITIVE and MISS is the whole point.
At a weighbridge, a wrong plate on a weight record is worse than no plate,
because nothing downstream signals that anything went wrong. A single headline
"accuracy" number hides that difference; these do not.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

SEED_MARKER = "SEEDED-VERIFY"

EXACT = "EXACT"
WRONG_READ = "WRONG_READ"
MISS = "MISS"
FALSE_POSITIVE = "FALSE_POSITIVE"
TRUE_NEGATIVE = "TRUE_NEGATIVE"
UNLABELLED = "UNLABELLED"

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def normalise_plate(value: str | None) -> str:
    """
    Uppercase, strip everything that is not alphanumeric.

    'RJ 14-GT.4976' and 'rj14gt4976' both normalise to 'RJ14GT4976'. Applied to
    both prediction and label so formatting never masquerades as model error.
    The literal 'N/A' the pipeline emits for "no plate" normalises to empty.
    """
    if not value:
        return ""
    cleaned = _NON_ALNUM.sub("", str(value)).upper()
    return "" if cleaned in ("", "NA", "NONE", "NULL") else cleaned


def levenshtein(a: str, b: str) -> int:
    """Edit distance. Iterative two-row DP; plates are short, this is ample."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        current = [i]
        for j, ch_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ch_a != ch_b),  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(truth: str | None, prediction: str | None) -> float:
    """
    Edit distance normalised by the length of the truth.

    Can exceed 1.0 when the prediction is much longer than the truth — that is
    correct behaviour, not a bug. A 30-character watermark scored against a
    10-character plate should look terrible.

    CER is what shows near-misses. A one-character OCR confusion and a completely
    invented plate both count as "not exact"; only CER tells them apart, and they
    call for entirely different fixes.
    """
    truth, prediction = normalise_plate(truth), normalise_plate(prediction)
    if not truth:
        return 0.0 if not prediction else 1.0
    return levenshtein(truth, prediction) / len(truth)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


@dataclass
class Labels:
    plates: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    seeded: list[str] = field(default_factory=list)

    def __contains__(self, filename: str) -> bool:
        return filename in self.plates

    def __len__(self) -> int:
        return len(self.plates)

    @property
    def with_plate(self) -> list[str]:
        return [f for f, p in self.plates.items() if p]

    @property
    def without_plate(self) -> list[str]:
        """The false-positive test set: images known not to contain a readable plate."""
        return [f for f, p in self.plates.items() if not p]


def load_labels(path: str) -> Labels:
    """
    Read data/labels.csv. Missing file raises — callers decide whether to degrade.

    Rows still carrying the SEEDED-VERIFY marker were pre-filled from a model
    prediction and never confirmed by a human. They are loaded, but tracked, so
    the report can say how much of the "ground truth" is actually unverified.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No labels at '{path}'. Accuracy cannot be measured without ground truth."
        )

    labels = Labels()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            note = (row.get("notes") or "").strip()
            labels.plates[filename] = normalise_plate(row.get("true_plate"))
            labels.notes[filename] = note
            if SEED_MARKER in note.upper():
                labels.seeded.append(filename)
    return labels


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def classify(true_plate: str | None, predicted: str | None, labelled: bool = True) -> str:
    if not labelled:
        return UNLABELLED
    truth, pred = normalise_plate(true_plate), normalise_plate(predicted)
    if truth and pred:
        return EXACT if truth == pred else WRONG_READ
    if truth and not pred:
        return MISS
    if not truth and pred:
        return FALSE_POSITIVE
    return TRUE_NEGATIVE


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. No numpy dependency, and exact for small sets."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(pct / 100.0 * len(ordered) + 0.5) - 1))
    return ordered[idx]


@dataclass
class Metrics:
    total_images: int = 0
    labelled: int = 0
    unlabelled: int = 0
    seeded_unverified: int = 0

    images_with_plate: int = 0
    images_without_plate: int = 0

    exact: int = 0
    wrong_read: int = 0
    miss: int = 0
    false_positive: int = 0
    true_negative: int = 0

    exact_match_rate: float = 0.0
    wrong_read_rate: float = 0.0
    miss_rate: float = 0.0
    false_positive_rate: float = 0.0
    mean_cer: float = 0.0
    precision: float = 0.0

    status_breakdown: dict[str, int] = field(default_factory=dict)
    provider_breakdown: dict[str, int] = field(default_factory=dict)

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_max_ms: float = 0.0

    per_image: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(rows: Iterable[dict[str, Any]], labels: Labels) -> Metrics:
    """
    Score eval rows against ground truth.

    `rows` are eval_report.json records: filename, plate, status, provider,
    exec_time_ms.
    """
    m = Metrics()
    cers: list[float] = []
    latencies: list[float] = []

    for row in rows:
        filename = row.get("filename", "")
        predicted = row.get("plate")
        m.total_images += 1

        status = str(row.get("status", "UNKNOWN"))
        m.status_breakdown[status] = m.status_breakdown.get(status, 0) + 1

        provider = str(row.get("provider", "N/A"))
        m.provider_breakdown[provider] = m.provider_breakdown.get(provider, 0) + 1

        if row.get("exec_time_ms") is not None:
            latencies.append(float(row["exec_time_ms"]))

        is_labelled = filename in labels
        outcome = classify(labels.plates.get(filename), predicted, labelled=is_labelled)

        if not is_labelled:
            m.unlabelled += 1
        else:
            m.labelled += 1
            truth = labels.plates[filename]
            if truth:
                m.images_with_plate += 1
                cers.append(character_error_rate(truth, predicted))
            else:
                m.images_without_plate += 1

            if outcome == EXACT:
                m.exact += 1
            elif outcome == WRONG_READ:
                m.wrong_read += 1
            elif outcome == MISS:
                m.miss += 1
            elif outcome == FALSE_POSITIVE:
                m.false_positive += 1
            elif outcome == TRUE_NEGATIVE:
                m.true_negative += 1

            if filename in labels.seeded:
                m.seeded_unverified += 1

        m.per_image.append(
            {
                "filename": filename,
                "true_plate": labels.plates.get(filename, "") if is_labelled else None,
                "predicted": normalise_plate(predicted),
                "outcome": outcome,
                "cer": round(character_error_rate(labels.plates.get(filename, ""), predicted), 3)
                if is_labelled and labels.plates.get(filename)
                else None,
                "status": status,
                "provider": provider,
                "exec_time_ms": row.get("exec_time_ms"),
            }
        )

    if m.images_with_plate:
        m.exact_match_rate = m.exact / m.images_with_plate
        m.wrong_read_rate = m.wrong_read / m.images_with_plate
        m.miss_rate = m.miss / m.images_with_plate
    if m.images_without_plate:
        m.false_positive_rate = m.false_positive / m.images_without_plate
    if cers:
        m.mean_cer = sum(cers) / len(cers)

    # Of every plate the service returned, how many were right?
    # This is the number an operator experiences.
    returned = m.exact + m.wrong_read + m.false_positive
    if returned:
        m.precision = m.exact / returned

    m.latency_p50_ms = _percentile(latencies, 50)
    m.latency_p95_ms = _percentile(latencies, 95)
    m.latency_max_ms = max(latencies) if latencies else 0.0

    return m


# ---------------------------------------------------------------------------
# regression gate (consumed by CI, issue #11)
# ---------------------------------------------------------------------------


@dataclass
class RegressionVerdict:
    passed: bool
    failures: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)


def compare_to_baseline(current: Metrics, baseline: dict[str, Any], tolerance: float = 0.01) -> RegressionVerdict:
    """
    Fail if accuracy dropped or false positives rose beyond `tolerance`.

    Asymmetric on purpose. exact_match_rate falling is a regression; rising is
    not. false_positive_rate rising is a regression; falling is not. The
    tolerance absorbs noise on a small corpus without absorbing a real move.
    """
    verdict = RegressionVerdict(passed=True)

    def check(field_name: str, higher_is_better: bool) -> None:
        before = baseline.get(field_name)
        after = getattr(current, field_name, None)
        if before is None or after is None:
            return
        delta = after - before
        worse = -delta if higher_is_better else delta
        if worse > tolerance:
            verdict.passed = False
            verdict.failures.append(f"{field_name}: {before:.3f} -> {after:.3f} ({delta:+.3f})")
        elif -worse > tolerance:
            verdict.improvements.append(f"{field_name}: {before:.3f} -> {after:.3f} ({delta:+.3f})")

    check("exact_match_rate", higher_is_better=True)
    check("precision", higher_is_better=True)
    check("false_positive_rate", higher_is_better=False)
    check("wrong_read_rate", higher_is_better=False)
    check("mean_cer", higher_is_better=False)
    return verdict


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def format_report(m: Metrics) -> str:
    """Plain-text report. No tabulate dependency so CI can print it anywhere."""
    line = "=" * 68
    out = [line, "ACCURACY (measured against data/labels.csv)", line]

    if not m.labelled:
        out += [
            "",
            "  No labelled images matched this run.",
            "  Every 'success rate' without ground truth is an extraction rate,",
            "  not accuracy.",
            "",
            line,
        ]
        return "\n".join(out)

    out += [
        f"  corpus                    {m.total_images} images ({m.labelled} labelled, {m.unlabelled} unlabelled)",
        f"  with a legible plate      {m.images_with_plate}",
        f"  with no legible plate     {m.images_without_plate}   <- false-positive test set",
        "",
        "  ON IMAGES THAT HAVE A PLATE",
        f"    exact match             {m.exact:>4} / {m.images_with_plate}  {m.exact_match_rate:>7.1%}",
        f"    wrong read              {m.wrong_read:>4} / {m.images_with_plate}  {m.wrong_read_rate:>7.1%}   <- dangerous",
        f"    missed entirely         {m.miss:>4} / {m.images_with_plate}  {m.miss_rate:>7.1%}",
        f"    mean character error    {m.mean_cer:>7.3f}",
        "",
        "  ON IMAGES THAT HAVE NO PLATE",
        f"    false positive          {m.false_positive:>4} / {m.images_without_plate}  {m.false_positive_rate:>7.1%}   <- dangerous",
        f"    correctly returned none {m.true_negative:>4} / {m.images_without_plate}",
        "",
        f"  PRECISION (of plates returned, share correct)   {m.precision:>7.1%}",
        "",
        "  LATENCY",
        f"    p50 {m.latency_p50_ms:>10.1f} ms",
        f"    p95 {m.latency_p95_ms:>10.1f} ms",
        f"    max {m.latency_max_ms:>10.1f} ms",
        "",
        f"  pipeline status breakdown  {dict(sorted(m.status_breakdown.items()))}",
        f"  provider breakdown         {dict(sorted(m.provider_breakdown.items()))}",
    ]

    if m.unlabelled:
        out += [
            "",
            f"  NOTE: {m.unlabelled} images had no label and are excluded from every",
            "        rate above. Rates are computed only over labelled images.",
        ]

    if m.seeded_unverified:
        out += [
            "",
            f"  WARNING: {m.seeded_unverified} labels still carry the {SEED_MARKER}",
            "           marker. Those were pre-filled from model predictions and",
            "           never confirmed by a human. Any accuracy figure that",
            "           depends on them is partly the model grading itself.",
        ]

    if m.wrong_read or m.false_positive:
        out += [
            "",
            f"  {m.wrong_read + m.false_positive} images returned a plate that was wrong.",
            "  At a weighbridge these are worse than a miss: the weight record",
            "  gets a confident wrong plate and nothing downstream flags it.",
        ]

    out += [line]
    return "\n".join(out)


def format_worst_offenders(m: Metrics, limit: int = 15) -> str:
    """The rows to actually go and look at. Dangerous outcomes first."""
    priority = {FALSE_POSITIVE: 0, WRONG_READ: 1, MISS: 2}
    bad = [r for r in m.per_image if r["outcome"] in priority]
    if not bad:
        return ""
    bad.sort(key=lambda r: (priority[r["outcome"]], -(r["cer"] or 0)))

    line = "=" * 68
    out = [
        line,
        f"WORST OFFENDERS (top {min(limit, len(bad))} of {len(bad)})",
        line,
        f"  {'outcome':<15} {'file':<16} {'expected':<13} {'got':<13} {'cer':>5}",
    ]
    for r in bad[:limit]:
        cer = r["cer"]
        cer_text = f"{cer:.2f}" if cer is not None else "-"
        expected = r["true_plate"] or "-"
        got = r["predicted"] or "-"
        out.append(f"  {r['outcome']:<15} {str(r['filename'])[:15]:<16} {expected:<13} {got:<13} {cer_text:>5}")
    out.append(line)
    return "\n".join(out)
