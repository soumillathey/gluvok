"""
Tests for the accuracy metrics (issue #3).

These matter more than they look. Every claim the pilot makes about accuracy is
computed here, so a bug in this module is a bug in the number you quote to a
customer — and unlike a pipeline bug, nothing else would ever contradict it.

Deliberately stdlib-only: app.eval.metrics imports nothing heavy, so these run
without torch, paddle or ultralytics.
"""

import csv

import pytest

from app.eval.metrics import (
    EXACT,
    FALSE_POSITIVE,
    MISS,
    SEED_MARKER,
    TRUE_NEGATIVE,
    UNLABELLED,
    WRONG_READ,
    Labels,
    character_error_rate,
    classify,
    compare_to_baseline,
    evaluate,
    format_report,
    levenshtein,
    load_labels,
    normalise_plate,
)

# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("RJ 14-GT.4976", "RJ14GT4976"),
        ("rj14gt4976", "RJ14GT4976"),
        ("  MH12AB1234  ", "MH12AB1234"),
        ("N/A", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_normalise_plate(raw, expected):
    """Formatting must never masquerade as model error."""
    assert normalise_plate(raw) == expected


def test_na_sentinel_is_treated_as_no_plate():
    """
    The pipeline emits the literal string 'N/A' for "no plate found". If that
    were treated as a prediction, every miss would score as a wrong read and the
    two would become indistinguishable.
    """
    assert normalise_plate("N/A") == ""
    assert classify("RJ14GT4976", "N/A") == MISS


# ---------------------------------------------------------------------------
# edit distance and CER
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("ABC", "ABC", 0),
        ("ABC", "ABD", 1),
        ("ABC", "AB", 1),
        ("", "ABC", 3),
        ("ABC", "", 3),
        ("RJ14GT4976", "RJ14GT4975", 1),
    ],
)
def test_levenshtein(a, b, expected):
    assert levenshtein(a, b) == expected


def test_cer_single_character_confusion():
    """The 0/O case. Small CER, so a near-miss reads as a near-miss."""
    assert character_error_rate("RJ09GA0165", "RJ09GA0I65") == pytest.approx(0.1)


def test_cer_can_exceed_one_for_long_garbage():
    """
    A 30-character watermark scored against a 10-character plate should look
    terrible. Clamping to 1.0 would hide how wrong it is.
    """
    assert character_error_rate("RJ14GT4976", "ALAMYIMAGEIDCRE1JFWWWALAMYCOM") > 1.0


def test_cer_zero_when_both_empty():
    assert character_error_rate("", "") == 0.0


# ---------------------------------------------------------------------------
# outcome classification — the core of the module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "truth,pred,expected",
    [
        ("RJ14GT4976", "RJ14GT4976", EXACT),
        ("RJ14GT4976", "RJ14GT4975", WRONG_READ),
        ("RJ14GT4976", "", MISS),
        ("RJ14GT4976", None, MISS),
        ("", "BP2A4904", FALSE_POSITIVE),
        ("", "", TRUE_NEGATIVE),
    ],
)
def test_classify(truth, pred, expected):
    assert classify(truth, pred) == expected


def test_unlabelled_is_excluded_not_counted_as_correct():
    """
    An unlabelled image must not silently count as a pass. Missing ground truth
    inflates accuracy if treated as anything other than excluded.
    """
    assert classify(None, "RJ14GT4976", labelled=False) == UNLABELLED


def test_watermark_case_is_a_false_positive():
    """
    The concrete regression: 01.jpg has no plate; the pipeline returned
    BP2A4904 from an Alamy watermark and the old eval counted it as SUCCESS.
    """
    assert classify("", "BP2A4904") == FALSE_POSITIVE


# ---------------------------------------------------------------------------
# aggregate scoring
# ---------------------------------------------------------------------------


@pytest.fixture
def scored():
    labels = Labels(
        plates={
            "01.jpg": "",  # no plate  -> pipeline returns one  -> FP
            "02.jpg": "RJ43GA2012",  # correct                            -> EXACT
            "03.jpg": "NL02K7556",  # correct                            -> EXACT
            "04.jpg": "OR02BU3389",  # one char off                       -> WRONG_READ
            "05.jpg": "KA25B3155",  # nothing returned                   -> MISS
            "06.jpg": "",  # no plate, nothing returned         -> TN
        }
    )
    rows = [
        {
            "filename": "01.jpg",
            "plate": "BP2A4904",
            "status": "SUCCESS",
            "provider": "paddleocr",
            "exec_time_ms": 2970.0,
        },
        {
            "filename": "02.jpg",
            "plate": "RJ43GA2012",
            "status": "SUCCESS",
            "provider": "paddleocr",
            "exec_time_ms": 246.0,
        },
        {
            "filename": "03.jpg",
            "plate": "NL02K7556",
            "status": "SUCCESS",
            "provider": "paddleocr",
            "exec_time_ms": 211.0,
        },
        {"filename": "04.jpg", "plate": "OR02BU3388", "status": "SUCCESS", "provider": "nvidia", "exec_time_ms": 450.0},
        {
            "filename": "05.jpg",
            "plate": "N/A",
            "status": "NO_PLATE_DETECTED",
            "provider": "N/A",
            "exec_time_ms": 5300.0,
        },
        {"filename": "06.jpg", "plate": "N/A", "status": "NO_PLATE_DETECTED", "provider": "N/A", "exec_time_ms": 300.0},
        {
            "filename": "99.jpg",
            "plate": "MH12AB1234",
            "status": "SUCCESS",
            "provider": "paddleocr",
            "exec_time_ms": 400.0,
        },
    ]
    return evaluate(rows, labels)


def test_outcome_counts(scored):
    assert scored.exact == 2
    assert scored.wrong_read == 1
    assert scored.miss == 1
    assert scored.false_positive == 1
    assert scored.true_negative == 1


def test_rates_are_computed_over_the_right_denominator(scored):
    """
    Exact match is over images that HAVE a plate (4 of them). False positive is
    over images that have NONE (2 of them). Mixing the denominators is the
    classic way to produce a flattering, meaningless number.
    """
    assert scored.images_with_plate == 4
    assert scored.images_without_plate == 2
    assert scored.exact_match_rate == pytest.approx(0.5)
    assert scored.false_positive_rate == pytest.approx(0.5)


def test_precision_counts_every_returned_plate(scored):
    """2 correct out of 4 plates returned (2 exact + 1 wrong read + 1 FP)."""
    assert scored.precision == pytest.approx(0.5)


def test_unlabelled_image_is_excluded_from_rates(scored):
    assert scored.total_images == 7
    assert scored.labelled == 6
    assert scored.unlabelled == 1


def test_latency_percentiles(scored):
    assert scored.latency_max_ms == 5300.0
    assert scored.latency_p50_ms > 0


def test_extraction_rate_would_have_flattered_this_run(scored):
    """
    The old metric counted 5 SUCCESS statuses out of 7 images -> "71%".
    True exact-match accuracy is 50%, and one of those successes was a
    watermark. This test documents the gap the whole issue is about.
    """
    old_style = scored.status_breakdown.get("SUCCESS", 0) / scored.total_images
    assert old_style > scored.exact_match_rate


# ---------------------------------------------------------------------------
# labels file
# ---------------------------------------------------------------------------


def _write_labels(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "true_plate", "notes"])
        writer.writerows(rows)


def test_load_labels(tmp_path):
    path = tmp_path / "labels.csv"
    _write_labels(
        path,
        [
            ["01.jpg", "", "no plate"],
            ["02.jpg", "rj43 ga2012", ""],
        ],
    )
    labels = load_labels(str(path))
    assert labels.plates["02.jpg"] == "RJ43GA2012"
    assert labels.without_plate == ["01.jpg"]
    assert labels.with_plate == ["02.jpg"]


def test_seeded_rows_are_tracked(tmp_path):
    """
    Rows pre-filled from model predictions and never human-verified must be
    counted, or the model ends up grading its own homework invisibly.
    """
    path = tmp_path / "labels.csv"
    _write_labels(
        path,
        [
            ["01.jpg", "RJ43GA2012", SEED_MARKER],
            ["02.jpg", "NL02K7556", ""],
        ],
    )
    labels = load_labels(str(path))
    assert labels.seeded == ["01.jpg"]

    metrics = evaluate(
        [
            {
                "filename": "01.jpg",
                "plate": "RJ43GA2012",
                "status": "SUCCESS",
                "provider": "paddleocr",
                "exec_time_ms": 100,
            }
        ],
        labels,
    )
    assert metrics.seeded_unverified == 1
    assert SEED_MARKER in format_report(metrics)


def test_missing_labels_file_raises_with_guidance(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_labels(str(tmp_path / "nope.csv"))
    assert "No labels at" in str(exc.value)


# ---------------------------------------------------------------------------
# regression gate
# ---------------------------------------------------------------------------


def test_accuracy_drop_fails(scored):
    baseline = {
        "exact_match_rate": 0.90,
        "false_positive_rate": 0.0,
        "precision": 0.90,
        "wrong_read_rate": 0.0,
        "mean_cer": 0.0,
    }
    verdict = compare_to_baseline(scored, baseline)
    assert not verdict.passed
    assert any("exact_match_rate" in f for f in verdict.failures)


def test_false_positive_rise_fails():
    labels = Labels(plates={"01.jpg": ""})
    metrics = evaluate(
        [
            {
                "filename": "01.jpg",
                "plate": "BP2A4904",
                "status": "SUCCESS",
                "provider": "paddleocr",
                "exec_time_ms": 100,
            }
        ],
        labels,
    )
    verdict = compare_to_baseline(metrics, {"false_positive_rate": 0.0})
    assert not verdict.passed


def test_improvement_passes_and_is_reported(scored):
    baseline = {
        "exact_match_rate": 0.10,
        "false_positive_rate": 0.90,
        "precision": 0.10,
        "wrong_read_rate": 0.90,
        "mean_cer": 2.0,
    }
    verdict = compare_to_baseline(scored, baseline)
    assert verdict.passed
    assert verdict.improvements


def test_noise_within_tolerance_passes(scored):
    baseline = {
        "exact_match_rate": scored.exact_match_rate + 0.005,
        "false_positive_rate": scored.false_positive_rate - 0.005,
        "precision": scored.precision + 0.005,
        "wrong_read_rate": scored.wrong_read_rate - 0.005,
        "mean_cer": scored.mean_cer - 0.005,
    }
    assert compare_to_baseline(scored, baseline).passed


def test_baseline_missing_fields_are_skipped(scored):
    assert compare_to_baseline(scored, {}).passed


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------


def test_report_names_the_dangerous_outcomes(scored):
    report = format_report(scored)
    assert "false positive" in report.lower()
    assert "wrong read" in report.lower()
    assert "worse than a miss" in report.lower()


def test_report_is_honest_when_nothing_is_labelled():
    metrics = evaluate(
        [
            {
                "filename": "x.jpg",
                "plate": "RJ14GT4976",
                "status": "SUCCESS",
                "provider": "paddleocr",
                "exec_time_ms": 100,
            }
        ],
        Labels(),
    )
    report = format_report(metrics)
    assert "extraction rate" in report.lower()
    assert "0.0%" not in report
