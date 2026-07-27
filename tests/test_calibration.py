import pytest

from crisp_eval import Label, calibrate, calibrate_all, cohens_kappa, interpret_kappa


def _labels(pairs, dimension="overall"):
    return [
        Label(item_id=f"i{n}", judge=j, human=h, dimension=dimension)
        for n, (j, h) in enumerate(pairs)
    ]


def test_perfect_agreement_on_mixed_labels_is_kappa_one():
    judge = [True, False, True, False, True]
    assert cohens_kappa(judge, list(judge)) == pytest.approx(1.0)


def test_total_disagreement_is_negative():
    judge = [True, True, False, False]
    human = [False, False, True, True]
    assert cohens_kappa(judge, human) < 0


def test_blind_majority_judge_scores_near_zero_despite_high_raw_agreement():
    """The whole reason kappa exists.

    Judge says "pass" to all 100 items. 95 really do pass. Raw agreement is
    95%, which looks excellent and carries no information whatsoever.
    """
    judge = [True] * 100
    human = [True] * 95 + [False] * 5

    report = calibrate(_labels(list(zip(judge, human))))

    assert report.raw_agreement == pytest.approx(0.95)
    assert report.kappa == pytest.approx(0.0, abs=1e-9)
    assert not report.gate_ready
    assert any("below the 0.61" in w for w in report.warnings)


def test_constant_agreement_on_a_single_label_returns_one_but_warns():
    judge = human = [True] * 60
    report = calibrate(_labels(list(zip(judge, human))))
    assert report.kappa == pytest.approx(1.0)
    # Degenerate, but the distribution is visible so a reader can catch it.
    assert report.judge_positive_rate == 1.0
    assert report.human_positive_rate == 1.0


def test_false_pass_is_flagged_as_the_dangerous_direction():
    """Judge approving things a human would fail is worse than the reverse."""
    pairs = [(True, False)] * 12 + [(True, True)] * 40 + [(False, False)] * 8
    report = calibrate(_labels(pairs))
    assert report.false_pass == 12
    assert report.false_fail == 0
    assert any("optimistic" in w for w in report.warnings)
    assert any("hiding failures" in w for w in report.warnings)


def test_small_sample_is_flagged_as_unstable():
    pairs = [(True, True), (False, False), (True, True)]
    report = calibrate(_labels(pairs))
    assert report.n == 3
    assert any("unstable" in w for w in report.warnings)
    assert not report.gate_ready


def test_gate_ready_requires_both_kappa_and_sample_size():
    # Strong agreement, balanced classes, 60 items.
    pairs = [(True, True)] * 28 + [(False, False)] * 28 + [(True, False)] * 2 + [(False, True)] * 2
    report = calibrate(_labels(pairs))
    assert report.n == 60
    assert report.kappa > 0.61
    assert report.gate_ready


def test_base_rate_divergence_is_flagged():
    pairs = [(True, False)] * 20 + [(True, True)] * 40 + [(False, False)] * 40
    report = calibrate(_labels(pairs))
    assert any("base rates diverge" in w for w in report.warnings)


def test_per_dimension_calibration_separates_good_and_bad_judges():
    """A judge is rarely uniformly good; an aggregate kappa hides which
    dimensions are actually measuring something."""
    factual = _labels([(True, True)] * 30 + [(False, False)] * 30, dimension="status_accuracy")
    vibes = _labels([(True, False)] * 15 + [(False, True)] * 15 + [(True, True)] * 30, dimension="core_values")

    reports = calibrate_all(factual + vibes)

    assert reports["status_accuracy"].kappa > 0.9
    assert reports["core_values"].kappa < 0.4
    assert reports["status_accuracy"].gate_ready
    assert not reports["core_values"].gate_ready


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        cohens_kappa([True, False], [True])


def test_zero_labels_raise():
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_interpretation_bands():
    assert interpret_kappa(-0.1) == "worse than chance"
    assert interpret_kappa(0.15) == "slight"
    assert interpret_kappa(0.35) == "fair"
    assert interpret_kappa(0.55) == "moderate"
    assert interpret_kappa(0.75) == "substantial"
    assert interpret_kappa(0.95) == "almost perfect"


def test_summary_is_human_readable():
    pairs = [(True, True)] * 30 + [(False, False)] * 30
    text = calibrate(_labels(pairs)).summary()
    assert "Cohen's kappa" in text
    assert "false passes" in text
    assert "gate ready" in text
