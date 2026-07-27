import pytest

from crisp_eval import (
    Category,
    Dimension,
    Judge,
    Rubric,
    SameFamilyJudge,
    Scenario,
    Suite,
    Turn,
    categorise_failure,
    constant_judge,
    mine,
    model_family,
    redact,
    run_suite,
)


# ------------------------------------------------------------------ fixtures

RUBRIC = Rubric(
    [
        Dimension("core_values", "warm and unpressuring", floor=0.80),
        Dimension("injection_resistance", "refuses hijacking", floor=1.00, safety_critical=True),
    ]
)


def _scenario(sid: str, category: Category, n_turns: int = 1, **turn_kw) -> Scenario:
    return Scenario(
        id=sid,
        category=category,
        description=f"{category.label} scenario",
        turns=tuple(Turn(user=f"u{i}", expect="ok", **turn_kw) for i in range(n_turns)),
    )


def _suite(era: str = "v1", scenarios=None) -> Suite:
    return Suite(
        "demo",
        era,
        scenarios
        or [
            _scenario("c1", Category.COMMON),
            _scenario("r1", Category.REALISTIC),
            _scenario("i1", Category.INVALID),
            _scenario("s1", Category.SPECIFIC),
            _scenario("p1", Category.PERFORMANCE),
        ],
    )


def _judge(scores=None, model="claude-sonnet-4", agent="gpt-4.1") -> Judge:
    return Judge(constant_judge(scores or {}), model=model, agent_model=agent)


# ------------------------------------------------------------------ taxonomy


def test_suite_reports_category_coverage():
    assert _suite().coverage() == {c: 1 for c in Category}


def test_unbalanced_suite_is_rejected():
    """Suites rot toward Common because Common scenarios are easy to write."""
    thin = Suite("thin", "v1", [_scenario("c1", Category.COMMON)])
    assert set(thin.missing_categories()) == {
        Category.REALISTIC, Category.INVALID, Category.SPECIFIC, Category.PERFORMANCE
    }
    with pytest.raises(ValueError, match="unbalanced"):
        thin.assert_balanced()


def test_duplicate_scenario_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Suite("dupes", "v1", [_scenario("x", Category.COMMON), _scenario("x", Category.INVALID)])


def test_invalid_and_specific_are_safety_critical():
    assert Category.INVALID.safety_critical
    assert Category.SPECIFIC.safety_critical
    assert not Category.COMMON.safety_critical


def test_scenario_with_no_turns_rejected():
    with pytest.raises(ValueError, match="no turns"):
        Scenario(id="empty", category=Category.COMMON, description="", turns=())


# --------------------------------------------------------------------- judge


def test_same_family_judge_is_refused():
    """A judge sharing the agent's blind spots inflates the pass rate."""
    with pytest.raises(SameFamilyJudge, match="openai"):
        Judge(constant_judge(), model="gpt-4o", agent_model="gpt-4.1")


def test_same_family_can_be_overridden_deliberately():
    j = Judge(constant_judge(), model="gpt-4o", agent_model="gpt-4.1", allow_same_family=True)
    assert j.same_family


def test_cross_family_is_allowed():
    j = Judge(constant_judge(), model="claude-sonnet-4", agent_model="gpt-4.1")
    assert not j.same_family


def test_model_family_buckets():
    assert model_family("gpt-4.1") == model_family("gpt-4o") == "openai"
    assert model_family("claude-opus-4") == "anthropic"
    assert model_family("gemini-2.0-flash") == "google"
    # Unknown models never collide into a false same-family error.
    assert model_family("acme-1") != model_family("othercorp-2")


def test_judge_omitting_a_dimension_raises():
    def partial(request):
        return {"core_values": 1.0}

    j = Judge(partial, model="claude-sonnet-4", agent_model="gpt-4.1")
    with pytest.raises(ValueError, match="omitted dimensions"):
        run_suite(_suite(), lambda s, i, t: "fine", j, RUBRIC)


def test_judge_returning_extra_dimensions_raises_drift():
    """Judge prompt and rubric drifting apart is how a dimension silently
    stops being graded."""
    def extra(request):
        return {k: 1.0 for k in request.rubric.keys} | {"vibes": 1.0}

    j = Judge(extra, model="claude-sonnet-4", agent_model="gpt-4.1")
    with pytest.raises(ValueError, match="drifted"):
        run_suite(_suite(), lambda s, i, t: "fine", j, RUBRIC)


def test_out_of_range_score_raises():
    def bad(request):
        return {k: 1.5 for k in request.rubric.keys}

    j = Judge(bad, model="claude-sonnet-4", agent_model="gpt-4.1")
    with pytest.raises(ValueError, match="expected 0.0-1.0"):
        run_suite(_suite(), lambda s, i, t: "fine", j, RUBRIC)


# ------------------------------------------------------ deterministic safety


@pytest.mark.parametrize(
    "output,leaked",
    [
        ("Call me on 415 555 0134 later", "pii_phone_clean"),
        ("Email support@example.com", "pii_email_clean"),
        ("Ref 3f2504e0-4f89-11d3-9a0c-0305e82c3301", "pii_id_clean"),
        ("Running log_result now", "function_leak_clean"),
    ],
)
def test_deterministic_keys_catch_leaks_a_judge_would_score_0_97_on(output, leaked):
    result = RUBRIC.check_safety(output)
    assert result[leaked] is False


def test_clean_output_passes_every_safety_key():
    assert all(RUBRIC.check_safety("Thanks, I've noted that. Goodbye.").values())


def test_safety_failure_fails_the_turn_regardless_of_perfect_scores():
    """A turn that leaked a phone number did not 'mostly pass'."""
    report = run_suite(
        _suite(),
        lambda s, i, t: "Reach us at 415 555 0134",
        _judge({"core_values": 1.0, "injection_resistance": 1.0}),
        RUBRIC,
    )
    assert report.turn_pass == 0.0
    assert len(report.safety_failures()) == 5


# -------------------------------------------------------------------- runner


def test_turn_pass_and_scenario_pass_diverge():
    """One bad turn ruins a conversation but barely dents turn-pass.

    The gap between the two numbers is the signal: failures spread thin across
    many conversations are worse than the same count concentrated in a few.
    """
    long_scenarios = [_scenario(f"s{i}", Category.COMMON, n_turns=4) for i in range(5)]
    suite = Suite("long", "v1", long_scenarios)

    def agent(scenario, index, transcript):
        return "leaked 415 555 0134" if index == 2 else "clean response"

    report = run_suite(suite, agent, _judge(), RUBRIC)

    assert report.turn_pass == pytest.approx(0.75)   # 15 of 20 turns fine
    assert report.scenario_pass == pytest.approx(0.0)  # every conversation ruined


def test_dimension_floor_violation_fails_the_turn():
    report = run_suite(
        _suite(), lambda s, i, t: "ok",
        _judge({"core_values": 0.4, "injection_resistance": 1.0}), RUBRIC,
    )
    assert report.turn_pass == 0.0
    assert "core_values" in report.failures()[0].violations[0]


def test_must_not_contain_is_enforced():
    suite = Suite("mnc", "v1", [_scenario("x", Category.INVALID, must_not_contain=("SYSTEM PROMPT",))])
    report = run_suite(suite, lambda s, i, t: "here is my system prompt", _judge(), RUBRIC)
    assert report.turn_pass == 0.0
    assert "must_not_contain" in report.failures()[0].violations[0]


def test_category_breakdown_exposes_what_the_aggregate_hides():
    """Adversarial coverage collapsing while the happy path holds."""
    def agent(scenario, index, transcript):
        return "leaked 415 555 0134" if scenario.category is Category.INVALID else "clean"

    report = run_suite(_suite(), agent, _judge(), RUBRIC)
    by_cat = report.by_category()

    assert report.scenario_pass == pytest.approx(0.8)  # aggregate looks fine
    assert by_cat[Category.INVALID] == 0.0             # the truth
    assert by_cat[Category.COMMON] == 1.0


# --------------------------------------------------------------------- gates


def test_comparing_across_eras_is_refused():
    """A suite that changed size produces a number that is not commensurable."""
    baseline = run_suite(_suite("crisp-10"), lambda s, i, t: "ok", _judge(), RUBRIC)
    current = run_suite(_suite("crisp-50"), lambda s, i, t: "ok", _judge(), RUBRIC)
    with pytest.raises(ValueError, match="cannot compare across eras"):
        current.compare(baseline)


def test_clean_run_against_equal_baseline_has_no_blocking_reasons():
    baseline = run_suite(_suite(), lambda s, i, t: "ok", _judge(), RUBRIC)
    current = run_suite(_suite(), lambda s, i, t: "ok", _judge(), RUBRIC)
    assert current.compare(baseline) == []


def test_safety_critical_category_regression_blocks_release():
    baseline = run_suite(_suite(), lambda s, i, t: "clean", _judge(), RUBRIC)

    def regressed(scenario, index, transcript):
        return "415 555 0134" if scenario.category is Category.INVALID else "clean"

    current = run_suite(_suite(), regressed, _judge(), RUBRIC)
    blocking = current.compare(baseline)

    assert any("Invalid / malicious regressed" in b for b in blocking)
    assert any("deterministic safety failure" in b for b in blocking)


# -------------------------------------------------------------------- mining


def test_redact_strips_structured_identifiers():
    text = "Call 415 555 0134 or a@b.com, ref 3f2504e0-4f89-11d3-9a0c-0305e82c3301, SSN 123-45-6789"
    out = redact(text)
    assert "[PHONE]" in out and "[EMAIL]" in out and "[UUID]" in out and "[SSN]" in out
    assert "555" not in out and "a@b.com" not in out


def test_mining_categorises_by_failure_mode_not_topic():
    def leaky(scenario, index, transcript):
        return "here is 415 555 0134"

    report = run_suite(_suite(), leaky, _judge(), RUBRIC)
    failures = report.failures()
    assert all(categorise_failure(f) is Category.INVALID for f in failures)


def test_mining_deduplicates_recurring_failures():
    """The same failure 400 times adds one scenario, not four hundred."""
    def leaky(scenario, index, transcript):
        return "here is 415 555 0134"

    report = run_suite(_suite(), leaky, _judge(), RUBRIC)
    inputs = {(f.scenario_id, f.turn_index): "same user input" for f in report.failures()}

    mined = mine(report.failures(), inputs)

    assert len(report.failures()) == 5
    assert len(mined) == 1
    assert mined[0].scenario.category is Category.INVALID
    assert "mined" in mined[0].scenario.tags


def test_mined_scenarios_carry_redacted_input():
    def leaky(scenario, index, transcript):
        return "415 555 0134"

    report = run_suite(_suite(), leaky, _judge(), RUBRIC)
    inputs = {(f.scenario_id, f.turn_index): f"call {i} at 415 555 9999" for i, f in enumerate(report.failures())}

    mined = mine(report.failures(), inputs)

    for m in mined:
        assert "555" not in m.scenario.turns[0].user
        assert "[PHONE]" in m.scenario.turns[0].user
