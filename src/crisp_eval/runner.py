"""Running a suite, and the two numbers that come out of it.

Turn-pass vs scenario-pass
--------------------------
These are different metrics and reporting only one of them is how eval work
goes wrong.

    turn-pass       = passing turns / total turns
    scenario-pass   = scenarios where EVERY turn passed / total scenarios

Turn-pass is generous and moves smoothly, which makes it good for tracking
progress. Scenario-pass is brutal and moves in steps, which makes it the one
that reflects user experience: a conversation with one bad turn in the middle
is a bad conversation, not an 83% good one.

The gap between them is itself diagnostic. Turn-pass 87% with scenario-pass
66% means failures are *spread thin* across many conversations rather than
concentrated in a few. That is worse, not better, and an aggregate would have
hidden it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Mapping

from .judge import Judge, JudgeRequest
from .rubric import Rubric
from .taxonomy import Category, Scenario, Suite

__all__ = ["TurnResult", "ScenarioResult", "RunReport", "run_suite", "AgentFn"]

# Your agent under test: given a scenario and the turns so far, produce output.
AgentFn = Callable[[Scenario, int, tuple[str, ...]], str]


@dataclass(frozen=True)
class TurnResult:
    scenario_id: str
    turn_index: int
    output: str
    scores: Mapping[str, float]
    safety: Mapping[str, bool]
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        # A deterministic safety failure is fatal regardless of graded scores.
        # A turn that leaked a phone number did not "mostly pass".
        return not self.violations and all(self.safety.values())

    @property
    def safety_failures(self) -> tuple[str, ...]:
        return tuple(k for k, ok in self.safety.items() if not ok)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    turns: tuple[TurnResult, ...]

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.turns)

    @property
    def first_failure(self) -> TurnResult | None:
        return next((t for t in self.turns if not t.passed), None)


@dataclass
class RunReport:
    suite_name: str
    era: str
    judge_model: str
    agent_model: str
    results: tuple[ScenarioResult, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------- headline

    @property
    def turn_pass(self) -> float:
        turns = [t for r in self.results for t in r.turns]
        if not turns:
            return 0.0
        return sum(t.passed for t in turns) / len(turns)

    @property
    def scenario_pass(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)

    # ------------------------------------------------------------ breakdown

    def by_category(self) -> dict[Category, float]:
        """Scenario-pass per category. This is the number that means something."""
        out: dict[Category, float] = {}
        for category in Category:
            subset = [r for r in self.results if r.scenario.category is category]
            out[category] = (
                sum(r.passed for r in subset) / len(subset) if subset else 0.0
            )
        return out

    def dimension_means(self) -> dict[str, float]:
        turns = [t for r in self.results for t in r.turns]
        if not turns:
            return {}
        keys = turns[0].scores.keys()
        return {k: mean(t.scores[k] for t in turns) for k in keys}

    def safety_failures(self) -> list[TurnResult]:
        return [t for r in self.results for t in r.turns if t.safety_failures]

    def failures(self) -> list[TurnResult]:
        return [t for r in self.results for t in r.turns if not t.passed]

    # ---------------------------------------------------------------- gates

    def compare(self, baseline: "RunReport") -> list[str]:
        """Release gate. Returns blocking reasons; empty list means ship it.

        Refuses to compare across eras, because a suite that changed size or
        difficulty produces a number that is not commensurable with the old
        one. Comparing them anyway is the single most common way teams
        convince themselves a regression is an improvement.
        """
        if self.era != baseline.era:
            raise ValueError(
                f"cannot compare across eras ({baseline.era!r} -> {self.era!r}). "
                "The suite changed, so the numbers are not commensurable. "
                "Report the new era as a fresh baseline instead."
            )

        blocking: list[str] = []

        if self.safety_failures():
            for t in self.safety_failures():
                blocking.append(
                    f"deterministic safety failure in {t.scenario_id} turn {t.turn_index}: "
                    f"{', '.join(t.safety_failures)}"
                )

        base_cat, new_cat = baseline.by_category(), self.by_category()
        for category in Category:
            if category.safety_critical and new_cat[category] < base_cat[category]:
                blocking.append(
                    f"safety-critical category {category.label} regressed "
                    f"{base_cat[category]:.0%} -> {new_cat[category]:.0%}"
                )

        if self.scenario_pass < baseline.scenario_pass:
            blocking.append(
                f"scenario-pass regressed {baseline.scenario_pass:.0%} "
                f"-> {self.scenario_pass:.0%}"
            )
        return blocking


def run_suite(
    suite: Suite,
    agent: AgentFn,
    judge: Judge,
    rubric: Rubric,
) -> RunReport:
    results: list[ScenarioResult] = []

    for scenario in suite:
        transcript: list[str] = []
        turn_results: list[TurnResult] = []

        for index, turn in enumerate(scenario.turns):
            output = agent(scenario, index, tuple(transcript))
            transcript.extend([turn.user, output])

            scores = judge.score(
                JudgeRequest(
                    scenario=scenario,
                    turn=turn,
                    turn_index=index,
                    agent_output=output,
                    rubric=rubric,
                    transcript=tuple(transcript),
                )
            )
            safety = rubric.check_safety(output)
            violations = list(rubric.violations(scores))

            # Turn-level assertions declared on the scenario itself.
            for banned in turn.must_not_contain:
                if banned.lower() in output.lower():
                    violations.append(f"must_not_contain: {banned!r} appeared")

            turn_results.append(
                TurnResult(
                    scenario_id=scenario.id,
                    turn_index=index,
                    output=output,
                    scores=scores,
                    safety=safety,
                    violations=tuple(violations),
                )
            )

        results.append(ScenarioResult(scenario=scenario, turns=tuple(turn_results)))

    return RunReport(
        suite_name=suite.name,
        era=suite.era,
        judge_model=judge.model,
        agent_model=judge.agent_model,
        results=tuple(results),
    )
