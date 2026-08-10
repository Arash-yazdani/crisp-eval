"""Judge-versus-human calibration.

The question nobody asks their own eval
--------------------------------------
You have a judge model scoring your agent. Your pass rate is 87%. Is that
true?

You do not know. You have never checked the judge against anything. The only
evidence that your judge is a good judge is that it produces numbers you find
plausible, which is exactly the standard a broken judge also meets.

The fix is unglamorous: label a sample by hand, compare, and report the
agreement. If a judge disagrees with you on a third of cases, its pass rate is
noise and every decision you made on it was a coin flip.

Why Cohen's kappa and not raw agreement
---------------------------------------
Raw agreement is inflated by class imbalance. If 95% of turns pass, a judge
that blindly says "pass" every time scores 95% agreement while carrying zero
information. Kappa corrects for agreement expected by chance:

    kappa = (observed - expected) / (1 - expected)

A blind-majority judge lands at ~0.0 no matter how high its raw agreement is.
That is the number worth putting in a README.

Rough reading (Landis & Koch, and treat it as a rough guide, not a law):

    < 0.00   worse than chance, something is inverted
    0.01-0.20  slight
    0.21-0.40  fair
    0.41-0.60  moderate
    0.61-0.80  substantial      <- usable for gating releases
    0.81-1.00  almost perfect

Below ~0.6, do not gate a release on this judge. Fix the judge first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "Label",
    "CalibrationReport",
    "cohens_kappa",
    "calibrate",
    "interpret_kappa",
]


@dataclass(frozen=True)
class Label:
    """One item labelled by both a judge and a human."""

    item_id: str
    judge: bool
    human: bool
    dimension: str = "overall"


def cohens_kappa(judge: Sequence[bool], human: Sequence[bool]) -> float:
    """Cohen's kappa for two binary raters.

    Returns 1.0 for the degenerate case where both raters gave the same single
    label to everything and agreed. That case carries no information either,
    which is why `CalibrationReport` also reports the label distribution.
    """
    if len(judge) != len(human):
        raise ValueError("judge and human label sequences must be the same length")
    n = len(judge)
    if n == 0:
        raise ValueError("cannot calibrate on zero labels")

    observed = sum(j == h for j, h in zip(judge, human)) / n

    # Probability of agreeing by chance, given each rater's own base rate.
    pj = sum(judge) / n
    ph = sum(human) / n
    expected = pj * ph + (1 - pj) * (1 - ph)

    if expected == 1.0:
        # Both raters gave the same constant label to every item.
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def interpret_kappa(kappa: float) -> str:
    if kappa < 0.0:
        return "worse than chance"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    kappa: float
    raw_agreement: float
    judge_positive_rate: float
    human_positive_rate: float
    false_pass: int  # judge said pass, human said fail  <- the dangerous one
    false_fail: int  # judge said fail, human said pass
    dimension: str = "overall"

    @property
    def interpretation(self) -> str:
        return interpret_kappa(self.kappa)

    @property
    def gate_ready(self) -> bool:
        """Whether this judge is trustworthy enough to block a release on."""
        return self.kappa >= 0.61 and self.n >= 50

    @property
    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.n < 50:
            out.append(
                f"only {self.n} labelled items; kappa is unstable below ~50 "
                "and should not be quoted as a headline"
            )
        if self.kappa < 0.61:
            out.append(
                f"kappa {self.kappa:.2f} ({self.interpretation}) is below the 0.61 "
                "threshold for gating releases; fix the judge before trusting it"
            )
        if self.false_pass > self.false_fail:
            out.append(
                f"judge is optimistic: {self.false_pass} false passes vs "
                f"{self.false_fail} false fails. It is hiding failures, which is "
                "the more expensive direction to be wrong in."
            )
        if abs(self.judge_positive_rate - self.human_positive_rate) > 0.15:
            out.append(
                f"base rates diverge ({self.judge_positive_rate:.0%} judge vs "
                f"{self.human_positive_rate:.0%} human); the judge and the rubric "
                "likely disagree about what 'pass' means"
            )
        return out

    def summary(self) -> str:
        lines = [
            f"dimension:        {self.dimension}",
            f"labelled items:   {self.n}",
            f"Cohen's kappa:    {self.kappa:.3f} ({self.interpretation})",
            f"raw agreement:    {self.raw_agreement:.1%}",
            f"judge pass rate:  {self.judge_positive_rate:.1%}",
            f"human pass rate:  {self.human_positive_rate:.1%}",
            f"false passes:     {self.false_pass}  (judge said pass, human said fail)",
            f"false fails:      {self.false_fail}",
            f"gate ready:       {'yes' if self.gate_ready else 'NO'}",
        ]
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


def calibrate(labels: Sequence[Label], dimension: str = "overall") -> CalibrationReport:
    subset = [l for l in labels if l.dimension == dimension] if dimension != "overall" else list(labels)
    if not subset:
        raise ValueError(f"no labels for dimension {dimension!r}")

    judge = [l.judge for l in subset]
    human = [l.human for l in subset]
    n = len(subset)

    return CalibrationReport(
        n=n,
        kappa=cohens_kappa(judge, human),
        raw_agreement=sum(j == h for j, h in zip(judge, human)) / n,
        judge_positive_rate=sum(judge) / n,
        human_positive_rate=sum(human) / n,
        false_pass=sum(1 for l in subset if l.judge and not l.human),
        false_fail=sum(1 for l in subset if not l.judge and l.human),
        dimension=dimension,
    )


def calibrate_all(labels: Sequence[Label]) -> dict[str, CalibrationReport]:
    """Per-dimension calibration. Judges are rarely uniformly good.

    A judge can be substantial on 'was the outcome recorded correctly' and
    worse than chance on 'was the tone warm', because one has a fact of the
    matter and the other does not. Reporting a single aggregate kappa hides
    which of your dimensions are actually measuring something.
    """
    dims = sorted({l.dimension for l in labels})
    return {d: calibrate(labels, d) for d in dims}
