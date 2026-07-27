"""The CRISP scenario taxonomy.

Most eval suites are a pile of happy-path transcripts with a pass rate on top.
That number goes up and to the right and tells you nothing, because the suite
never contained the cases that break you.

CRISP forces coverage across five categories that fail for different reasons:

    C  Common      the modal interaction, the thing that happens all day
    R  Realistic   messy but legitimate input: interruptions, accents, noise
    I  Invalid     adversarial: prompt injection, social engineering, probing
    S  Specific    edge cases and safety-critical branches
    P  Performance latency, truncation, degraded upstream, timeout behaviour

The point of the split is diagnostic, not decorative. An aggregate pass rate
hides the fact that Invalid can collapse while Common holds steady. Category
scores are the actual signal; the aggregate is a headline.

Two properties fall out of this and both matter:

1. **Category scores move independently.** A hardening pass aimed at Invalid
   will raise Invalid and may *lower* Common, because the guardrails you add
   make the agent terser on the happy path. If you only track the aggregate
   you will read that as noise instead of a tradeoff you chose.

2. **Suites are not comparable across eras.** Adding scenarios changes what
   the number means. A suite that grows from 10 to 50 adversarial cases will
   show a falling pass rate while the agent is objectively improving. Compare
   within an era, never across one. `Suite.era` exists to enforce this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

__all__ = ["Category", "Turn", "Scenario", "Suite"]


class Category(str, Enum):
    COMMON = "C"
    REALISTIC = "R"
    INVALID = "I"
    SPECIFIC = "S"
    PERFORMANCE = "P"

    @property
    def label(self) -> str:
        return {
            "C": "Common case",
            "R": "Realistic input",
            "I": "Invalid / malicious",
            "S": "Specific / edge",
            "P": "Performance",
        }[self.value]

    @property
    def safety_critical(self) -> bool:
        """Invalid and Specific carry the branches that hurt someone.

        A regression here is not a quality dip, it is an incident. Callers
        should gate releases on these two independently of the aggregate.
        """
        return self in (Category.INVALID, Category.SPECIFIC)


@dataclass(frozen=True)
class Turn:
    """One exchange. `user` is what goes in, `expect` is what must hold."""

    user: str
    expect: str
    must_call_tool: str | None = None
    must_not_contain: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    id: str
    category: Category
    description: str
    turns: tuple[Turn, ...]
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError(f"scenario {self.id} has no turns")

    @property
    def turn_count(self) -> int:
        return len(self.turns)


class Suite:
    """A named, versioned set of scenarios.

    `era` is the guard against the most common self-deception in eval work:
    quietly changing the suite and then comparing the new number to the old
    one. Two suites with different eras refuse to be compared.
    """

    def __init__(self, name: str, era: str, scenarios: Sequence[Scenario]) -> None:
        if not scenarios:
            raise ValueError("a suite needs at least one scenario")
        ids = [s.id for s in scenarios]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate scenario ids: {sorted(dupes)}")
        self.name = name
        self.era = era
        self._scenarios = tuple(scenarios)

    def __iter__(self) -> Iterable[Scenario]:
        return iter(self._scenarios)

    def __len__(self) -> int:
        return len(self._scenarios)

    @property
    def scenarios(self) -> tuple[Scenario, ...]:
        return self._scenarios

    @property
    def turn_count(self) -> int:
        return sum(s.turn_count for s in self._scenarios)

    def by_category(self, category: Category) -> tuple[Scenario, ...]:
        return tuple(s for s in self._scenarios if s.category is category)

    def coverage(self) -> dict[Category, int]:
        return {c: len(self.by_category(c)) for c in Category}

    def missing_categories(self) -> list[Category]:
        """Any category with zero scenarios. A suite with gaps is not a suite."""
        return [c for c, n in self.coverage().items() if n == 0]

    def assert_balanced(self, minimum_per_category: int = 1) -> None:
        """Raise if any category is under-represented.

        Call this in CI. Suites rot toward Common over time because Common
        scenarios are the easy ones to write.
        """
        thin = {
            c.label: n
            for c, n in self.coverage().items()
            if n < minimum_per_category
        }
        if thin:
            raise ValueError(
                f"suite {self.name!r} is unbalanced, "
                f"need >= {minimum_per_category} per category: {thin}"
            )
