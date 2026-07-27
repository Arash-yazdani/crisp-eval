"""Turning production failures into regression scenarios.

A suite written up front only contains failures you could imagine. Production
contains the ones you could not. Failure mining is the loop that moves the
second set into the first, so the same mistake cannot ship twice.

The discipline is simple and almost nobody does it:

    1. A live call fails a dimension or a deterministic safety key.
    2. That transcript becomes a scenario, categorised by *why* it failed.
    3. The scenario joins the suite permanently.
    4. The suite gets harder over time, and the pass rate goes DOWN.

Step four is the part people cannot stomach. A suite that only ever gets easier
to pass is a suite that has stopped doing its job. If your pass rate has been
rising monotonically for six months, you are not mining failures.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

from .runner import TurnResult
from .taxonomy import Category, Scenario, Turn

__all__ = ["MinedScenario", "categorise_failure", "mine", "redact"]


_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"(?:\+?\d[\s.\-()]?){9,}\d"), "[PHONE]"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "[UUID]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive)\b"), "[ADDRESS]"),
)


def redact(text: str) -> str:
    """Strip direct identifiers before a live transcript enters the suite.

    Mandatory, not optional. A regression suite is a file in a repo that gets
    read by CI, copied into notebooks, and pasted into bug reports. Anything
    that lands in it has effectively been published. Run a real NER pass on
    top of this for anything regulated; these patterns catch the structured
    identifiers, not names.
    """
    out = text
    for pattern, token in _REDACTIONS:
        out = pattern.sub(token, out)
    return out


def categorise_failure(result: TurnResult) -> Category:
    """Assign a CRISP category based on how the turn failed.

    Categorising by failure mode rather than by surface topic is what keeps
    the suite balanced. Left alone, mined scenarios drift toward Common,
    because Common is where the traffic is.
    """
    if result.safety_failures:
        return Category.INVALID  # a leak is an adversarial-surface problem

    joined = " ".join(result.violations).lower()
    if any(k in joined for k in ("injection", "boundary", "must_not_contain")):
        return Category.INVALID
    if any(k in joined for k in ("escalation", "safety")):
        return Category.SPECIFIC
    if any(k in joined for k in ("latency", "timeout", "truncat")):
        return Category.PERFORMANCE
    if "status_accuracy" in joined:
        return Category.SPECIFIC
    return Category.REALISTIC


@dataclass(frozen=True)
class MinedScenario:
    scenario: Scenario
    source_turn: TurnResult
    reason: str


def mine(
    failures: Sequence[TurnResult],
    user_inputs: dict[tuple[str, int], str],
    *,
    prefix: str = "mined",
) -> list[MinedScenario]:
    """Convert failing turns into redacted, deduplicated regression scenarios.

    `user_inputs` maps (scenario_id, turn_index) -> the user utterance, since
    a TurnResult only carries the agent side.

    Deduplication is by a hash of the redacted input plus the failure reason,
    so the same recurring failure adds one scenario, not four hundred.
    """
    seen: set[str] = set()
    out: list[MinedScenario] = []

    for failure in failures:
        raw_input = user_inputs.get((failure.scenario_id, failure.turn_index), "")
        user = redact(raw_input)
        reason = "; ".join(failure.violations + failure.safety_failures) or "unspecified"

        fingerprint = hashlib.sha256(f"{user}|{reason}".encode()).hexdigest()[:12]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        category = categorise_failure(failure)
        out.append(
            MinedScenario(
                scenario=Scenario(
                    id=f"{prefix}-{fingerprint}",
                    category=category,
                    description=f"regression from {failure.scenario_id} turn {failure.turn_index}: {reason}",
                    turns=(
                        Turn(
                            user=user,
                            expect=f"must not repeat: {reason}",
                            must_not_contain=tuple(failure.safety_failures),
                        ),
                    ),
                    tags=frozenset({"mined", "regression"}),
                ),
                source_turn=failure,
                reason=reason,
            )
        )
    return out
