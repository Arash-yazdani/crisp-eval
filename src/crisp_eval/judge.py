"""The judge protocol, and the guard that keeps a judge honest.

A judge is any callable that scores one turn against a rubric. There is no
vendor SDK in here on purpose: you supply the callable, the harness supplies
the discipline around it.

    def my_judge(request: JudgeRequest) -> dict[str, float]:
        resp = openai.chat.completions.create(model="gpt-4o", ...)
        return json.loads(resp.choices[0].message.content)

    judge = Judge(my_judge, model="gpt-4o", agent_model="gpt-4.1")

The `model` / `agent_model` pair is not bookkeeping. It is the single most
important control in LLM-as-judge evaluation and almost nobody enforces it.

Why judge family matters
------------------------
If the judge and the agent are the same model, the judge shares the agent's
blind spots, its phrasing preferences, and its failure modes. It will reliably
approve outputs that a different model, or a human, would flag. The pass rate
goes up and the product does not get better.

The failure has a particularly nasty form when you also tune the judge against
the agent's behaviour: the two co-adapt, the suite goes green, and you have
built a machine that produces the number you wanted. `Judge` refuses to run
same-family by default. Override it only if you know why you are doing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

from .rubric import Rubric
from .taxonomy import Scenario, Turn

__all__ = ["JudgeRequest", "JudgeFn", "Judge", "SameFamilyJudge", "model_family"]


@dataclass(frozen=True)
class JudgeRequest:
    scenario: Scenario
    turn: Turn
    turn_index: int
    agent_output: str
    rubric: Rubric
    transcript: tuple[str, ...] = field(default_factory=tuple)


class JudgeFn(Protocol):
    def __call__(self, request: JudgeRequest) -> Mapping[str, float]: ...


class SameFamilyJudge(RuntimeError):
    """Raised when the judge and the agent come from the same model family."""


_FAMILIES: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-", "o1", "o3", "o4", "chatgpt", "davinci"),
    "anthropic": ("claude",),
    "google": ("gemini", "palm", "bison"),
    "meta": ("llama",),
    "mistral": ("mistral", "mixtral"),
    "cohere": ("command",),
    "qwen": ("qwen",),
    "deepseek": ("deepseek",),
}


def model_family(model: str) -> str:
    """Best-effort family bucket from a model string.

    Deliberately conservative: an unknown model returns its own name, so two
    unknowns never collide into a false "same family" error, and an unknown
    never silently passes as different from a known one it might actually be.
    """
    lowered = model.strip().lower()
    for family, prefixes in _FAMILIES.items():
        if any(p in lowered for p in prefixes):
            return family
    return lowered


class Judge:
    """Wraps a judge callable with validation the callable will not do itself."""

    def __init__(
        self,
        fn: JudgeFn,
        *,
        model: str,
        agent_model: str,
        allow_same_family: bool = False,
    ) -> None:
        self._fn = fn
        self.model = model
        self.agent_model = agent_model
        self.same_family = model_family(model) == model_family(agent_model)

        if self.same_family and not allow_same_family:
            raise SameFamilyJudge(
                f"judge {model!r} and agent {agent_model!r} are both "
                f"{model_family(model)!r}. A judge sharing the agent's blind spots "
                f"inflates the pass rate without improving the product. Use a "
                f"different family, or pass allow_same_family=True and document why."
            )

    def score(self, request: JudgeRequest) -> dict[str, float]:
        raw = self._fn(request)

        missing = [k for k in request.rubric.keys if k not in raw]
        if missing:
            raise ValueError(f"judge omitted dimensions: {missing}")

        scores: dict[str, float] = {}
        for key in request.rubric.keys:
            value = float(raw[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"judge returned {key}={value}, expected 0.0-1.0")
            scores[key] = value

        extra = set(raw) - set(request.rubric.keys)
        if extra:
            # Not fatal, but it means the judge prompt and the rubric have
            # drifted apart, which is how a dimension silently stops being
            # graded. Surface it.
            raise ValueError(
                f"judge returned dimensions not in the rubric: {sorted(extra)}. "
                "Judge prompt and rubric have drifted."
            )
        return scores


def constant_judge(scores: Mapping[str, float] | None = None) -> JudgeFn:
    """A judge that always returns the same scores. For tests and dry runs.

    Defaults every dimension to 1.0, so `constant_judge()` gives you a judge
    that approves everything. Useful for exercising the deterministic safety
    keys in isolation: anything that still fails did so without a model's help.
    """
    fixed = dict(scores or {})

    def _fn(request: JudgeRequest) -> Mapping[str, float]:
        return {k: fixed.get(k, 1.0) for k in request.rubric.keys}

    return _fn
