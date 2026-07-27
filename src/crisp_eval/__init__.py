"""crisp-eval — an opinionated harness for LLM-as-judge evaluation.

Five opinions, each of which exists because skipping it produces a number that
looks good and means nothing:

    1. Scenarios are categorised (CRISP). Aggregate pass rates hide the fact
       that adversarial coverage can collapse while the happy path holds.
    2. The judge must not share a model family with the agent under test.
    3. Anything a regex can decide is decided by a regex, not by a model.
    4. Turn-pass and scenario-pass are both reported. They disagree, and the
       disagreement is the signal.
    5. Suites carry an era. Numbers from different eras refuse to be compared.

And one habit: calibrate the judge against human labels and publish the kappa.
An uncalibrated judge is an opinion with a decimal point.
"""

from .calibration import (
    CalibrationReport,
    Label,
    calibrate,
    calibrate_all,
    cohens_kappa,
    interpret_kappa,
)
from .judge import Judge, JudgeRequest, SameFamilyJudge, constant_judge, model_family
from .mining import MinedScenario, categorise_failure, mine, redact
from .rubric import (
    DEFAULT_DIMENSIONS,
    DEFAULT_SAFETY_KEYS,
    Dimension,
    Rubric,
    SafetyKey,
)
from .runner import RunReport, ScenarioResult, TurnResult, run_suite
from .taxonomy import Category, Scenario, Suite, Turn

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # taxonomy
    "Category", "Scenario", "Suite", "Turn",
    # rubric
    "Rubric", "Dimension", "SafetyKey", "DEFAULT_DIMENSIONS", "DEFAULT_SAFETY_KEYS",
    # judge
    "Judge", "JudgeRequest", "SameFamilyJudge", "model_family", "constant_judge",
    # runner
    "run_suite", "RunReport", "ScenarioResult", "TurnResult",
    # calibration
    "Label", "CalibrationReport", "calibrate", "calibrate_all",
    "cohens_kappa", "interpret_kappa",
    # mining
    "mine", "MinedScenario", "categorise_failure", "redact",
]
