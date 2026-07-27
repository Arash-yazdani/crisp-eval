# crisp-eval

An opinionated harness for LLM-as-judge evaluation, in ~900 lines with **no
dependencies outside the Python standard library**.

```
pip install -e ".[dev]" && pytest -q
42 passed in 0.03s
```

Five opinions, each of which exists because skipping it produces a number that
looks good and means nothing.

## 1. An aggregate pass rate is a headline, not a measurement

Scenarios are categorised with **CRISP** — Common, Realistic, Invalid/malicious,
Specific/edge, Performance — split by *failure mode* rather than topic.

Here is why that matters. A hardening pass aimed at adversarial robustness
produced this:

| Category | Before | After | Δ |
|---|---|---|---|
| I · Invalid / malicious | 30% | 80% | **+50** |
| S · Specific / edge | 60% | 80% | **+20** |
| P · Performance | 30% | 50% | **+20** |
| C · Common case | 70% | 60% | −10 |
| R · Realistic input | 90% | 60% | **−30** |

The aggregate moved 56% → 66%, which as a single number is a shrug. The
category view says something specific: safety-critical categories climbed hard
and the happy path fell, because guardrails that resist injection make an agent
terser on ordinary calls. That is a tradeoff you *chose*, and it is only
visible per-category.

Full spec: **[docs/CRISP.md](docs/CRISP.md)**.

## 2. Both metrics, always

```
turn-pass      = passing turns / total turns
scenario-pass  = scenarios where EVERY turn passed / total scenarios
```

Turn-pass is generous and moves smoothly. Scenario-pass is brutal and reflects
user experience — a conversation with one bad turn in the middle is a bad
conversation, not an 83% good one.

**The gap between them is itself a measurement.** Turn-pass 87% with
scenario-pass 66% means failures are spread thin across many conversations
rather than concentrated in a few. That is the worse shape, and an aggregate
hides it.

## 3. The judge must not share a model family with the agent

```python
Judge(my_judge_fn, model="gpt-4o", agent_model="gpt-4.1")
# SameFamilyJudge: judge 'gpt-4o' and agent 'gpt-4.1' are both 'openai'.
```

A same-family judge shares the agent's blind spots, its phrasing preferences,
and its failure modes. It reliably approves output a different model, or a
human, would flag. The pass rate rises and the product does not improve.

This gets actively dangerous when you also tune the judge against the agent's
observed behaviour: the two co-adapt, the suite goes green, and you have built
a machine that produces the number you wanted. If a write-up describes the
agent and judges as "co-optimized," it has described this failure as a feature.

Refused by default. Override with `allow_same_family=True` and document why.

## 4. Anything a regex can decide is decided by a regex

Two kinds of check:

- **Graded dimensions** — scored 0.0–1.0 by a judge. Warmth, escalation
  handling, sign-off quality. Genuinely subjective.
- **Deterministic keys** — computed by code. Did a phone number leak? Did an
  internal tool name appear in spoken output?

Handing a deterministic question to a judge is how you get 0.97 on "no PII
leaked" and no idea which three percent leaked.

A turn failing a deterministic key is a **failed turn**, regardless of graded
scores. It did not "mostly pass."

## 5. Suites carry an era, and eras refuse to be compared

```python
current.compare(baseline)
# ValueError: cannot compare across eras ('crisp-10' -> 'crisp-50').
# The suite changed, so the numbers are not commensurable.
```

Adding scenarios changes what the number means. A suite growing from 10 to 50
adversarial cases shows a **falling** pass rate while the agent is objectively
improving. Comparing across that boundary is the most common self-deception in
eval work, and it runs both ways: teams celebrate a rise that came from
dropping hard scenarios, and panic at a fall that came from adding them.

---

## And one habit: calibrate the judge, publish the kappa

An uncalibrated judge is an opinion with a decimal point. You have a judge
scoring your agent at 87%. Is that true? You do not know — you have never
checked it against anything. The only evidence your judge is a good judge is
that it produces numbers you find plausible, which is the standard a broken
judge also meets.

```python
from crisp_eval import Label, calibrate

report = calibrate([Label(f"call-{i}", judge=..., human=...) for i in ...])
print(report.summary())
```

```
labelled items:   120
Cohen's kappa:    0.087 (slight)
raw agreement:    84.2%
judge pass rate:  90.0%
human pass rate:  90.8%
false passes:     9  (judge said pass, human said fail)
false fails:      10
gate ready:       NO
  warning: kappa 0.09 (slight) is below the 0.61 threshold for gating
           releases; fix the judge before trusting it
```

**84% raw agreement, kappa 0.087.** That is the entire argument for using
kappa. Raw agreement is inflated by class imbalance: if 95% of turns pass, a
judge that says "pass" to everything scores 95% agreement while carrying zero
information. Kappa puts that judge at 0.0, which is the truth.

Below 0.61, do not gate a release on the judge — fix the judge first. And
calibrate **per dimension** via `calibrate_all()`: a judge is often substantial
on "was the outcome recorded correctly" and near-useless on "was the tone
warm," because one has a fact of the matter and the other does not.

`false_pass` is flagged separately from `false_fail` because a judge that hides
failures is expensive in a way a judge that flags healthy output is not.

## Failure mining

A suite written up front contains only the failures you could imagine.
Production contains the ones you could not.

```python
mined = mine(report.failures(), user_inputs)   # redacted + deduplicated
```

Failures become permanent regression scenarios, categorised by *why* they
failed rather than what they were about, deduplicated by fingerprint so a
recurring failure adds one scenario rather than four hundred, and redacted
before they enter the repo.

**The suite gets harder over time and the pass rate goes down.** That is the
part people cannot stomach. If your pass rate has risen monotonically for six
months, you are not mining failures — you are grading your own homework with an
answer key you wrote.

## Release gate

`RunReport.compare(baseline)` returns blocking reasons. Empty means ship.

Blocks on: any deterministic safety key failure; a safety-critical category
(I or S) scoring below the previous run *even if still above its floor*; any
safety-critical dimension regressing; overall scenario-pass falling within the
same era.

"We only got slightly worse at refusing prompt injection" is not a sentence
anyone should ship on.

## Try it

```bash
python examples/demo_suite.py
```

Runs the full loop offline against a synthetic bike-shop reminder agent — no
API keys, deterministic output. Two scenarios are deliberately buggy so the
demo shows the harness catching something rather than printing all green.

Swap `stub_agent` for your agent and `stub_judge` for a real model call.
Nothing else changes.

## Layout

```
src/crisp_eval/
  taxonomy.py     CRISP categories, Scenario, Suite, era + balance guards
  rubric.py       graded dimensions, floors, deterministic safety keys
  judge.py        judge protocol, cross-family enforcement, drift detection
  runner.py       turn-pass / scenario-pass, category breakdown, release gate
  calibration.py  Cohen's kappa, per-dimension, false-pass asymmetry
  mining.py       failures -> redacted, deduplicated regression scenarios
docs/CRISP.md     the taxonomy specification
tests/            42 tests, no network
examples/         runnable synthetic demo
```

## What the tests assert

Not the happy path. The things that go wrong:

- a blind-majority judge scores 95% raw agreement and kappa ≈ 0.0
- a same-family judge pairing is refused before it can run
- a judge returning an extra dimension raises *drift*, because a rubric and a
  judge prompt silently diverging is how a dimension stops being graded
- a deterministic safety failure fails the turn despite 1.00 on every dimension
- turn-pass and scenario-pass diverge (0.75 vs 0.00) on the same run
- category breakdown exposes Invalid at 0% while the aggregate reads 80%
- comparing across eras raises rather than returning a misleading delta
- mining 5 identical failures produces 1 scenario, with the input redacted

## License

MIT.
