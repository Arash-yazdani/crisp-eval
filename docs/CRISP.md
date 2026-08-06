# The CRISP scenario taxonomy

A specification for structuring an agent evaluation suite so its number means
something.

## The problem

Most eval suites are a pile of transcripts with a pass rate on top. The rate
climbs, everyone feels good, and the agent still breaks in production, because
the suite never contained the cases that break it. Worse, the suite was
*written by the same person who wrote the agent*, so it encodes the same
assumptions about what users do.

An aggregate pass rate is a headline, not a measurement. CRISP is a discipline
for making the measurement.

## The five categories

Every scenario belongs to exactly one. The split is by **failure mode**, not by
topic, because scenarios that fail for the same reason should move together.

### C: Common case

The modal interaction. What happens all day, with a cooperative user and
everything working.

High traffic, low information. These scenarios pass early and stay passing, so
they tell you almost nothing after week one. Keep them anyway: they are the
canary for a hardening pass that made the agent worse at its actual job.

### R: Realistic input

Legitimate but messy. Interruptions, background noise, accents, half-answers,
people talking over the agent, someone else picking up, a user who answers a
different question than the one asked.

The gap between C and R is the gap between your demo and your product.

### I: Invalid / malicious

Adversarial. Prompt injection, instruction hijacking, social engineering,
probing for the system prompt, attempts to extract other users' data, requests
to do something out of scope framed as an emergency.

**Safety-critical.** A regression here is an incident, not a quality dip.

### S: Specific / edge

Rare branches where being wrong is expensive. Escalation paths, emergencies,
the ambiguous-intent case, the "user said something that could mean two things
and one of them is urgent" case, voicemail, callbacks, wrong-number.

**Safety-critical.** These are the scenarios that justify the whole product's
existence and the ones a demo never covers.

### P: Performance

Degraded conditions. Latency, truncation, upstream timeouts, partial
responses, rate limits, a tool call that hangs. Not "is it fast" but "what does
it do when it isn't."

## Two metrics, always both

```
turn-pass      = passing turns / total turns
scenario-pass  = scenarios where EVERY turn passed / total scenarios
```

Turn-pass is generous and moves smoothly. Good for tracking progress
week to week.

Scenario-pass is brutal and moves in steps. It is the one that reflects user
experience, because a conversation with one bad turn in the middle is a bad
conversation, not an 83% good one.

**The gap between them is itself a measurement.** Turn-pass 87% with
scenario-pass 66% means failures are spread thin across many conversations
rather than concentrated in a few. That is the worse of the two shapes, and
reporting a single aggregate would have hidden it entirely.

## Category scores are the signal

The aggregate is for the changelog. The category breakdown is for decisions.

A real example of why. A hardening pass aimed at adversarial robustness
produced this:

| Category | Before | After | Δ |
|---|---|---|---|
| I · Invalid / malicious | 30% | 80% | **+50** |
| S · Specific / edge | 60% | 80% | **+20** |
| P · Performance | 30% | 50% | **+20** |
| C · Common case | 70% | 60% | −10 |
| R · Realistic input | 90% | 60% | **−30** |

Aggregate scenario-pass rose 56% → 66%, which as a single number is a
shrug. The category view says something specific: the safety-critical
categories climbed hard, and the happy-path categories fell, because the
guardrails added to resist injection made the agent terser on ordinary calls.

That is a **tradeoff you chose**, and it is only visible per-category. Tracking
the aggregate alone, you would read the C and R dips as noise. Here they were a
diagnosable artifact: a benign closing pattern the judge scored as "dead air",
traced, fixed with a redesigned sign-off, and confirmed clean in production.

## Eras: why suites refuse to be compared

Adding scenarios changes what the number means. A suite that grows from 10 to
50 adversarial cases will show a **falling** pass rate while the agent is
objectively improving, because the bar moved.

Comparing across that boundary is the single most common self-deception in
eval work, in both directions: teams celebrate a rise that came from dropping
hard scenarios, and teams panic at a fall that came from adding them.

So every `Suite` carries an `era`, and `RunReport.compare()` raises if the
eras differ. When you change the suite, you start a new era and a new
baseline. Report improvement *within* an era. Never draw one line across two.

State this explicitly wherever you publish numbers. "The May and June suites
are not apples-to-apples" is a sentence that buys you more credibility than
any pass rate.

## Graded dimensions vs deterministic keys

Two kinds of check, and the difference matters more than it sounds.

**Graded dimensions** are scored 0.0–1.0 by a judge model. Warmth, whether an
escalation was handled well, whether a sign-off was clean. Genuinely
subjective; a model is a reasonable grader.

**Deterministic keys** are computed by code. Did a phone number leak? Did an
internal function name appear in spoken output? These have a correct answer
that a regex knows and a language model can only guess at.

Handing a deterministic question to a judge is how you end up with 0.97 on "no
PII leaked" and no idea which three percent leaked.

**Rule: every check you can make deterministic, you should.** Deterministic
checks do not drift, cost nothing, and cannot be talked out of their answer by
the thing they are grading. A turn that fails one is a failed turn regardless
of how the graded dimensions scored. It did not "mostly pass."

## Judge hygiene

**The judge must not share a model family with the agent.** A same-family judge
shares the agent's blind spots, its phrasing preferences, and its failure
modes. It reliably approves output that a different model, or a human, would
flag. The pass rate rises; the product does not improve.

This gets actively dangerous when you also tune the judge against the agent's
observed behaviour. The two co-adapt, the suite goes green, and you have built
a machine that produces the number you wanted. If your write-up says the agent
and judges are "co-optimized," you have described this failure as a feature.

`Judge` refuses same-family pairs by default.

**Calibrate against human labels and publish the number.** An uncalibrated
judge is an opinion with a decimal point. Label a sample by hand, compute
an agreement statistic, and report it next to the pass rate.

Use kappa rather than raw agreement, because raw agreement is inflated by class
imbalance: if 95% of turns pass, a judge that says "pass" to everything scores
95% agreement while carrying zero information. Kappa puts that judge at 0.0,
which is the truth.

Rough guide: below 0.61, do not gate a release on the judge. Fix the judge
first. And calibrate **per dimension**, because a judge is often substantial on "was
the outcome recorded correctly" and near-useless on "was the tone warm,"
because one has a fact of the matter and the other does not.

## Failure mining

A suite written up front contains only the failures you could imagine.
Production contains the ones you could not.

1. A live interaction fails a dimension or a deterministic key.
2. That transcript is **redacted** and becomes a scenario, categorised by *why*
   it failed rather than what it was about.
3. The scenario joins the suite permanently.
4. The suite gets harder. The pass rate goes **down**.

Step four is the part people cannot stomach. **A suite that only ever gets
easier to pass has stopped doing its job.** If your pass rate has risen
monotonically for six months, you are not mining failures; you are grading
your own homework with an answer key you wrote.

Categorise mined failures by failure mode, not topic. Left alone, mined
scenarios drift toward Common, because Common is where the traffic is.

Redaction before a transcript enters the suite is mandatory, not optional. A
regression suite is a file in a repo that CI reads, people copy into notebooks,
and paste into bug reports. Anything that lands in it has effectively been
published.

## Release gate

Block a release when any of these hold:

- any deterministic safety key failed anywhere
- a safety-critical category (**I** or **S**) scored below the previous run,
  even if it is still above its floor
- a safety-critical dimension regressed at all
- overall scenario-pass fell within the same era

"We only got slightly worse at refusing prompt injection" is not a sentence
anyone should ship on.

---

MIT licensed. Implemented in [`crisp_eval`](../src/crisp_eval).
