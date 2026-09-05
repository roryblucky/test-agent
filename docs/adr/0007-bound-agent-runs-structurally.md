# Bound Agent Runs structurally without a Run wall-clock cutoff

Status: accepted

The POC has no Run-wide wall-clock cutoff and no whole-Specialist elapsed-time
limit because a Specialist may perform legitimate bounded multi-hop or
fan-out/fan-in work. It instead bounds each model and Tool call with its owning
adapter's timeout and prevents loops with maximum Tasks, Coordination Rounds,
actor-local requests/Tools/retries/output, and LangGraph recursion. Expected
Tool unavailability, including an individual Tool-call timeout, becomes a typed
result inside the active Specialist run, so multi-hop may continue and one
internal fan-out branch cannot cancel successful siblings. A valid partial
Specialist Result records bounded Data Gaps and remains `TaskSucceeded`; only a
terminal inability of the Specialist run itself to produce a valid result may
become `TaskFailed` after its outer retry policy. Already dispatched Tasks are
not cancelled merely because the Run has been executing for a particular
duration.

When an accepted Task fails, an accepted Data Gap exists, or the maximum Task or
Coordination Round count prevents further dispatch, the Graph derives a
conservative incomplete-research projection from accepted outcomes. With
eligible Evidence it invokes Synthesis and code adds the disclosure; without
eligible Evidence it returns the deterministic insufficient-Evidence answer with
the same disclosure. Later work does not clear an accepted limitation in this
POC because no domain coverage-equivalence contract exists. These intentional
bounded outcomes use the existing `done` event and checkpoint the exact
assistant answer. External cancellation and authorization, invariant,
programmer, or checkpoint failures remain non-`done` failure paths.

Aggregate parallel model requests, Tool attempts, tokens, and cost are measured
and reported but are not atomically pre-authorized ceilings. The platform exposes
only read, fetch, and deterministic Calculation operations, so Tool-call limits
prevent loops rather than protect transfers, mutations, business budgets, or
quotas. The Agent Graph uses an explicit recursion limit of 40; recursion overflow
and external cancellation remain fatal.
