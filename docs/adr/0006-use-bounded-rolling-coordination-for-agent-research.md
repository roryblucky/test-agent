# Use bounded rolling coordination for Agent research

Status: accepted

Agent-mode research first uses the existing PydanticAI Query Understanding
actor to resolve bounded Conversation context into a standalone query and one
Business Intent. Deterministic Graph code validates that Intent and fixes a
Tenant-bounded Research Scope before coordination; downstream actors do not
receive Conversation history, and the scope cannot expand Tenant authority.

Research then uses one static LangGraph with bounded rolling Coordination
Rounds rather than compiling a complete model-authored Task DAG. A PydanticAI
Coordinator Agent proposes a discriminated `DispatchBatch | Finish` decision;
deterministic Graph code validates and concurrently dispatches each batch,
collects one terminal `TaskOutcome` per Task at a barrier, and invokes the
Coordinator again with bounded prior Results. Later Tasks may reference Results
from earlier rounds, which provides fan-out/fan-in, multi-hop dependencies,
Specialist delegation, and outcome-driven replanning without a workflow DSL or
dynamic Graph compiler. The Coordinator selects prior Results through
`context_task_ids`; after validation, the Graph assigns stable Task IDs and
materializes only those selected Results for the receiving Specialist.

One top-level Task delegates a business objective to one autonomous Specialist
Agent. The Specialist selects its eligible Skills and read-only Tools, including
registered deterministic calculation Tools, inside that run; Tool and
calculation calls are not separate top-level Tasks. LangGraph owns validation,
dispatch, limits, retry state, outcome collection, routing, and checkpoints,
while PydanticAI owns all model interaction. This supersedes ADR-0001 only where
that record requires complete Planner-authored Task DAGs or top-level
Calculation Tasks; its underlying LangGraph/PydanticAI boundary remains in
force.

Expected inability to obtain data from one registered Tool is a bounded typed
Tool outcome inside the Specialist run, not an exception that terminates that
run. This permits model-led multi-hop fallback and partial internal fan-out/fan-in.
A completed partial Specialist Result carries bounded, adapter-derived Data Gaps
from the accepted attempt and is successful; `TaskFailed` is reserved for an allowlisted terminal inability of
the Specialist run itself to produce a valid result after its outer policy.

The complete-DAG alternative could expose more pipeline parallelism, but it
would require a Plan compiler, dependency scheduler, typed-edge contracts, and
larger schema context before the core research E2E proves those mechanisms are
needed. The rolling design deliberately accepts batch barriers; dependency-
driven pipeline scheduling remains a future extension.

Parallel collection uses two-phase batch acceptance. Specialist branches write
only immutable staged terminal outcomes, including expected `TaskFailed` values.
The barrier alone may validate all expected Tasks and atomically promote one
immutable accepted batch, allowing successful siblings to remain eligible when
another Task fails as expected. External cancellation, authorization, checkpoint,
invariant, and programmer failures remain fatal and are not converted into staged
business outcomes.
