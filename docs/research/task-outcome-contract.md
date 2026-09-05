# Terminal Task outcome contract for parallel Specialist execution

Date: 2026-09-03

## Question

What is the smallest terminal outcome contract that lets a LangGraph fan-out
batch finish even when an expected Specialist failure occurs? In particular,
should the contract be a discriminated success/failure union, a nullable
envelope, or ordinary exception propagation, and should it also carry usage,
failure classification, attempts, and raw errors?

The repository currently pins LangGraph `1.1.10` and uses PydanticAI for model
execution. The answer below is scoped to the current bounded rolling
Coordinator--Specialist POC, not a general job-processing platform.

## Conclusion

The original option A has the right **success-or-failure union**, but too many
fields. The better missing option is a smaller D:

```python
class TaskSucceeded(BaseModel):
    kind: Literal["succeeded"] = "succeeded"
    task_id: str
    result: SpecialistResult


class TaskFailed(BaseModel):
    kind: Literal["failed"] = "failed"
    task_id: str


TaskOutcome = Annotated[
    TaskSucceeded | TaskFailed,
    Field(discriminator="kind"),
]
```

This contract has two real consumers:

1. The deterministic batch collector needs exactly one terminal value for each
   `task_id` before the next Coordinator round.
2. The Coordinator needs to know whether a prior Task yielded a usable
   `SpecialistResult`; for failure it already sees the failed Task's Specialist
   and objective in the accepted earlier Coordinator Decision.

Do **not** put `ModelUsage`, `failure_kind`, attempts, retry history, raw
exception text, or stack traces in this model-visible outcome. Usage is
Graph-owned accounting state. Attempts and diagnostics belong to the execution
adapter's telemetry/logging. No current business consumer justifies persisting
them in `TaskOutcome`.

The discriminated union is worthwhile even though no surveyed framework
mandates it. It encodes the only two valid states and prevents nullable
combinations such as `status="succeeded", result=None` or
`status="failed", result=<value>`. Pydantic documents tagged unions as the
predictable way to select and validate one union member
([Pydantic discriminated unions](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions)).

## Why a failure must sometimes be a value in this graph

LangGraph executes parallel nodes in one superstep. Its official Graph API
documentation states that if a branch raises, the superstep errors and none of
that superstep's state updates are applied; with a checkpointer, writes from
successful nodes are retained for retry/resume. The same documentation says
applications can catch errors in node code, while retry policies handle
retryable exceptions
([parallel supersteps and exception handling](https://docs.langchain.com/oss/python/langgraph/use-graph-api#exception-handling)).

Therefore option C, "every Specialist failure propagates as an exception",
cannot implement the required partial-failure behavior. A batch such as:

```text
prices succeeds + holdings succeeds + report provider is unavailable
```

must still reach its barrier with three terminal entries if the Coordinator is
expected to choose substitute research or finish. The `execute_specialist`
node must convert an **allowlisted terminal failure of the Specialist run
itself** to `TaskFailed` after its registered technical retry policy is
exhausted. Expected unavailability from an individual read/fetch/Calculation
Tool is instead a typed value inside the active Specialist run, so the Agent may
try an allowed fallback or finish with a partial Result.

The batch state should remain a reducer-backed map keyed by `task_id`, because
LangGraph also warns that parallel update order is not stable. The reducer can
be associative, commutative, and idempotent: identical duplicate writes are a
no-op; different outcomes under the same Task ID are an invariant conflict
([parallel update ordering and reducers](https://docs.langchain.com/oss/python/langgraph/use-graph-api#run-graph-nodes-in-parallel)).

## Which failures become `TaskFailed`

Convert only failures the execution policy recognizes as an inability to
complete this delegated research Task safely:

- an allowlisted provider/network failure remains after technical retries;
- PydanticAI exhausts model-output validation/retry for the Specialist's
  structured `SpecialistResult`; or
- an actor-local request, Tool-call, or output bound prevents the Specialist
  from producing any valid terminal `SpecialistResult`.

PydanticAI validates typed output and raises when usage or retry bounds are
exhausted rather than returning a normal result. Its output is normally wrapped
in `AgentRunResult`, while usage limits can terminate a run
([PydanticAI output](https://ai.pydantic.dev/output/),
[usage limits](https://ai.pydantic.dev/agent/#usage-limits)). The adapter should
map only the known, policy-owned terminal cases; it should not use a broad
`except Exception: TaskFailed(...)`.

These cases are not Task failures:

- An individual registered Tool cannot provide requested data for an expected,
  allowlisted reason, including that Tool's own call timeout. Its binding returns
  a `ToolReturn` whose model-visible value is typed unavailability and whose
  metadata contains no Evidence or Calculation Artifact;
  it does not raise `ModelRetry`, `ToolFailed`, or a generic exception. The
  Specialist may continue multi-hop or join successful fan-out siblings.
- A Specialist completes with usable partial evidence and bounded Data Gaps.
  That is a successful `SpecialistResult`, not a failed Task.
- A Specialist successfully determines that a source contains no relevant
  information. That is a valid `SpecialistResult` whose summary reports the
  negative finding, not a runtime failure.
- A structural Run limit such as maximum Tasks or Coordination Rounds is reached.
  The Agent Graph routes to bounded completion; it does not present that as a
  failed Specialist Task.
- User cancellation, LangGraph interrupt/control-flow signals, programmer
  errors, registry/configuration errors, authorization or tenant-boundary
  violations, corrupted Graph state, and reducer collisions. These propagate
  and fail or suspend the Run.

This distinction is essential: `TaskFailed` means "this research attempt did
not produce a result, but the Graph remains valid", not "the platform caught
something".

## Why `failure_kind` is not in the Coordinator view

The Coordinator has already been constrained to business decisions and must
not control timeout, retry, or budgets. Labels such as `provider_error`,
`timeout`, and `output_validation_exhausted` are technical facts with no
allowed model action. Exposing them adds prompt vocabulary without changing a
valid decision: the Coordinator may try a different research Task or finish.

Stable business distinctions such as `source_has_no_records` and
`source_temporarily_unavailable` belong in the bounded Specialist Result as a
negative Finding or Data Gap. The Coordinator receives no raw exception,
provider detail, retry history, or stack trace.

The execution adapter may still classify an exception for retry selection,
metrics, and logs. That internal classification is not part of `TaskOutcome`
and must not include raw provider text in the Coordinator prompt.

## Why usage is separate

PydanticAI treats usage as run accounting, not as the agent's business output.
It exposes usage through `AgentRunResult`/`AgentRun`, and its multi-agent guide
recommends passing a shared `RunUsage` when nested delegate usage should accrue
to the controlling run
([PydanticAI Agent usage](https://github.com/pydantic/pydantic-ai/blob/main/docs/agent.md#accessing-usage-and-final-output),
[multi-agent usage](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#agent-delegation-and-dependencies)).

For this POC, sibling Specialists are dispatched concurrently by LangGraph,
not nested inside one PydanticAI parent run. Each branch should account for its
own usage and return it in a **separate Graph-owned accounting update**; the
Graph aggregates those updates for hard limits and final response metadata.
The Coordinator and Synthesis prompts do not receive it.

Keeping usage out of `TaskOutcome` also handles failed runs correctly: the
adapter can retain the mutable/task-local usage accumulator even when no
`AgentRunResult` is returned, without inventing a fake business result. The
exact accounting-state shape can remain an implementation detail until the
Graph-state model is written; it does not require a new public envelope.

## What the surveyed projects actually do

There is no shared open-source `TaskOutcome` standard.

### LangGraph supplies execution semantics, not this business contract

LangGraph supplies `Send`, supersteps, retry policies, checkpointer writes, and
reducers. It does not prescribe a success/failure schema for worker output.
Because an unhandled branch exception errors the superstep, applications that
want partial progress must choose and implement their own expected-error value
at the node boundary
([Graph API exception handling](https://docs.langchain.com/oss/python/langgraph/use-graph-api#exception-handling),
[persistence and pending writes](https://docs.langchain.com/oss/python/langgraph/persistence#checkpoints)).

### OpenAI examples show both policies, selected by the application

The OpenAI Agents SDK parallelization example uses `asyncio.gather` over three
agent runs and does not convert exceptions; one exception therefore fails that
simple all-or-nothing workflow
([parallelization example](https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/parallelization.py)).

Its research-bot manager needs partial search results, so every search catches
an exception and returns `None`; the collector counts failures and sends only
successful summaries to the writer
([research manager](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/manager.py#L59-L103)).
That supports the required failure-as-value direction, but `None` is too lossy
for this rolling Coordinator: it cannot be correlated to the failed Task once
results complete out of order. Adding `task_id` and a success/failure tag is
the minimum necessary difference.

### Deep Agents returns output directly and lets invocation errors propagate

Deep Agents invokes a synchronous subagent with `ainvoke()` and converts a
successful structured response or final AI message into the parent tool
message. The task boundary itself does not wrap arbitrary subagent exceptions
as a typed terminal result
([subagent invocation and result conversion](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py#L613-L723)).
That design is suitable for its conversational `task` tool but does not provide
the partial-failure fan-in guarantee required here.

### AutoGen's `TaskResult` is a successful termination record, not an error sum type

AutoGen's `TaskResult` contains produced messages and an optional
`stop_reason`. `stop_reason` describes why the team termination condition fired
(for example, maximum messages, timeout termination, or a function execution),
not a general `succeeded | failed` result union
([TaskResult source](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/base/_task.py#L11-L44),
[termination conditions](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/termination.html)).
Team runtime exceptions are carried through a separate termination/error path,
which reinforces the separation between normal output and operational errors
([group-chat runtime source](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_base_group_chat.py)).

AutoGen therefore does not justify adding a free-form `stop_reason` or
technical failure taxonomy to this Coordinator-visible outcome.

## Review of Q14's original choices

### A: discriminated `TaskSucceeded | TaskFailed`

Keep the union and `task_id`; remove `usage` and `failure_kind` from the
model-visible contract. The original option mixed three concerns:

- fan-in control (`kind`, `task_id`);
- business output (`SpecialistResult`);
- operations (`usage`, technical failure classification).

Only the first two belong here.

### B: one nullable envelope

Reject. A single model with `result | None`, `error | None`, and a string status
admits impossible combinations unless custom validators recreate the union by
hand. It is neither smaller nor clearer once checkpoint serialization and
prompt projection are considered.

### C: propagate every failure

Reject for expected terminal failures because it aborts the LangGraph
superstep. Retain exception propagation for invariant/control failures and
Run-wide stop conditions.

### D: minimal discriminated outcome plus separate operational accounting

Recommend. It is the smallest contract that preserves Task correlation,
partial-failure fan-in, deterministic validation, and bounded replanning
without exposing technical state to the Coordinator.

## Tests implied by the recommendation

- Mixed batch: two successes plus one expected terminal failure produces three
  collected outcomes and reaches the Coordinator again.
- The failed outcome contains only `kind` and `task_id` in the Coordinator
  projection; no raw error, retry count, budget, or usage appears.
- Eligible provider/output-validation failure is mapped only after its
  registered outer retry policy is exhausted; actor-local-limit exhaustion is
  mapped immediately because its counters do not reset across attempts.
- A Tool-call timeout returns an unavailable value inside the same actor run;
  a multi-hop fallback can still succeed, and one unavailable internal fan-out
  branch does not cancel successful siblings.
- Cancellation, authorization failure, reducer conflict, and an unexpected
  programmer exception propagate rather than becoming `TaskFailed`.
- A successful partial or negative finding remains `TaskSucceeded` and carries
  bounded Data Gaps where applicable.
- Usage from both successful and expected-failed runs contributes to separate
  Graph-owned accounting and final metadata.
- Reducer merge is order-independent; identical duplicate Task writes are
  idempotent and conflicting duplicate writes fail the Run.

## Primary sources

- [LangGraph Graph API: parallel execution and exception handling](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [LangGraph persistence and pending writes](https://docs.langchain.com/oss/python/langgraph/persistence)
- [OpenAI Agents SDK research manager](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/manager.py)
- [OpenAI Agents SDK parallelization example](https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/parallelization.py)
- [Deep Agents subagent implementation](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py)
- [AutoGen `TaskResult` source](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/base/_task.py)
- [AutoGen termination conditions](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/termination.html)
- [PydanticAI agent and usage](https://github.com/pydantic/pydantic-ai/blob/main/docs/agent.md)
- [PydanticAI multi-agent applications](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md)
- [Pydantic discriminated unions](https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions)
