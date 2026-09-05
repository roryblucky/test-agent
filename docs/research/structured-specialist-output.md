# Structured Specialist output in agent orchestration

Date: 2026-09-03

## Question and conclusion

This note answers Q15 for the Agent Graph POC: must every autonomous Specialist
Agent return a structured business object, particularly when Specialist Tasks
participate in a multi-hop dependency chain?

**Conclusion:** use a hybrid boundary. A Specialist Agent remains autonomous
inside its invocation, but its terminal result must satisfy the contract required
by the consuming Task. Require a registered typed payload when downstream code,
a deterministic reducer/router, or a field-level multi-hop dependency consumes
the result. Permit a narrative result when it is only displayed or interpreted by
another LLM. Do not require every Specialist to implement one universal rich
business schema.

This produces three distinct concepts:

1. The Specialist's internal run: model turns, Skill selection, and bounded Tool
   calls. These are not workflow Tasks or part of its result contract.
2. The Specialist's primary result: narrative content and, when required by the
   delegated Task, a validated domain payload.
3. The platform-owned `TaskOutcome`: execution status, attempts, usage, errors,
   and Evidence references recorded by the LangGraph adapter. This is not the
   PydanticAI Agent's `output_type`.

Typing the terminal handoff does not make a Specialist a function. It constrains
what the Agent commits when it finishes a particular delegated Task; the Agent
still decides how to achieve the goal within its Tool, Skill, and budget policy.

## What the primary sources show

### PydanticAI supports both narrative and typed Agent results

PydanticAI defines Agent output as the final value of a run. It can be plain text,
structured data, an image, or an output-function result. `AgentRunResult` wraps
that primary value separately from run usage and message history. Its
`output_type` supports Pydantic models, dataclasses, `TypedDict`, lists, dicts,
scalars, unions, and multiple alternatives. The official example explicitly
allows either a structured `Box` or `str`; therefore structured output is an
available per-Agent contract, not a requirement that all Agents share the same
shape ([PydanticAI output documentation](https://pydantic.dev/docs/ai/core-concepts/output/)).

PydanticAI's official delegation example is directly relevant. A parent Agent
calls a delegate Agent, the delegate runs its own Agent loop, and control returns
to the parent after the delegate finishes. The delegate returns a typed
`list[str]`, demonstrating that autonomy and a typed terminal result are
compatible. The same guide recommends graph-based control flow for more complex
multi-Agent applications and distinguishes it from model-chosen delegation
([PydanticAI multi-agent applications](https://pydantic.dev/docs/ai/guides/multi-agent-applications/),
[source Markdown](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md)).

Pydantic validation proves conformance to a shape, not factual correctness.
PydanticAI provides output validators that may raise `ModelRetry`, including for
asynchronous or semantic checks that do not fit a Pydantic field validator. In
this platform, Evidence eligibility, citation existence, authorization, and
calculation correctness must still be checked by deterministic platform code;
they must not be inferred from successful schema parsing
([PydanticAI output validators](https://pydantic.dev/docs/ai/core-concepts/output/#output-validators)).

### LangGraph types workflow state where orchestration needs it

LangGraph's official orchestrator-worker example uses structured output for the
planner, gives each dynamically created worker its own state through `Send`, and
merges worker results into a declared shared state field. The example workers
return strings because their only consumer concatenates report sections. This is
evidence for typing control and dataflow boundaries according to the consumer,
not for requiring every worker to produce a rich business object
([LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)).

LangGraph state itself has an explicit schema, and each state key may have its
own reducer. A graph may also declare different input, output, and internal state
schemas. An Agent can therefore run inside a graph node while a small adapter
selects and validates what enters shared graph state
([LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api),
[LangChain custom multi-agent workflow](https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow)).

Parallel updates in one LangGraph super-step are not guaranteed to arrive in a
consistent order. The official guidance is to include an ordering value and sort
explicitly when a predetermined order is required. Consequently, typed worker
payloads alone do not make map-reduce deterministic: every mapped result also
needs a stable map key/order key, and the registered reducer must sort or merge
by that key ([LangGraph Graph API parallel execution](https://docs.langchain.com/oss/python/langgraph/use-graph-api#run-graph-nodes-in-parallel)).

### Deep Agents makes subagent structure optional

LangChain's open-source Deep Agents implementation accepts an optional
`response_format` for each subagent. When present, the subagent's
`structured_response` is serialized back to the parent; when absent, the last
non-empty Agent message is returned. Its task API can also supply an optional
response schema for an invocation. This is a concrete open-source precedent for
selective structured handoffs while retaining a fully agentic subagent loop
([Deep Agents subagent source](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py),
[Deep Agents dynamic subagents](https://docs.langchain.com/oss/python/deepagents/dynamic-subagents)).

The implementation also keeps the subagent description used for delegation
separate from its response format, Tools, and Skills. Thus a compact capability
descriptor does not need to expose a complete JSON schema to the Orchestrator.

### Other open-source agent SDKs also separate the run envelope from optional structure

OpenAI's open-source Agents SDK defaults an Agent to plain text but supports a
Pydantic-compatible `output_type`. Its run result exposes `final_output`
separately from rich run items, raw responses, and interruption state. It also
notes that handoffs can change which Agent finishes a run, so the aggregate
`final_output` cannot always be statically narrowed even though a particular
Agent may have a typed output
([OpenAI Agents SDK Agent output types](https://openai.github.io/openai-agents-python/agents/#output-types),
[OpenAI Agents SDK results](https://openai.github.io/openai-agents-python/results/#final-output)).

Microsoft Agent Framework's design ADR surveys multiple agent SDKs and reaches a
similar tradeoff. It separates the Agent's primary result from progress/internal
updates, notes that nested-agent consumers usually want only the bounded primary
response, and explicitly rejects forcing every custom Agent implementation to
perform a structured final-output step. The ADR still supports typed extraction
for Agents that provide structured output
([Microsoft Agent Framework ADR-0001](https://github.com/microsoft/agent-framework/blob/main/docs/decisions/0001-agent-run-response.md)).

AutoGen provides a `StructuredMessage` whose content is a Pydantic model while
also retaining ordinary text and handoff message types. This corroborates that
typed Agent-to-Agent messages are useful but need not replace every narrative
message ([AutoGen message types](https://microsoft.github.io/autogen/stable/reference/python/autogen_agentchat.messages.html)).

## Recommended POC contract

Keep the platform control envelope independent from the Specialist's primary
result:

```python
class TaskOutcome(BaseModel):
    task_id: TaskId
    spec_hash: str
    status: TaskStatus
    attempt: int
    usage: Usage
    evidence_ids: tuple[EvidenceId, ...]
    error: TaskError | None
    result: SpecialistResult | None
```

Use two explicit Specialist result modes rather than a payload that is always
optional:

```python
class NarrativeSpecialistResult(BaseModel):
    content: str
    evidence_ids: tuple[EvidenceId, ...]


class TypedSpecialistResult[T](BaseModel):
    content: str
    evidence_ids: tuple[EvidenceId, ...]
    payload: T
```

The distinction matters. If `payload: T | None` were used for every Task, a
multi-hop plan could pass compilation and still discover at runtime that a
required machine-readable value is absent. A typed-result Task must require its
payload; a narrative-result Task must not be referenced through payload fields.

For the POC:

- A Specialist Task feeding a registered calculation, reducer, router,
  deterministic validator, or field projection must select a registered output
  contract and return `TypedSpecialistResult[T]`.
- A terminal research section or a result used only as LLM context may return
  `NarrativeSpecialistResult`.
- `content` is a bounded human/LLM-readable synthesis, not the Specialist's raw
  message or Tool history.
- `evidence_ids` are stable references; raw Evidence and Tool payloads stay in
  request-local storage as already specified.
- The LangGraph adapter, not the model, constructs `TaskOutcome` and accounts for
  status, attempts, usage, and errors.
- PydanticAI validates the selected final result type. Platform validators then
  verify referenced Evidence and any domain invariants before making the outcome
  available to downstream Tasks.
- Fan-out results include a stable map key. Deterministic fan-in sorts or merges
  by that key rather than relying on branch completion order.

This recommendation preserves the current responsibility split: PydanticAI owns
the autonomous Agent interaction and terminal model contract; LangGraph owns
scheduling, state transitions, and deterministic data movement.

## The unresolved design consequence

The platform cannot simultaneously guarantee arbitrary field-level typed
dataflow before dispatch and hide all knowledge of result contracts from the
Planner unless code can infer every data mapping.

There are two coherent choices:

### A. Compact contract identities are planner-visible (recommended)

Expose only registered capability/result-contract IDs and one-line summaries in
the Specialist descriptor, not complete JSON schemas. A Task selects one of
those IDs. The plan compiler resolves the actual Pydantic type from trusted code
and validates downstream selectors and compatibility.

This slightly expands what the Orchestrator sees, but does not expose thousands
of Skills or turn the Specialist into a function. It gives the Planner enough
vocabulary to request a machine-consumable result and preserves deterministic
pre-dispatch validation.

### B. The Planner sees only Specialist identities and Task dependencies

The Planner delegates goals and says only that Task B depends on Task A. The
receiving Specialist gets A's bounded narrative/typed result, and either Agent
reasoning or hard-coded adapters decide how to interpret it.

This keeps descriptors minimal, but the plan compiler can validate topology,
permissions, and whole-result availability only. It cannot prove arbitrary
field-level type compatibility before execution. Multi-hop still works
semantically, but the earlier requirement for deterministic typed dataflow is
weaker.

The next design discussion should select A or B explicitly. Calling Specialist
Agents autonomous does not remove this tradeoff; it only clarifies that the
contract applies at the orchestration boundary rather than dictating the Agent's
internal procedure.

## Primary sources

- [PydanticAI output](https://pydantic.dev/docs/ai/core-concepts/output/)
- [PydanticAI multi-agent applications](https://pydantic.dev/docs/ai/guides/multi-agent-applications/)
- [PydanticAI multi-agent guide source](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md)
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph Graph API: parallel nodes](https://docs.langchain.com/oss/python/langgraph/use-graph-api#run-graph-nodes-in-parallel)
- [LangChain custom multi-agent workflow](https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow)
- [Deep Agents subagent implementation](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py)
- [Deep Agents dynamic subagents](https://docs.langchain.com/oss/python/deepagents/dynamic-subagents)
- [OpenAI Agents SDK Agent output types](https://openai.github.io/openai-agents-python/agents/#output-types)
- [OpenAI Agents SDK results](https://openai.github.io/openai-agents-python/results/#final-output)
- [Microsoft Agent Framework ADR-0001](https://github.com/microsoft/agent-framework/blob/main/docs/decisions/0001-agent-run-response.md)
- [AutoGen message types](https://microsoft.github.io/autogen/stable/reference/python/autogen_agentchat.messages.html)
