# Q1-Q3 contract options: open-source review

Date: 2026-09-03

## Question

Critically review three selected POC decisions against current primary-source
open-source implementations:

1. a minimal common Specialist Result containing `summary + evidence_ids`, with
   optional domain-specific typed output;
2. one business `Task` concept rather than separate `Assignment` and `Task`
   concepts;
3. a `Finish` decision with no business payload.

The goal is not to prove the selected answers correct. It is to identify where
the original A/B choices omitted relevant alternatives or mixed different
layers of the system.

## Executive conclusion

The selected **A/A/A** remains a reasonable POC direction, but the original
choices need three corrections:

- `summary + evidence_ids` is a platform-specific minimum, not an open-source
  standard. Current implementations range from plain final text to a
  task-specific structured response. Do not implement a generic extension
  framework until the E2E scenario has a real typed consumer.
- Unifying the domain vocabulary as `Task` is useful, but it does not eliminate
  runtime concepts such as an async execution, attempt, or outcome. Those are
  implementation state, not a second business noun.
- `Finish` can have no business payload, but it cannot literally be an
  unidentifiable empty object in a typed union. It needs a discriminator, and
  the Agent Graph must separately retain its own code-derived termination
  cause for operations. That runtime cause is not Coordinator output and is not
  Synthesis input.

## Q1: minimal Specialist Result

### Primary-source facts

Deep Agents defaults to returning the subagent's last non-empty assistant
message to the parent. If a `response_format` is configured, it instead returns
the task-specific `structured_response`. The subagent definition therefore
supports both free-form and per-agent typed outputs; it does not impose a
universal `warnings` or `gaps` contract
([`SubAgent.response_format`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py#L149-L180),
[`_return_command_with_state_update`](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py#L613-L647)).

Open Deep Research exposes `ResearcherOutputState` as
`compressed_research + raw_notes`. It has no generic warnings/gaps fields, and
its supervisor consumes the compressed result as a tool message
([state](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/state.py#L72-L84),
[collection](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L283-L319)).

The OpenAI Agents SDK research example is simpler still: each search agent
returns a short text summary, the manager retains successful strings, and
failed searches are omitted from Writer input
([search collection](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/manager.py#L59-L100)).
More generally, the SDK allows an agent's `output_type` to be plain text or a
Pydantic-compatible task-specific type
([agent output types](https://openai.github.io/openai-agents-python/agents/#output-types)).

These examples do not establish `evidence_ids` as a community-standard field.
That field follows this platform's own requirement for auditable, eligible
Evidence and deterministic citation checks.

### What the original A/B framing missed

The choice implied only two possibilities: a universal minimal result or a
larger universal result. Open-source implementations demonstrate a third:
return plain final content by default and request a task-specific schema only
when a downstream consumer needs it.

Also, "the Coordinator can understand limitations from `summary`" is only an
LLM interpretation. It must not be described as typed or deterministic gap
handling.

### Recommendation for this POC

Keep the common result at `summary + evidence_ids` because both fields already
have consumers in this design. Remove universal `warnings/gaps`.

Treat domain-specific typed output as a permitted future seam, not as a generic
inheritance/payload framework that must be built now. Add such output in the
golden scenario only if a downstream Task actually needs machine-readable
values; for example, deterministic price calculations consumed by another
Specialist. When that case exists, its schema should belong to that Specialist
or Task contract, while the Coordinator continues to see only the common
projection.

## Q2: `Task` versus `Assignment + Task`

### Primary-source facts

Deep Agents exposes one `task` tool whose input is a detailed task description
plus a subagent type; one invocation runs a complete subagent loop and returns
its result. It does not create a separate business `Assignment` object
([task schema and description](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py#L374-L415)).

AutoGen similarly defines `TaskRunner.run(task=...) -> TaskResult`; the public
contract does not require an Assignment layer
([`TaskRunner` and `TaskResult`](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/base/_task.py#L8-L37)).

However, OpenAI's research example shows that domain planning input and runtime
execution remain mechanically distinct even when the vocabulary stays simple:
`WebSearchItem(query, reason)` values are converted into Python
`asyncio.Task`s for concurrent execution
([planner types](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/agents/planner_agent.py#L11-L20),
[execution](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/manager.py#L59-L96)).
LangGraph's `Send` likewise carries a per-branch state object to a node; it is a
runtime dispatch primitive, not a domain model
([LangGraph `Send`](https://docs.langchain.com/oss/python/langgraph/graph-api#send)).

### What the original A/B framing missed

This is not primarily a choice between one object and two objects. It is a
choice about domain vocabulary and lifecycle identity.

Using one business noun does not mean using one object for planning, scheduling,
an in-flight coroutine, retry attempts, and the terminal result. Conversely,
having internal execution state does not justify exposing a second domain noun
to the Coordinator.

### Recommendation for this POC

Use `Task` and `TaskOutcome` in the platform contract. Keep `task_id` stable
across technical retries and keep `attempt` as Graph-owned execution state.
LangGraph/Python runtime tasks must remain implementation details.

Introduce a separate `Assignment` or `Execution` domain entity only if it later
gets an independent lifecycle—for example, durable queueing, cancellation,
reassignment, or multiple executions of one delegated intent. None is required
by the current E2E.

## Q3: empty `Finish`

### Primary-source facts

Open Deep Research defines `ResearchComplete` as an empty Pydantic model. The
supervisor tool node checks only whether that tool was called, then routes to
LangGraph `END` while updating the graph's research notes
([empty completion signal](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/state.py#L18-L20),
[termination routing](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L237-L253)).
This is direct evidence that a model-facing completion signal need not contain
a business explanation or report content.

The same implementation can also stop because no tool was called or because an
iteration limit was exceeded. Those are runtime conditions, not fields authored
inside `ResearchComplete`
([exit criteria](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L231-L253)).

AutoGen makes the other layer visible: its application-level `TaskResult`
contains a `stop_reason`, and its termination conditions emit `StopMessage`s
([termination documentation](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/termination.html)).
This does not imply that the report-writing agent needs that information; it
shows that model intent and operational termination are different contracts.

### What the original A/B framing missed

The options conflated three things:

1. the Coordinator's model-facing decision;
2. the graph routing signal;
3. the runtime/audit record explaining why execution ended.

In addition, a typed `DispatchBatch | Finish` union needs a discriminator such
as `kind: Literal["finish"]`. "No business fields" is precise; "literally empty
object" is not.

### Recommendation for this POC

Keep a discriminated, payload-free Coordinator decision:

```python
class Finish(BaseModel):
    kind: Literal["finish"] = "finish"
```

The Agent Graph routes this to Synthesis when eligible Evidence exists, or to
the deterministic insufficient-evidence response when none exists. Separately,
the Graph may record a code-owned termination cause such as Coordinator finish,
hard limit, or execution failure for tracing and API behavior. Do not put that
cause into `Finish`, and do not pass it to Synthesis.

## Final disposition

No selected answer needs to be reversed, but none should be presented as a
universal community best practice:

- **Q1 A: retain with scope correction.** The two common fields serve this
  platform; optional typed output should be demand-driven.
- **Q2 A: retain as vocabulary simplification.** Do not collapse internal
  execution and retry state into the business Task model.
- **Q3 A: retain with layer separation.** `Finish` has a discriminator but no
  business payload; termination cause remains Graph-owned operational state.

