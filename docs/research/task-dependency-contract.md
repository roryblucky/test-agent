# Task dependency contract for the rolling Coordinator loop

Date: 2026-09-03

## Question

What is the smallest Coordinator-to-Specialist `Task` contract that still lets
the Agent Graph validate and demonstrate deterministic multi-hop execution?

The proposed contract was:

```python
class Task(BaseModel):
    task_id: str
    specialist_id: str
    objective: str
    input_task_ids: tuple[str, ...] = ()
```

## Conclusion

The proposal is close, but the original A/B/C choices omitted a better fourth
option:

```python
class Task(BaseModel):
    specialist_id: str
    objective: str
    context_task_ids: tuple[str, ...] = ()
```

`Task` is the Pydantic output authored by the Coordinator Agent. After
validation, the Agent Graph assigns its stable `task_id` in a Graph-owned
record. This prevents the model from inventing duplicate or malformed
identities without reintroducing a second business concept such as Assignment.
Because same-batch dependencies are forbidden, a new task never needs an ID so
another task in the same Coordinator Decision can reference it.

No separate `result_id` is needed while one Task produces exactly one terminal
`TaskOutcome`: the Graph can resolve a completed `task_id` to its accepted
`SpecialistResult`.

`context_task_ids` is preferable to `input_task_ids` or `depends_on`:

- the references must name successful Tasks from completed earlier rounds;
- the referenced `SpecialistResult` values are materialized as model context
  before the receiving Specialist runs;
- they are not unresolved scheduler dependencies, and they do not turn the
  rolling batch loop back into a general DAG.

This explicit field is **not** an open-source convention. It is a small
platform-specific addition required by this project's stated acceptance test:
the Graph must deterministically validate that a particular earlier result was
selected and attached to a later research Task. It cannot prove that an LLM
semantically used the context; without the reference, even the dataflow exists
only in prose and cannot be checked before dispatch.

## What current open-source systems do

### LangGraph materializes worker input

LangGraph's orchestrator-worker example gives each dynamic `Send` its own
worker state, such as `Send("llm_call", {"section": s})`. Workers write results
to a reducer-backed shared state. The example passes the materialized input; it
does not define dependency or result IDs
([LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents#creating-workers-in-langgraph)).

`Send` is therefore the dispatch mechanism, not the business contract. This
application remains responsible for validating a `Task` and building the
worker state passed to `Send`. The current `Send` type contains only a target
node, arbitrary argument state, and an optional timeout
([LangGraph `Send` source](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py#L649-L714)).

LangGraph separately exposes generated execution-task IDs and results in task
stream and checkpoint payloads. Those IDs describe Pregel node executions for
runtime observation and recovery; they are not model-authored business Task IDs
or dependency arguments to `Send`
([LangGraph task payloads](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py#L132-L186),
[Pregel task types](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py#L581-L628)).

### Deep Agents delegates with a self-contained description

Deep Agents' current task tool accepts only a detailed `description` and a
`subagent_type`. An isolated subagent gets that description as its new human
message; the instructions tell the caller to include all required context. A
fork-mode subagent may instead inherit the parent conversation. Neither mode
defines dependency/result references
([task schema](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py#L375-L396),
[state construction](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py#L663-L694)).

Deep Agents does require the surrounding model tool-call ID and uses it on the
returned `ToolMessage`, but that is internal call/result correlation rather
than an upstream Task reference
([result return and invocation](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py#L613-L723)).

This is suitable when an LLM directly calls subagents as tools. It does not
provide the dispatch-time dependency validation required here.

### PydanticAI and OpenAI Agents SDK leave context selection to application code

PydanticAI documents both agent delegation through a tool and programmatic
handoff. Its examples pass direct prompts, runtime dependencies, or selected
message history and return the delegate's typed output to the caller; it does
not prescribe dependency IDs
([agent delegation](https://pydantic.dev/docs/ai/guides/multi-agent-applications/#agent-delegation),
[programmatic handoff](https://pydantic.dev/docs/ai/guides/multi-agent-applications/#programmatic-agent-hand-off)).

The OpenAI Agents SDK likewise distinguishes LLM orchestration from code
orchestration. Its deterministic examples chain agents by transforming one
agent's output into the next one's input, while independent agents can run in
parallel. It does not impose a task dependency schema
([OpenAI Agents SDK orchestration](https://openai.github.io/openai-agents-python/multi_agent/)).
The research-bot example plans search items, runs them concurrently, and passes
the collected summaries to the writer; no IDs are used because it has no
iterative selective multi-hop dispatch
([research-bot architecture](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/README.md),
[research manager](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/manager.py#L59-L100)).

## Who selects prior results?

The Coordinator should select the prior results required by each new objective
through `context_task_ids`. The Agent Graph should validate and materialize
those references; it should not attach every earlier result to every
Specialist.

Attaching all results is mechanically simpler, but it has three costs:

1. a Specialist sees unrelated material and its context grows with every round;
2. the Graph cannot prove which prior result the Coordinator intended the Task
   to use;
3. the golden test cannot distinguish real outcome-driven multi-hop from a new
   independent task whose prompt happens to mention the earlier work.

This selection does not require the Coordinator to know result schemas. Its
round input already contains bounded common `SpecialistResult` projections and
their stable Task IDs. The receiving Specialist gets only the selected common
results, in canonical Task order.

## Model-facing versus Graph-owned fields

| Owner | Fields |
|---|---|
| Coordinator model output | `specialist_id`, `objective`, `context_task_ids` |
| Agent Graph Task record | `task_id`, round number, dispatch index, validated references plus the model-authored Task |
| Agent Graph execution state | status, attempt, per-call timeout/retry policy, timestamps, usage counters and terminal failure |
| Specialist input projection | objective plus materialized referenced `SpecialistResult` values |

Dispatch validation must prevent dispatch if a Task:

- names an unknown or unavailable Specialist;
- references a missing Task, a current-batch Task, a failed Task, or a Task from
  another Run/Tenant;
- exceeds the configured task or context limits.

No `constraints`, accepted-input schema, expected-output schema, retry, timeout,
or budget fields are needed in the Coordinator output for the current E2E.
Business constraints belong in `objective`; technical policy belongs to the
Agent Graph or Specialist configuration.

## What was wrong with the original choices

- Original A made the Coordinator own `task_id`, although the Graph is the
  identity and retry owner.
- `input_task_ids` was ambiguous: the values identify completed Tasks whose
  **results** become context, not Task objects used as raw input.
- Original B grouped unrelated speculative features together.
- Original C is compatible with common lightweight subagent APIs, but cannot
  satisfy this project's deterministic multi-hop validation requirement.

The recommended decision is therefore **D: Graph-assigned identity plus
explicit prior-result context references**.
