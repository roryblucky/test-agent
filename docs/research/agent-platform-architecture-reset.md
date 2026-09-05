# Agent platform architecture reset

Date: 2026-09-03

## Question and conclusion

This note deliberately discards the earlier platform assumptions and starts from
the product need: one read-only information/research API must handle variable
queries by composing independently owned Specialist Agents. Each Specialist has
its own model, prompt, Tools, and scoped or explicitly shared Skills. The runtime
must be able to perform semantic multi-hop research, parallel fan-out and fan-in,
main-to-Specialist delegation, and bounded replanning.

**Conclusion:** the minimum end-to-end architecture should be a **bounded
supervisor-Specialist loop inside a static custom LangGraph workflow**. The
Coordinator delegates one or more focused tasks, receives bounded findings,
reassesses progress, and either delegates again or finishes. Each Specialist is
a PydanticAI Agent that autonomously selects its eligible Skills and read-only
Tools.

Do not begin with an LLM-generated full DAG, a general Plan Compiler, typed
cross-Agent edges, or a workflow DSL. Those mechanisms may become justified by
future machine-to-machine pipelines, but they are not required to demonstrate
the current research product or the four requested behavioral patterns.

This is a community-aligned product architecture, not a framework-provided
template. Its important primitives, however, are the current first-party
recommendations across LangChain/LangGraph, PydanticAI, and the OpenAI Agents
SDK.

## What the primary sources show

### Multi-agent architecture is chiefly a context-engineering decision

LangChain's current multi-agent guide says that a single Agent is often enough,
and identifies three main reasons to introduce multiple Agents: context
management, distributed ownership, and parallelization. Its comparison rates
the subagents pattern highly for distributed development, parallelization, and
multi-hop work, while a router does not support multi-hop orchestration. It also
explicitly permits composing patterns, such as subagents that load Skills
on-demand
([LangChain multi-agent overview](https://docs.langchain.com/oss/python/langchain/multi-agent/)).

That description fits this product more closely than a generic plan executor:
the key boundary is which Specialist sees which prompt, Skill summaries, Tools,
and intermediate context.

### A manager with a single dispatch operation is the scalable Specialist seam

LangChain's subagent documentation distinguishes a tool-per-Agent approach from
a single parameterized dispatch tool. It recommends the single dispatch approach
when many Agents are developed by separate teams, when the Coordinator should
not change whenever an Agent is added, and when context isolation is important.
The convention is intentionally small: select an Agent by name, send it a task
description, and return only its final result. The child performs its detailed
work autonomously in an isolated context
([LangChain subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)).

The same guide says that names and short descriptions are the main information
the Coordinator needs for delegation. For a small, stable registry it recommends
enumerating those descriptors. For a large or dynamic registry it recommends a
`list_agents` or `search_agents` discovery Tool, specifically to reduce prompt
size and support independently managed Agents. This validates progressive
Specialist discovery rather than exposing every Specialist, Skill, Tool, or
schema to the Coordinator.

OpenAI's Agents SDK independently describes the corresponding "agents as tools"
manager pattern as the fit when one central Agent owns the final answer and must
combine multiple Specialist results. It contrasts that with handoffs, where the
Specialist takes control of the user conversation. It also recommends mixing
LLM decisions with code orchestration and uses ordinary concurrency primitives
for independent parallel work
([OpenAI Agents SDK orchestration](https://openai.github.io/openai-agents-python/multi_agent/)).

### Current first-party research examples use bounded iterative delegation

LangChain's current Deep Agents research tutorial does not compile an upfront
query-specific DAG. The main Agent keeps a TODO list, delegates focused research
tasks, evaluates the returned findings, and performs another delegation round if
needed. Its instructions recommend one subagent for most queries, multiple
parallel subagents only for explicit comparisons or clearly independent aspects,
and hard maximums for concurrency and delegation rounds
([Deep Agents research tutorial](https://docs.langchain.com/oss/python/deepagents/deep-research)).

The current Open Deep Research implementation follows the same control model:

1. turn the conversation into a research brief;
2. run a `supervisor` / `supervisor_tools` loop;
3. let the supervisor call `ConductResearch` one or more times;
4. execute the selected researcher subgraphs concurrently;
5. return compressed findings to the supervisor;
6. repeat until `ResearchComplete` or an iteration bound;
7. generate the final report.

The repository README labels its earlier full plan-and-execute workflow as a
legacy implementation and says the legacy implementations are less performant
than the current one
([Open Deep Research source](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py),
[Open Deep Research README](https://github.com/langchain-ai/open_deep_research)).

This does not prove that a DAG planner is always wrong. It does show that the
closest first-party implementation of open-ended deep research deliberately
uses evidence-driven iterative delegation rather than requiring the entire
execution graph before research begins.

### LangGraph should own the stable control shell

LangGraph distinguishes workflows, whose code paths are predetermined, from
Agents, whose processes and Tool usage are dynamic. Its orchestrator-worker
example uses structured planning plus the `Send` API to create an unknown number
of workers dynamically; worker outputs merge into shared state before synthesis
([LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)).

LangChain recommends a custom LangGraph workflow when an application must mix
deterministic logic and agentic behavior, perform multi-stage processing, or
embed an entire Agent as a graph node. Such a workflow supports static stages,
conditional branches, loops, and parallel execution without compiling a new
graph for each query
([LangChain custom workflow](https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow)).

### PydanticAI should own Agent/model boundaries, not a second orchestrator

PydanticAI describes an explicit complexity ladder: single-Agent workflows,
Agent delegation, programmatic handoff, and then graph-based control flow for
the most complex cases. Its delegation pattern lets a parent invoke a delegate
Agent and regain control after that Agent completes. Different Agents may use
different models and dependency sets
([PydanticAI multi-agent applications](https://ai.pydantic.dev/multi-agent-applications/)).

For this stack, PydanticAI should therefore provide the Coordinator and
Specialist Agent loops, Tool calling, output validation, usage accounting, and
model abstraction. LangGraph should provide the one durable orchestration
runtime. Adding `pydantic_graph` as another workflow engine would duplicate
ownership of transitions and state.

Deep Agents provides useful architectural confirmation but is broader than this
POC. Its custom subagents can have isolated Tools and Skills, and optionally a
structured `response_format`; custom subagent Skills do not automatically leak
to the parent
([Deep Agents subagents](https://docs.langchain.com/oss/python/deepagents/subagents)).
The platform can adopt those boundaries without adopting Deep Agents' complete
filesystem and task-management harness.

## Recommended minimum E2E architecture

```text
POST /v2/query/stream
        |
        v
prepare request / research brief
        |
        v
Coordinator Agent (PydanticAI)
        |
        +-- AnswerDirectly ------------------------------+
        |                                                |
        +-- DelegateBatch[1..N]                          |
                    |                                    |
                    v                                    |
        deterministic LangGraph dispatch                 |
                    |                                    |
          +---------+----------+                         |
          |         |          |                         |
          v         v          v                         |
      Specialist Specialist Specialist  (PydanticAI)     |
          |         |          |                         |
          +---------+----------+                         |
                    |                                    |
                    v                                    |
          bounded findings + evidence refs               |
                    |                                    |
                    +------> Coordinator reassesses -----+
                                 |                       |
                                 +-- Delegate again       |
                                 +-- FinishResearch ------+
                                                            v
                                                    final synthesis
                                                            |
                                                            v
                                                           END
```

The LangGraph topology is compiled once:

```text
START -> prepare -> coordinate -> dispatch -> coordinate -> synthesize -> END
                         |                         ^
                         +------ conditional -----+
```

`coordinate` and every Specialist invocation cross a PydanticAI boundary.
`dispatch` and graph routing are application code. A query changes the number,
order, and content of delegations, not the compiled graph topology.

### Coordinator decisions

Use a small validated action union rather than a complete execution plan:

```python
class DelegatedTask(BaseModel):
    delegation_id: str
    specialist_id: str
    objective: str
    relevant_finding_ids: tuple[str, ...] = ()


class DelegateBatch(BaseModel):
    tasks: tuple[DelegatedTask, ...]


class FinishResearch(BaseModel):
    reason: str


CoordinatorDecision = DelegateBatch | FinishResearch | AnswerDirectly
```

This structure validates what the runtime must execute while leaving research
strategy with the Coordinator. The runtime must reject unknown Specialist IDs,
duplicate delegation IDs, oversized batches, and limit violations before
dispatching that batch. It does not attempt to prove that the whole future
research process is valid before any evidence exists.

### Specialist registry and Skill visibility

Each trusted `SpecialistDefinition` should bind:

- a stable Specialist ID and short capability/when-to-use description;
- a PydanticAI Agent factory or instance;
- its model and system prompt;
- a read-only Tool allowlist;
- Specialist-scoped Skills plus explicitly assigned shared Skills;
- hardcoded per-run timeout, retry, token, Tool-call, and output-size limits.

The Coordinator initially sees only the small set of Specialist IDs and
descriptions. When the registry becomes large, add a bounded
`search_specialists(query)` operation and expose only its top results. Do not
make all Skills globally visible. A shared Skill is reusable configuration that
must still be assigned to each eligible Specialist; it is not ambient authority.

### Specialist result boundary

For research delegation, the useful default is a bounded result:

```python
class SpecialistFinding(BaseModel):
    summary: str
    evidence_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()
```

The Coordinator should not receive the Specialist's raw Tool payloads, complete
message history, or chain of intermediate actions. The Specialist's detailed
context remains isolated; evidence IDs refer to request-local or durable
evidence storage as appropriate.

A Specialist may use a task-specific structured output when an actual downstream
machine consumer needs it. That is a local Agent or fixed mini-workflow contract,
not a reason to require a universal cross-Agent Artifact registry and compiler
in the first POC.

### Minimal graph state

The shared state needs only:

- request and conversation identity;
- user query and normalized research brief;
- bounded Coordinator decision history or messages;
- current round and total/concurrent delegation counters;
- findings keyed by delegation ID;
- Evidence references and warnings;
- completion reason and final response;
- aggregate usage required for enforcing limits.

Specialist internal messages and large Tool results should not accumulate in the
Coordinator state. The graph checkpointer may persist this lightweight control
state, but checkpointing need not define a public resume or HITL product contract
for this POC.

## How the four patterns emerge

### Multi-hop research

Multi-hop is an evidence-dependent sequence of delegation rounds:

```text
Coordinator -> Profile Specialist -> profile finding
Coordinator -> Market Specialist with selected profile context -> analysis
Coordinator -> synthesis
```

The second step is chosen after the first result is known. The Coordinator names
the relevant prior finding IDs; application code resolves them into bounded
context. This is closer to how current research Agents operate than generating
all downstream steps before discovering the first result.

If a stable domain workflow later requires fixed field-level transformations,
implement it as a deterministic Tool or compiled Specialist subgraph. Do not
force semantic research dependencies to pretend to be typed function edges.

### Fan-out and fan-in

The Coordinator emits several independent `DelegatedTask` values in one
`DelegateBatch`. The dispatch node runs them concurrently using LangGraph
`Send` semantics or bounded `asyncio.gather`. The runtime waits for the batch and
stores results by stable delegation ID before the next Coordinator turn.

This makes the scheduling barrier and aggregation order deterministic. The
semantic synthesis of several research findings remains agentic. When a concrete
business calculation truly needs deterministic reduction, use an explicit
calculation/reducer Tool with a typed input owned by that domain rather than a
generic platform-wide map-reduce DSL.

### Main-to-Specialist delegation

Delegation is the primary composition primitive:

- Coordinator decides **who** should work and **what outcome** is needed.
- Specialist decides **how** to work: which eligible Skill to activate, which
  read-only Tools to call, and when its bounded task is complete.
- LangGraph controls dispatch, concurrency, accounting, and return to the
  Coordinator.

This is a manager/subagents design, not a peer handoff. Specialists do not own
the user conversation or final answer.

### Bounded research replanning

After each batch, the Coordinator sees the accumulated bounded findings and
chooses another batch or `FinishResearch`. It may maintain a user-visible TODO or
research ledger, but that ledger is a reasoning/progress aid, not an immutable
execution DAG.

Application code enforces maximum rounds, total tasks, concurrent tasks, model
requests, Tool calls, tokens, and elapsed time. Reaching a bound routes to final
synthesis with an explicit incomplete/limited reason instead of allowing an
unbounded Agent loop.

## Deterministic and agentic responsibilities

| Deterministic platform code | Agentic decisions |
| --- | --- |
| Authentication and tenant scope | Query decomposition |
| Read-only Tool allowlists | Specialist selection |
| Specialist registry resolution | Delegated objective wording |
| Coordinator action validation | Whether independent work merits fan-out |
| Concurrency, timeout, retry, and budget enforcement | Which prior findings matter next |
| Stable batch IDs, barrier, and result ordering | Skill and Tool choices inside a Specialist |
| Evidence reference validation and result bounding | Evidence sufficiency assessment |
| Checkpoint, status, and SSE event transitions | Follow-up research and semantic synthesis |
| Forced termination at hard limits | Early completion within those limits |

This boundary does not promise deterministic factual truth. It guarantees that
the runtime executes only authorized read-only capabilities, respects resource
limits, and records results predictably while the model makes inherently
semantic research decisions.

## How query variability is handled

The same graph accommodates different query shapes:

- a simple factual query can be answered directly or delegated once;
- a domain-specific question selects one Specialist;
- a comparison emits several parallel tasks in one batch;
- a dependent question uses multiple Coordinator rounds;
- missing or conflicting evidence triggers another bounded round;
- a cross-domain query delegates to Specialists from several domains and
  synthesizes their bounded findings.

Adding a future domain means registering another Specialist with its own prompt,
model, Tool allowlist, scoped/shared Skills, limits, and descriptor. An Agent Team
can later be a named registry/configuration preset controlling which Specialists
are discoverable. A Team DSL and arbitrary graph composition are not required to
prove this extension model.

## Rejected alternatives

### Dynamic full-DAG Planner plus Plan Compiler

Reject for the initial E2E. Open-ended research frequently discovers entities,
ambiguities, missing evidence, and follow-up questions only after a Specialist
runs. An upfront DAG either guesses those branches prematurely or requires a
second replanning/compiler subsystem that duplicates the Coordinator loop.

The closest first-party research implementations use iterative delegation, and
Open Deep Research explicitly retains full plan-and-execute only as a legacy
alternative. A full compiler also introduces schema exposure, typed-edge
inference, task identity/reuse rules, and plan revision semantics before the
business has demonstrated a stable machine pipeline that needs them.

Reconsider a DAG/compiler only when at least one proven use case requires:

- audited, pre-approved execution plans;
- stable machine-readable Artifacts passed between deterministic stages;
- complex long-running offline scheduling independent of a Coordinator turn;
- cross-run task reuse or idempotent external side effects;
- user editing/approval of the whole workflow before execution.

### Router as the core architecture

A router is useful for one-shot classification and parallel fan-out across clear
verticals, but LangChain's own comparison does not treat it as a multi-hop
pattern. It can become an intake optimization later; it cannot replace the
stateful Coordinator loop.

### One static domain workflow per query type

Fixed workflows are appropriate when the business process is known in advance.
They do not cover arbitrary questions whose decomposition and stopping condition
depend on evidence gathered during the run.

### One mega-Agent with every Tool and Skill

This defeats the context-isolation and distributed-ownership reasons for using
multiple Agents. It exposes an ever-growing capability surface to one model and
does not enforce domain-specific Tool or Skill visibility.

### Peer handoffs

Handoffs are useful when a Specialist should take over direct conversation with
the user. This platform instead needs one component to combine evidence and own
the final answer, so manager-style delegation is the better default.

### Adopting Deep Agents wholesale

Deep Agents demonstrates relevant practices: isolated subagents, optional
structured outputs, bounded delegation, and Skill scoping. Its broader planning,
filesystem, and task harness is not necessary for a read-only information POC.
Reuse the architecture, not unneeded surface area.

### Two graph runtimes

Do not combine LangGraph and `pydantic_graph` for orchestration. PydanticAI Agents
should be invoked at LangGraph node boundaries. One runtime should own shared
state, persistence, routing, and streaming.

## Recommended name

Call the core concept a **bounded supervisor-Specialist agentic workflow** or a
**manager/subagents architecture**. It resembles a Deep Agent because it performs
iterative planning, delegation, and context isolation, but "Deep Agent" normally
also implies a larger harness with task management, filesystem/context tooling,
and other capabilities that are outside this POC.
