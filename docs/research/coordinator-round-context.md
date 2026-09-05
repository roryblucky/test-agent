# Coordinator context across research rounds

Date: 2026-09-03

## Question

When the Coordinator Agent makes a new Coordinator Decision, what information
should it receive from earlier Coordination Rounds?

The concrete alternatives are:

1. the complete Agent message and Tool-call history;
2. all earlier bounded Specialist Results and Coordinator Decisions;
3. a rolling summary or ledger;
4. only the latest Dispatch Batch;
5. stable result references, with details loaded only when needed;
6. large notes stored outside the prompt, for example in a filesystem.

## Conclusion

There is no single open-source standard. The closest current research systems
use a common two-part pattern:

- keep the Coordinator's own control history;
- return a compressed final result from each delegated Agent, not that Agent's
  internal Tool calls or messages.

Rolling ledgers, automatic message compaction, result lookup, and filesystem
offloading are techniques for long or unbounded work. They are not required for
a small Run with strict limits.

For this repository's first end-to-end implementation, do **not** add a
`ResearchLedger`. On each Coordination Round, construct the Coordinator input
from deterministic Agent Graph state:

- the original user query;
- remaining Run limits;
- compact Specialist Descriptors;
- all accepted earlier Coordinator Decisions;
- all bounded Task Outcomes, in stable round and dispatch order.

Each Task Outcome must include only status and the bounded Specialist Result or
bounded failure summary. It must not include raw Evidence bodies, raw Tool
payloads, or Specialist internal messages. An Assignment may use stable result
references to select earlier Specialist Results for a later Specialist, but the
Coordinator does not need a lookup Tool in the POC.

This is deliberately simpler than the earlier `ResearchLedger` proposal. The
Run already has at most three follow-up rounds and 32 Tasks. A second, lossy
summary state would create synchronization rules before prompt size is a
measured problem.

## What primary implementations do

### LangChain Deep Agents: full parent history, compressed child returns

Deep Agents describes subagents as a context-isolation mechanism. By default, a
subagent runs with fresh context containing the delegated task description, and
the parent receives one final result, not the subagent's many Tool calls. That
result becomes a Tool message in the parent's accumulated messages. A fork mode
can instead copy the parent's effective conversation into the subagent, but it
is not the default. Deep Agents does not automatically add a research result
index or facts ledger
([Deep Agents subagents](https://docs.langchain.com/oss/python/deepagents/subagents),
[Deep Agents subagent source](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/middleware/subagents.py)).

Current Deep Agents also does not install TODO management by default; it became
opt-in in version 0.7. A TODO list can help one parent Agent track actions, but
it is not a domain facts ledger or a replacement for message history
([Deep Agents changelog](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/CHANGELOG.md)).

For long work, Deep Agents can offload large content to a virtual filesystem and
can summarize old messages when context grows. Its own guidance says that the
main Agent should receive concise subagent reports and read only needed file
fragments. These are context-size controls, not required coordination semantics
([Deep Agents context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering)).

### Open Deep Research: supervisor messages plus compressed research

The current Open Deep Research graph keeps `supervisor_messages` in graph state.
One supervisor turn may issue several `ConductResearch` calls. The researcher
subgraphs run concurrently, and each result returns to the supervisor as a Tool
message containing `compressed_research`. The graph stores raw notes separately
for final report generation. Thus the next supervisor turn sees its accumulated
control conversation and compressed researcher returns, but it does not receive
the researcher subgraphs' complete internal histories
([Open Deep Research source](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py)).

This is the closest implementation to the proposed Coordinator–Specialist
loop. It does not maintain a separate `completed_goals`, `remaining_goals`, and
`unresolved_gaps` ledger.

### AutoGen Magentic-One: full inner-loop thread plus a ledger on reset

Magentic-One uses a more complex hybrid. During its inner loop, every progress
assessment is built from the complete group message thread. That thread contains
the current Task Ledger, Coordinator instructions, and Agent replies; low-level
Tool-call events are filtered out. The orchestrator keeps separate `task`,
`facts`, and `plan` fields. When progress stalls, it updates the facts and plan,
clears the inner-loop thread, and starts again with a new full Task Ledger
([Magentic-One source](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py),
[Magentic-One architecture](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/magentic-one.html)).

Magentic-One therefore supports a rolling ledger, but it does so for open-ended,
potentially stalled work. It is not evidence that every short bounded research
loop needs one. The ledger also adds another model-generated state that must be
parsed and kept consistent with the message thread.

### OpenAI Agents SDK research bot: one pass and explicit summaries

The OpenAI Agents SDK research bot has no iterative Coordinator. It creates one
search plan, runs all searches concurrently, collects each Search Agent's final
string, and passes the original query plus the list of summaries to the Writer
Agent. It demonstrates the minimum pattern: pass bounded delegated results, not
child histories. Its README lists an evaluation and improvement loop as a future
enhancement rather than an implemented ledger
([research manager source](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/manager.py),
[research bot README](https://github.com/openai/openai-agents-python/tree/main/examples/research_bot)).

The SDK also treats run `context` as application dependency/state that is not
automatically exposed to the model. A developer must explicitly place needed
information in Agent input or message history
([OpenAI Agents SDK agents](https://openai.github.io/openai-agents-python/agents/)).

### PydanticAI: the application selects and processes history

PydanticAI supports passing `message_history` between runs and even between
different Agents. It warns that Tool calls and Tool returns remain in that
history, so applications should share only messages meaningful to the receiving
Agent. Its history processors can keep recent messages or summarize older ones
to reduce token use. These are framework mechanisms; PydanticAI does not require
a Supervisor ledger or prescribe which research facts must appear on each round
([PydanticAI message history](https://ai.pydantic.dev/message-history/)).

## Comparison

| Context strategy | Seen in practice | Benefit | Cost or risk | POC decision |
|---|---|---|---|---|
| Complete parent message history | Deep Agents, Open Deep Research, Magentic-One inner loop | Simple; preserves why earlier work happened | Grows with every round | Use an explicit bounded equivalent from Agent Graph state |
| Complete child Agent history | Not recommended by the reviewed systems | Maximum detail | Context pollution and Skill/Tool leakage | Do not use |
| Bounded Specialist final results | Deep Agents, Open Deep Research, OpenAI research bot | Preserves useful findings without child trace | Result size must be limited | Use |
| Rolling facts/plan ledger | Magentic-One | Helps long, stalled, open-ended work restart | Lossy; model-generated; must stay consistent | Do not use yet |
| Optional TODO list | Deep Agents | Helps one parent Agent track actions | Not a source of research truth | Not needed as Core state |
| Latest Dispatch Batch only | No reviewed system uses it as the sole memory for iterative research | Small prompt | Forgets earlier findings and can repeat work | Do not use |
| Explicit result references | Common application-level technique; compatible with all frameworks | Selective multi-hop context | Needs lookup/materialization rules | Use Assignment references; no Coordinator lookup Tool yet |
| Filesystem/offloaded notes | Deep Agents; Open Deep Research keeps raw notes outside supervisor messages | Supports very large variable content | Adds storage and retrieval policy | Defer |

## Why the proposed ResearchLedger is premature

The proposed fields were:

```python
class ResearchLedger(BaseModel):
    completed_goals: list[str]
    remaining_goals: list[str]
    unresolved_gaps: list[str]
    result_refs: list[ResultId]
```

This appears small, but it introduces unresolved ownership questions:

- If the Coordinator Agent writes it, it can omit or rewrite an earlier fact.
- If deterministic code writes it, `completed_goals` and `unresolved_gaps` do
  not have deterministic meanings without a richer goal model.
- If it is mutable, it conflicts with immutable Coordination Rounds as the
  auditable record.
- If both the ledger and Task Outcomes describe progress, they can disagree.
- If a summary replaces earlier Specialist Results, information can be lost
  before the Synthesis Agent sees it.

The existing bounded records already answer the control questions:

- Coordinator Decisions show what was requested.
- Task Outcomes show what succeeded or failed.
- Specialist Results show the bounded findings and reported gaps.
- stable IDs show which earlier results a later Assignment used.
- Run counters show what budget remains.

A `ResearchLedger` becomes useful only after traces show that this canonical
record is too large, or that the Coordinator often fails to identify open work.
At that point it should be defined as a derived, replaceable model-input summary,
not as the source of truth.

## Concrete round-by-round example

User query:

> Compare the main reasons for NVDA and AMD's latest quarterly margin changes.

### Coordination Round 1 input

```text
Query: Compare the main reasons for NVDA and AMD's latest quarterly margin changes.
Specialists: Filing Research; Market Analysis; News Research
Prior rounds: none
Remaining limits: 3 follow-up rounds, 32 Tasks
```

The Coordinator Decision dispatches two independent Assignments:

```text
A1 -> Filing Research: find NVDA margin change and management explanation
A2 -> Filing Research: find AMD margin change and management explanation
```

### Coordination Round 2 input

The Agent Graph reconstructs the next input in stable order:

```text
Query: ...
Round 1 decision:
  A1 Filing Research, objective: ...
  A2 Filing Research, objective: ...
Round 1 Task Outcomes:
  A1 success; summary: NVDA gross margin ...; evidence: [E1, E2]; gaps: [...] 
  A2 success; summary: AMD gross margin ...; evidence: [E3, E4]; gaps: [...] 
Remaining limits: 2 follow-up rounds, 30 Tasks
```

The Coordinator now sees that the reports give different accounting periods.
It dispatches a follow-up Assignment:

```text
A3 -> Filing Research: align the compared fiscal periods and explain differences
      relevant_result_ids=[A1, A2]
```

### Coordination Round 3 input

```text
Query: ...
Round 1 decision and bounded outcomes: A1, A2
Round 2 decision and bounded outcome: A3
  A3 success; summary: aligned comparison ...; evidence: [E1, E3, E5]
Remaining limits: 1 follow-up round, 29 Tasks
```

The Coordinator returns `Finish`. The Synthesis Agent then receives the accepted
Specialist Results and the separately materialized eligible Evidence needed for
claim-level citations.

At no point does the Coordinator receive raw filing documents, web-search
responses, Specialist Tool histories, or a second mutable Research Ledger.

## When to add compaction

Add a derived Coordinator context summary only after one of these conditions is
measured:

- the canonical bounded context approaches the selected model's input limit;
- prompt cost or latency exceeds an agreed threshold;
- the Run limits increase enough that all earlier summaries are no longer small;
- traces show repeated work because the Coordinator cannot track the record.

When added, preserve immutable Coordinator Decisions, Task Outcomes, Specialist
Results, and Evidence as source records. Compaction must change model input only.
It must not become research truth or erase audit history.
