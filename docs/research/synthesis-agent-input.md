# Synthesis Agent input and incomplete research

Date: 2026-09-03

> Superseded: the proposed `report_scope` and global `known_gaps` fields were
> rejected after independent review because the POC has no deterministic
> business-coverage contract. See
> [Minimal synthesis input: independent review](synthesis-known-gaps-review.md).
> This note remains as research history rather than the accepted design.

## Question

Does the Synthesis Agent need orchestration state such as:

- `sufficient`, `cannot_progress`, or `budget_exhausted`;
- Task status, errors, and retry attempts;
- token, Tool-call, or time budgets;
- Specialist gaps and warnings?

This note distinguishes information held by the Agent Graph from information
that is placed in the Synthesis Agent prompt.

## Conclusion

Current open-source research systems usually give the final Writer or Synthesis
Agent the user request and accepted research content. They do not give it raw
orchestration state such as retry counts, remaining budgets, provider errors,
or a detailed termination reason.

That separation is correct for this repository. The Synthesis Agent writes a
report. It does not diagnose the execution runtime.

However, copied community examples can hide partial research and produce a
complete-sounding report. The Agent Graph must not silently remove a failed
Assignment if that failure leaves a material information gap. It should convert
the canonical Task Outcomes into a small, report-relevant synthesis input:

```python
class SynthesisBrief(BaseModel):
    query: str
    findings: tuple[SpecialistFinding, ...]
    evidence: tuple[EvidenceSummary, ...]
    report_scope: Literal["complete", "partial"]
    known_gaps: tuple[str, ...]
```

`report_scope` is not the Coordinator's raw `Finish.reason`. It is a
deterministic publication instruction derived by the Agent Graph. It tells the
Synthesis Agent whether it must present the answer as partial. `known_gaps`
contains only user-relevant missing information. It must not contain provider
stack traces, retry counters, or internal budget details.

Keep these fields outside the Synthesis Agent prompt:

- `Finish.reason`;
- Assignment attempt count;
- timeout and provider error details;
- used and remaining token counts;
- used and remaining Tool calls;
- internal Coordinator Decisions;
- Specialist Tool-call and message history.

Keep them in Graph state and observability records.

## What primary implementations do

### Open Deep Research

The current Open Deep Research final report node joins the collected `notes`
and formats its prompt with four values: `research_brief`, user `messages`,
`findings`, and the date. The Writer does not receive the supervisor iteration
count, Task statuses, retry counts, budgets, or a termination reason
([final report node](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py),
[final report prompt](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/prompts.py)).

The supervisor can stop because it used the `ResearchComplete` Tool, emitted no
Tool calls, or reached a configured iteration limit. Those paths feed notes and
the research brief into the outer graph. They do not create a model-visible
completion-state object. This shows a strong content/control separation, but it
also means the Writer cannot distinguish complete research from research stopped
by a limit unless that difference appears in the findings.

### OpenAI Agents SDK research examples

The OpenAI Agents SDK `research_bot` tracks successful and failed searches for
progress reporting. Failed searches are omitted. The Writer receives the
original query and successful search summaries. It does not receive the failed
search count, exceptions, retries, budget, or termination reason
([research manager](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/manager.py),
[example README](https://github.com/openai/openai-agents-python/tree/main/examples/research_bot)).

The `financial_research_agent` follows the same boundary. Its report input
contains the original query, research cutoff, and accepted evidence. Search
exceptions are handled before report generation and are not included in the
Writer input. A separate verifier checks the report against the evidence; it
also does not need retry or budget state
([financial research manager](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/manager.py),
[Writer Agent](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/agents/writer_agent.py),
[Verifier Agent](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/agents/verifier_agent.py)).

### GPT Researcher

GPT Researcher builds the report request from the research query and prepared
context. The report generator accepts presentation settings such as report
type, tone, and format, but it does not take per-search status, retry count,
budget, or a workflow termination reason
([report generation source](https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/actions/report_generation.py),
[Writer source](https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/skills/writer.py)).

Its deep-research example removes failed parallel results and then gives the
Writer the remaining learnings and citations. This is simple, but it can hide
coverage loss unless the application records the missing topic as a report
limitation
([deep-research example](https://github.com/assafelovic/gpt-researcher/blob/master/backend/report_type/deep_research/example.py)).

### STORM

STORM's article-generation stage receives the topic, an outline, and retrieved
information. Each section Writer receives the topic, the section definition,
and bounded snippets with citation numbers. It does not receive crawler errors,
retry state, budgets, or a completion reason
([article generation source](https://github.com/stanford-oval/storm/blob/main/knowledge_storm/storm_wiki/modules/article_generation.py)).

STORM therefore treats synthesis input as a curated knowledge package, not an
execution transcript.

### AutoGen Magentic-One

Magentic-One is a useful edge case. The orchestrator calls its final-answer
method both when the request is assessed as complete and when the maximum number
of rounds is reached. The method receives a textual `reason`, but the source
does not insert that reason into the final-answer prompt. The model sees the
original task, the accumulated message thread, and a prompt that says the task
is complete. The reason is used later in the termination event
([orchestrator source](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py),
[prompts](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_prompts.py)).

This confirms that raw termination state is not required for synthesis. It also
shows why this repository should add the small `report_scope` safeguard: a
Writer should not be told that work is complete when the runtime stopped it at
a hard limit.

### Deep Agents

Deep Agents uses subagents for context isolation. The parent receives a concise
final subagent result instead of the subagent's Tool history. Optional structured
responses can make that result machine-readable, but runtime counters do not
need to become report content
([Subagents](https://docs.langchain.com/oss/python/deepagents/subagents),
[context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering)).

Deep Agents does not prescribe a separate final Writer contract. Its context
guidance still supports the same rule: pass the result needed for the next task,
not the internal execution trace.

## Runtime state versus synthesis input

| Information | Agent Graph | Synthesis Agent | Reason |
|---|---:|---:|---|
| User query or Research Brief | Yes | Yes | Defines the report goal |
| Accepted bounded Findings | Yes | Yes | Supplies report content |
| Eligible Evidence summaries | Yes | Yes | Supports citations and verification |
| User-relevant information gaps | Yes | Yes | Prevents overclaiming |
| `complete` or `partial` report scope | Yes | Yes | Controls disclosure |
| `sufficient` / `cannot_progress` / `budget_exhausted` | Yes | No | Controls the Run, not report prose |
| Assignment status and attempts | Yes | No | Operational metadata |
| Provider and Tool errors | Yes | No | Operational metadata; may contain noise or secrets |
| Token, Tool-call, and time budgets | Yes | No | Enforced by code |
| Specialist internal messages | No durable Core record | No | Violates context isolation |

## Recommended behavior by scenario

### Research is sufficient

Build `SynthesisBrief(report_scope="complete", known_gaps=())`. Invoke the
Synthesis Agent and apply deterministic Evidence and publication gates.

### One Assignment fails but coverage remains sufficient

Keep the failed Task Outcome for audit. Do not expose the technical failure to
the Synthesis Agent. If another Finding covers the same required topic, use
`report_scope="complete"`.

### One Assignment fails and leaves a material gap

Convert the missing business topic into `known_gaps`, for example, “No eligible
source confirmed the segment revenue for fiscal 2025.” Use
`report_scope="partial"`. Do not pass “HTTP 429 after attempt 2.”

### The budget is exhausted after useful Evidence was found

The raw `Finish.reason="budget_exhausted"` stays in Graph state. Build a partial
Synthesis Brief from accepted Findings and known gaps. The report must disclose
the missing coverage, not the internal token budget.

### No eligible Evidence exists

Do not ask the Synthesis Agent to improvise a research report. The Agent Graph
should return a deterministic insufficient-evidence result, or call a separately
constrained response path that cannot make factual claims. This is safer than
passing an empty context to a general Writer.

### The runtime fails because of an invariant violation

Do not invoke the Synthesis Agent. Fail the Run. A programming error, state
conflict, or authorization breach is not an information gap.

## Recommendation for the first end-to-end implementation

Use one narrow `SynthesisBrief` boundary. Derive it after the final batch barrier
and before the Synthesis Agent runs.

The derivation should be deterministic:

1. Select only successful, accepted Specialist Findings.
2. Resolve and validate their Evidence IDs.
3. Collect and deduplicate the Findings' business `gaps` and `warnings`.
4. Decide `report_scope` from required research coverage, not from exception
   type.
5. If there is no eligible Evidence, bypass normal synthesis.
6. Keep all technical status, retry, and budget data in Task Outcomes and Run
   observability.

The POC does not need separate behavior for `cannot_progress` and
`budget_exhausted` inside the Synthesis Agent. Both may produce either a partial
report or an insufficient-evidence response. The available Evidence and material
gaps determine which result is safe.
