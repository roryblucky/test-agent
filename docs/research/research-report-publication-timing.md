# Research Report Publication Timing in Open-Source Agent Applications

Date: 2026-09-03

## Question

When a research application generates a final report, does it publish model tokens
immediately, wait for the complete report (and any verifier or deterministic gates), or
use a draft/revoke protocol? This note separates framework streaming capability from
application publication behavior and evaluates the three options proposed for the Agent
mode of `/v2/query/stream`.

## Conclusion

There is no single open-source convention for all research applications. Two patterns
appear in the inspected primary sources:

1. Applications that treat the model output as the finished artifact commonly assemble
   or validate the complete artifact before exposing it.
2. Applications that stream report text immediately do so without a later blocking
   referential-integrity gate and without a draft/revoke protocol.

The closest precedent to this repository is OpenAI's `financial_research_agent`: it uses
a streaming model call internally but consumes the events only to show generic progress,
waits for the complete typed report, verifies it, optionally revises and verifies again,
and prints the report only after successful verification. Therefore option A—progress
during synthesis, then release the canonical report after deterministic gates—is the
best fit for the POC. It is not a universal community rule; it follows from this repo's
stronger publication invariant.

## Application evidence

| Application | Model/framework streaming capability | Actual final-report publication behavior | Post-generation gate | Draft/revoke protocol |
| --- | --- | --- | --- | --- |
| OpenAI Agents SDK `research_bot` | Calls `Runner.run_streamed()` | Consumes stream events without forwarding their token payloads, emits rotating progress text, then returns `final_output_as(ReportData)` and prints the complete Markdown report | Pydantic/SDK final-output parsing; no citation verifier | No |
| OpenAI Agents SDK `financial_research_agent` | Calls `Runner.run_streamed()` | Consumes stream events as progress only; obtains complete `FinancialReportData`; verifies, optionally revises and verifies again; displays the report only after success | LLM verifier, not a deterministic citation gate | No |
| LangChain Open Deep Research | LangGraph can stream graph/model events | Final report node is tagged `langsmith:nostream`, calls `ainvoke()`, and stores `final_report.content` after the complete response | No separate final verifier in the inspected graph | No |
| STORM | Underlying LMs may support streaming | Generates complete sections, merges them into an article, runs `post_processing()`, then returns/dumps the article; no token publication path appears in the article-generation module | Deterministic article post-processing, but not a strict evidence-eligibility gate | No |
| GPT Researcher | Provider abstraction supports `stream=True` and a websocket | Final report generation passes `stream=True` and the websocket, so generated text may be sent while the model is producing it | No subsequent blocking referential-integrity gate in this path | No |

### OpenAI `research_bot`

The writer has a typed `ReportData` output containing `short_summary`,
`markdown_report`, and follow-up questions. The manager invokes the writer with
`Runner.run_streamed()`, but its `async for _ in result.stream_events()` loop ignores
event bodies and only advances generic status messages every five seconds. Only after the
stream ends does it call `result.final_output_as(ReportData)`, and `run()` prints
`report.markdown_report` afterward.

Sources:

- [`research_bot/manager.py`](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/manager.py)
- [`research_bot/agents/writer_agent.py`](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/agents/writer_agent.py)

This is an important distinction: use of a streaming SDK API does not imply publication
of raw output tokens to the user.

### OpenAI `financial_research_agent`

This example is more relevant because its application pipeline explicitly includes
verification. `_write_report()` consumes the writer's stream while showing progress and
returns the complete `FinancialReportData`. `_produce_verified_report()` then runs a
verifier; if it rejects the report, the manager generates a complete revision and verifies
again. A second rejection raises instead of publishing the candidate. The report is
printed only after `_produce_verified_report()` returns successfully.

Source:

- [`financial_research_agent/manager.py`](https://github.com/openai/openai-agents-python/blob/main/examples/financial_research_agent/manager.py)

The verifier is model-based, so this example does not establish a deterministic factual
guarantee. It does establish the relevant publication order:

```text
generate candidate -> verify -> optionally revise -> verify -> display
```

It does not publish a draft and later revoke it.

### Open Deep Research

The `final_report_generation` LangGraph node constructs the complete report prompt, calls
the final-report model with `ainvoke()`, and writes `final_report.content` into graph state.
Its model configuration includes the `langsmith:nostream` tag. The graph then ends. This
application therefore generates the final report as one completed graph-state value even
though LangGraph itself is capable of streaming events.

Source:

- [`open_deep_research/deep_researcher.py`](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py)

There is no separate citation verifier after this writer node in the inspected graph, so
the example supports buffering as an application choice but not this repository's exact
gate semantics.

### STORM

STORM generates article sections concurrently, collects the completed section outputs,
updates a copied `StormArticle`, calls `article.post_processing()`, and only then returns
the article. Its engine subsequently dumps the complete article and reference map. This is
artifact-oriented generation rather than end-user token streaming.

Sources:

- [`article_generation.py`](https://github.com/stanford-oval/storm/blob/main/knowledge_storm/storm_wiki/modules/article_generation.py)
- [`engine.py`](https://github.com/stanford-oval/storm/blob/main/knowledge_storm/storm_wiki/engine.py)

STORM's post-processing is not equivalent to the proposed Tenant/Run/Evidence/Calculation
marker gate, but it again demonstrates that complete-artifact assembly can precede
publication.

### GPT Researcher

GPT Researcher's final `generate_report()` call passes both `stream=True` and the active
websocket to its provider abstraction. Its writer awaits that call and returns the report.
Unlike OpenAI's financial example, this path has no subsequent blocking verifier or
deterministic referential-integrity gate. It is therefore evidence that immediate report
streaming is used in open-source research UX, but not evidence that immediate streaming is
compatible with a strict post-generation publication guarantee.

Sources:

- [`actions/report_generation.py`](https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/actions/report_generation.py)
- [`utils/llm.py`](https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/utils/llm.py)
- [`skills/writer.py`](https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/skills/writer.py)

No inspected application defines a protocol in which already displayed final-report text
is later revoked.

## Framework capability is not a publication guarantee

### PydanticAI

PydanticAI supports streaming partial text and structured output. Its documentation states
that output validators run on partial values and once on the complete value; validators
can use `RunContext.partial_output` to defer complete-only checks. With
`stream_text(delta=True)`, raw text deltas skip validators. Thus the framework can expose
partial output, but a validation rule that requires the complete report cannot certify an
earlier delta. PydanticAI also advises deferring side effects until `partial_output` is
false.

Source:

- [PydanticAI output and streaming documentation](https://pydantic.dev/docs/ai/core-concepts/output/)

### OpenAI Agents SDK

The Agents SDK exposes raw output-text delta events, but `final_output` remains `None`
until the stream finishes. Its output guardrails run on the final output after generation.
The application decides whether raw events are merely observed, displayed as progress, or
forwarded to the user. The two OpenAI research examples deliberately consume events
without forwarding final-report deltas.

Sources:

- [Agents SDK streaming](https://openai.github.io/openai-agents-python/streaming/)
- [Agents SDK result lifecycle](https://openai.github.io/openai-agents-python/results/)
- [Agents SDK guardrails](https://openai.github.io/openai-agents-python/guardrails/)

OpenAI's Guardrails Python documentation names the underlying trade-off directly:
non-streaming waits for all guardrails before showing output; streaming lowers latency but
may expose content before an output guardrail rejects it. This is a general safety example,
not the Agents SDK research application, but the publication timing principle is the same.

Source:

- [OpenAI Guardrails Python: streaming vs. blocking](https://openai.github.io/openai-guardrails-python/streaming_output/)

## Evaluation of the proposed options

### A. Progress while generating; publish report after all deterministic gates

**Recommended.**

This matches the strongest inspected research precedent (`financial_research_agent`) and
is the only option that makes the POC's stated invariant literally true: an invalid inline
Evidence or Calculation marker is never published. It does not require a new public SSE
event type. Existing `progress`, `token`, `citations`, `error`, and `done` events are enough.

The expected Agent-mode order can remain additive and wire-compatible:

```text
progress...                    # planning, tasks, tools, synthesis, gate
token...                       # canonical validated Markdown, chunked after approval
citations                      # resolved Evidence references
done                           # same canonical answer
```

The `token` events still preserve the established event shape; they are transport chunks,
not necessarily live provider tokens. The trade-off is time-to-first-answer-token. Progress
events must make the wait legible, and the implementation should impose a report-size bound.

### B. Forward live Markdown deltas; send `error` if the final gate rejects

**Valid only if the gate is advisory; incompatible with the current strict invariant.**

This resembles GPT Researcher's user experience and the existing Linear mode's advisory
post-assessment behavior. However, once a client has rendered a token, a later SSE error
cannot unpublish it. Choosing B would require changing the product claim from “invalid
markers are never published” to “the final canonical answer is validated, but drafts may
have been exposed.” It is not merely an implementation choice.

### C. Add explicit draft/revoke SSE semantics

**Not recommended for the POC.**

No inspected application provides this protocol. It adds client state, message identity,
draft/canonical distinction, revoke ordering, reconnect/replay behavior, persistence rules,
and backward-compatibility work. It is only justified if live report-token latency is a
confirmed product requirement and clients are designed to render provisional content.

## Repository-specific note

The existing v2 contract already has `progress`, `token`, `citations`, `error`, and `done`
event types in `app/langgraph_v2/contracts.py`. Linear mode intentionally streams answer
tokens before advisory groundedness and post-moderation finish, as documented in
`docs/langgraph_v2_post_moderation.md`. That behavior should not be generalized to Agent
mode because the proposed Agent-mode marker check is a blocking publication gate, not an
advisory assessment.

Accordingly, option A should be scoped to Agent-mode synthesis. It can preserve the same
endpoint and SSE schema while applying different publication timing based on the Tenant's
server-side graph configuration.
