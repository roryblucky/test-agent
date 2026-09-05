# Query Understanding clarification boundary

Date: 2026-09-03

## Question

When Query Understanding cannot safely resolve the user's request or select a
Business Intent, should the platform:

1. complete the current Request with a user-visible clarification question and
   process the answer as a new Request;
2. pause and later resume the same LangGraph execution through an interrupt;
3. fail the Request; or
4. force the model to guess?

This note compares current open-source implementations and applies them to this
repository's `/v2/query/stream`, Conversation, and checkpoint boundaries.

## Conclusion

**Recommend option 1 for the POC.** Clarification is a successful terminal
business outcome of the current Request, not an execution failure and not a
paused Run:

1. Query Understanding returns a validated clarification result.
2. Graph does not resolve Research Scope, invoke Coordinator, dispatch
   Specialists, or run Synthesis.
3. The clarification question is published as the final assistant message and
   included in the existing `done.clarification` wire field.
4. The terminal `done` event is released only after the checkpoint containing
   both the current user message and clarification assistant message commits.
5. The client sends the user's answer as an ordinary new
   `/v2/query/stream` Request with the same Conversation ID and a new Request
   ID.
6. Query Understanding receives the bounded complete prior pair and resolves a
   new standalone query and Business Intent. Request-local coordination,
   research, evidence, and error state starts empty as already required by the
   Agent spec.

In domain terms the current **Request completes**, while the user-level
**Conversation continues**. No suspended Task, pending Tool call, resume token,
or resumable node state exists.

This matches Open Deep Research more closely than LangGraph interrupts. It also
preserves the POC decision that a checkpointer is injected without exposing
HITL/resume behavior.

## Primary-source findings

### Open Deep Research ends the current execution

Open Deep Research has a dedicated `clarify_with_user` node. It asks a model for
the structured `ClarifyWithUser` result. If clarification is required, the node
returns:

```python
Command(
    goto=END,
    update={"messages": [AIMessage(content=response.question)]},
)
```

It therefore records an ordinary assistant message and ends that graph
invocation; it does **not** call `interrupt()` and does not classify ambiguity
as an error
([Open Deep Research clarification node](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L1734-L1824)).

Its clarification prompt reads all exchanged messages and explicitly checks
whether a clarification was already asked. The next user answer is therefore
interpreted through Conversation history, not through a resume payload
([Open Deep Research clarification prompt](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/prompts.py#L932-L990)).

The same node also has an `allow_clarification` configuration switch. Disabling
it routes directly to research-brief generation. That is a conscious product
policy for deployments willing to proceed with incomplete detail, not an
error-recovery mechanism or an argument that the default should silently guess
([Open Deep Research clarification node](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L1752-L1760)).

### LangGraph interrupt preserves an unfinished execution

LangGraph documents `interrupt()` as a HITL primitive that saves the current
graph state, surfaces a JSON-serializable pending value, and waits indefinitely.
The caller must later invoke the graph with the same `thread_id` and
`Command(resume=...)`; the resume value becomes the return value of the earlier
`interrupt()` call
([LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts#pause-using-interrupt),
[resuming interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts#resuming-interrupts)).

This has material runtime semantics. On resume, LangGraph restarts the
interrupted node from its beginning, so code before the interrupt runs again.
Interrupt order and side effects must therefore be made replay-safe
([LangGraph interrupt rules](https://docs.langchain.com/oss/python/langgraph/interrupts#rules-of-interrupts)).

Those mechanics are appropriate when the same execution must remain pending,
for example approval of a prepared Tool call or review/edit of generated state.
They add no value when clarification occurs before Research Scope and Tasks
exist. Using an interrupt here would create a second public request shape
(`Command(resume=...)`), suspended-Run semantics, and node replay obligations
solely to reproduce normal multi-turn Conversation behavior.

### OpenAI Agents SDK treats a displayed answer and follow-up as separate turns

The OpenAI Agents SDK defines one `Runner.run*()` call as one logical chat turn,
even if several agents or Tool calls execute inside it. Once the Run produces
final output, the application shows that output; a later user follow-up starts
another Runner call. History can be carried by `to_input_list()`, a Session, a
Conversation ID, or a previous response ID
([OpenAI Agents SDK turn semantics](https://openai.github.io/openai-agents-python/running_agents/#conversationschat-threads),
[state and conversation strategies](https://openai.github.io/openai-agents-python/running_agents/#choose-a-memory-strategy)).

This supports modeling a clarification question as the current turn's final
assistant output and the user's answer as the next turn. The SDK separately
supports resuming paused Runs; that is not required merely because a later turn
uses Conversation history.

### PydanticAI deferred output is for pending Tool work, not query ambiguity

PydanticAI's stop-the-world flow ends an Agent run with
`DeferredToolRequests` when validated Tool calls await approval or external
execution. The caller then starts a separate Agent run with the original
message history plus `DeferredToolResults`; the follow-up has a new `run_id`
and correlation is retained through `conversation_id`
([PydanticAI deferred tools](https://github.com/pydantic/pydantic-ai/blob/main/docs/deferred-tools.md#deferred-tools)).

That contract preserves pending Tool-call IDs and supplies matching results.
Query clarification has no pending Tool call and no result that must be
injected at a particular point in a model transcript. Reusing deferred-tool or
HITL machinery would therefore manufacture lifecycle state absent from the
business problem.

## Why the other choices are weaker

### Interrupt/checkpoint resume

Reject for this POC. A checkpointer does not make every multi-turn interaction
an interrupt. Here it durably stores completed logical Conversation messages;
the next Request reconstructs Query Understanding context from them. Interrupt
resume should remain an extension point for a future execution that genuinely
has pending work.

### Failure/error

Reject for valid ambiguity. The model successfully produced a typed and
validated business result: more information is required. Reserve `error` for
invalid structured output after its configured repair/retry, provider/runtime
failure, checkpoint failure, authorization failure, or invariant violation.
Treating clarification as failure would make normal dialogue look unreliable
to clients and observability.

### Forced guess

Reject as the default for an enterprise information platform because an
incorrect resolved subject or Intent can narrow the run to the wrong data and
Tools. A Tenant may later opt into an explicit no-clarification policy like
Open Deep Research's `allow_clarification=False`, but that is separate from the
POC's default semantics. Even then, the model must not exceed deterministic
Tenant and Research Scope enforcement.

## Repository fit and current gaps

The legacy Flow Engine already implements almost the recommended business
branch. `_apply_query_understanding_clarification` creates a user-facing
`clarification_request`, places its first question in `llm_response`, sets
`metadata.stop_flow`, and records a specific stop reason. The engine then
breaks before later steps
([LLM handler](../../app/services/handlers/llm.py),
[Flow Engine](../../app/services/flow_engine.py)).

The public compatibility contract already reserves
`done.clarification: object | null` and `done.answer: string | null`
([v2 UAT contract](../../tests/fixtures/langgraph_v2/v2_uat_contract.json)).
However, the clean-room Linear response currently narrows `clarification` to
`None`, and the new Agent Graph is not implemented yet
([v2 contracts](../../app/langgraph_v2/contracts.py)). The Agent-mode response
contract must restore the existing structured clarification shape rather than
invent an interrupt event or a second endpoint.

The existing finalization/checkpoint boundary is also directly reusable: a
terminal `done` event is buffered until a Graph update confirms the synchronous
checkpoint, and only a non-null final answer currently adds the assistant
Conversation message
([stream finalization](../../app/langgraph_v2/stream.py),
[Graph finalization](../../app/langgraph_v2/graph.py)). For clarification, the
Graph must likewise persist the clarification text as the final assistant
message before releasing `done`. A terminal-checkpoint failure must not publish
`done` or leave a one-sided completed pair, consistent with the existing
finalization test
([finalization tests](../../tests/integration/test_langgraph_v2_finalization.py)).

## Recommended exact POC contract

- Query Understanding output remains a mutually interpreted result:
  - `clarification is None`: `standalone_query` and selected Intent proceed to
    deterministic validation and Research Scope resolution;
  - `clarification is not None`: the validated clarification wins and the Graph
    routes directly to clarification finalization. Any simultaneously generated
    resolved-query or Intent values are not used to authorize or execute work.
- Emit normal progress/step-completion events as appropriate, then one terminal
  `done`; do not emit `error`, `stopped`, or a LangGraph interrupt.
- In `done`, set `clarification` to the structured request, `answer` to the same
  user-visible clarification response for compatibility with the existing
  legacy behavior, `citations=[]`, and no research documents or groundedness
  result.
- Persist exactly one user message and one final assistant clarification
  message under the current Request ID.
- The user's answer uses the same Conversation ID and a new Request ID and
  starts a clean Agent Run. It is not a `resume` request.
- Add an E2E that verifies no Coordinator/Specialist/Synthesis call occurs, the
  terminal pair is checkpointed, and the next Request can resolve the referent
  from bounded Conversation history.

This keeps clarification ordinary at the product boundary while leaving true
LangGraph interrupt/HITL resume available for future pending-execution cases.
