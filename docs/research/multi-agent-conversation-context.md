# Conversation-history visibility in multi-agent systems

Date: 2026-09-03

## Question

Which model-facing actors should see prior user-visible Conversation messages
in this project's Coordinator–Specialist–Synthesis topology?

The options under review were:

- **A**: Coordinator and Synthesis receive bounded complete prior
  user/final-assistant pairs; a Specialist receives only its current objective
  and selected prior Task Results.
- **B**: only the Coordinator receives prior Conversation messages.
- **C**: every agent receives prior Conversation messages.
- **D**: every new query must be self-contained; no agent receives prior
  Conversation messages.

## Conclusion

There is no topology-independent open-source best practice that says either
"share all history" or "share no history." Current implementations make
history visibility follow the agent relationship:

| Topology | Typical history boundary |
|---|---|
| Manager / agent-as-tool | Manager owns the user conversation; a delegate gets a generated, self-contained task and returns a result |
| Orchestrator / worker | Orchestrator builds explicit worker state; each worker receives only that state |
| Handoff | The receiving agent becomes the active conversational agent and normally receives prior conversation history, subject to filtering |
| Group chat | Participants share or receive the common agent-to-agent message thread |

This project is a manager/orchestrator-worker system, not a handoff or group
chat. Therefore **C is the wrong default**. Specialist isolation is strongly
supported by LangChain, Deep Agents, OpenAI Agents SDK, and Open Deep Research.

The four proposed choices omit the best POC option:

- **E' (recommend for the POC)**: the existing Query Understanding actor alone
  receives bounded complete prior user/final-assistant pairs, the current query,
  and compact Tenant Intent Catalog descriptions. It produces the existing
  non-empty `V2ResolvedQuery.standalone_query` plus a typed selected Business
  Intent. Coordinator and Synthesis receive that resolved current-Run input, not
  Conversation history. A Specialist receives only its current Task objective,
  selected accepted prior Specialist Results, and trusted Research Scope.

Graph/runtime code may carry dependencies, policy, IDs, usage, and other
execution state without making that state visible to a model. Internal
Coordinator, Specialist, Tool, retry, and validation transcripts must not enter
the product Conversation or another actor's message history.

E' is not a new model-facing abstraction. The repository already has a combined
`QueryUnderstandingOutput(resolved_query, intent, clarification)` contract and
Query Understanding Agent, while ADR-0002 already places request refinement in
the v2 baseline before downstream work. The Agent Graph should reuse that
boundary as its request-understanding entry instead of projecting history
independently into each downstream actor
([Query Understanding Agent](../../app/agents/query_understanding.py),
[query contracts](../../app/models/workflow.py),
[ADR-0002](../adr/0002-replace-the-legacy-flow-engine-through-new-endpoints.md)).

Open Deep Research implements a related but not identical shape: it first
converts user messages to a structured research brief, gives the supervisor the
brief rather than the full user transcript, and creates each researcher with
only a generated `research_topic`. Its final writer receives both the brief and
the original messages. That writer choice is useful evidence that A can
preserve conversational nuance, but it is not a universal requirement. In this
project, an already-resolved standalone query and strict current-Run support
boundary make E' preferable: it removes old assistant prose from Synthesis
instead of relying only on a prompt to say that prose is non-Evidence.

## Three different things called "context"

The design should not collapse these into one history list.

### 1. User Conversation history

This is the bounded sequence of user messages and final publishable assistant
messages from prior Requests. It helps resolve references such as "compare it
with the benchmark". It is model-visible only when application code explicitly
projects it into an actor's input.

### 2. Runtime context and dependencies

This is application state such as Tenant policy, clients, registries, request
identity, per-call timeout policy, and usage accounting. It is available to orchestration
code and tools but is not automatically model-visible. OpenAI Agents SDK makes
the distinction explicit: `RunContextWrapper.context` is local application
data and "is not sent to the LLM"; LLM context must instead be placed in
instructions, input, history, or tool results
([OpenAI Agents SDK context management](https://openai.github.io/openai-agents-python/context/)).

PydanticAI similarly makes continuity application-owned: another agent receives
earlier messages only when the application passes `message_history`; dependency
objects are a separate run input
([PydanticAI multi-agent applications](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md#programmatic-agent-hand-off),
[PydanticAI message history](https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#sharing-messages-between-agents)).

### 3. Internal agent and Tool transcripts

These are actor-local model responses, reasoning/tool calls, Tool results,
retries, and validation/repair exchanges. They can be retained in trace or
request-local execution state, but they are not synonymous with the user's
Conversation. PydanticAI warns that shared model history can retain Tool-call
and Tool-return parts and recommends passing only messages the receiving agent
can understand; this supports the project's projection boundary rather than
reusing raw actor histories
([PydanticAI sharing messages between agents](https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#sharing-messages-between-agents)).

AutoGen also explicitly distinguishes agent-to-agent chat messages from an
agent's internal events. That distinction remains necessary even in systems
that choose a shared group thread
([AutoGen messages](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/messages.html)).

## Primary-source findings

### LangChain subagents and Deep Agents: parent owns memory

LangChain's current subagent documentation states that the main agent maintains
conversation memory while subagents are stateless and start in clean context
windows. The parent chooses the subagent input and combines returned results.
The minimal example invokes the subagent with only a new user message containing
the delegated query. Passing full history or other state is supported, but it
is an explicit application transformation rather than the default
([LangChain subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)).

Deep Agents describes subagents as a context-quarantine mechanism: a main agent
delegates a task and receives a concise result without the subagent's
intermediate Tool traffic bloating the main context. Its skill state is also
isolated for custom subagents unless explicitly configured
([Deep Agents subagents](https://docs.langchain.com/oss/python/deepagents/subagents)).

This directly supports the proposed Specialist boundary: the Coordinator sees
results, while each Specialist receives a self-contained objective and only the
prior results selected for that objective.

### LangGraph orchestrator-worker: explicit per-worker state

LangGraph's orchestrator-worker example uses `Send` to build a separate
`WorkerState` for each generated unit of work. Each worker gets its section,
writes its output to a reducer-backed shared key, and the synthesizer consumes
the collected outputs. The example does not automatically copy an orchestrator
message history into each worker
([LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents#orchestrator-worker)).

LangGraph therefore provides transport and state mechanics, not a universal
history policy. The application decides exactly which validated values are put
in each `Send` payload. This supports constructing Specialist input from the
Task objective and materialized `context_task_ids`, rather than giving every
Specialist the shared Conversation state.

### Open Deep Research: separate brief, researcher, and writer contexts

Open Deep Research turns the user `messages` into a structured
`research_brief`, then initializes the supervisor with a system message and a
single human message containing that brief
([brief construction and supervisor initialization](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L1874-L1914)).

When the supervisor delegates, each researcher subgraph is initialized with
only the generated `research_topic`, not the user's full Conversation or the
supervisor's internal transcript. Researchers return compressed research to the
supervisor as Tool results
([researcher dispatch](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L2105-L2136)).

The final writer is different: its prompt includes the research brief, the user
messages, and the collected findings
([final report generation](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L2589-L2660)).

This is the closest first-party analogue to this project. It supports isolated
Specialists, a context-aware Synthesis actor, and a future normalized brief. It
does not support sending the raw user and supervisor/tool transcript to every
researcher.

### OpenAI Agents SDK: agent-as-tool and handoff are intentionally different

The OpenAI Agents SDK documents two distinct patterns. With agents-as-tools, a
manager keeps control of the user conversation and calls Specialists for
bounded work. With handoffs, the receiving Specialist becomes the active agent
for the rest of the turn
([OpenAI Agents SDK orchestration](https://openai.github.io/openai-agents-python/multi_agent/)).

The API reference makes the visibility difference explicit: a handoff agent
receives conversation history, while an agent used as a Tool receives generated
input and the original agent continues the conversation
([OpenAI `Agent.as_tool`](https://openai.github.io/openai-agents-python/ref/agent/#agents.agent.Agent.as_tool)).

For handoffs, full prior history is the default because ownership of the active
conversation moves to the new agent. The SDK provides input filters when the
application wants to remove Tool activity or otherwise control the receiving
history
([OpenAI handoff input filters](https://openai.github.io/openai-agents-python/handoffs/#input-filters)).

This project's Specialists do not take over the user conversation, so the
agent-as-tool semantics—not handoff semantics—are the relevant comparison.

### PydanticAI: history transfer is explicit

PydanticAI treats agents as stateless reusable objects. Delegation calls another
agent from a Tool and returns control to the parent; programmatic handoff calls
agents in succession. In either case, passing previous conversation context is
an application decision made with the `message_history` argument
([PydanticAI multi-agent applications](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md)).

The message-history guide also notes that Tool calls and returns remain in a
shared history, so applications should pass only the parts meaningful to the
receiving agent. This supports the existing ADR-0005 projection of logical
user/final-assistant pairs instead of exposing raw PydanticAI actor transcripts
([PydanticAI message history](https://github.com/pydantic/pydantic-ai/blob/main/docs/message-history.md#sharing-messages-between-agents)).

### AutoGen group chat: a valid counterexample for another topology

AutoGen's `RoundRobinGroupChat` deliberately has all agents share the same
context; each response is broadcast to every participant. `SelectorGroupChat`
similarly selects the next speaker from the shared conversation context and
broadcasts the response to all participants
([AutoGen teams](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/teams.html),
[AutoGen selector group chat](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/selector-group-chat.html)).

This proves that C is appropriate for a real group-chat topology. It does not
make C appropriate for a manager/worker research graph. Shared chat increases
cross-agent visibility by design, whereas this project explicitly wants
bounded delegation, deterministic fan-in, Specialist-specific Skills/Tools,
and no internal transcript pollution.

## Assessment of A–D

### A — workable, but redundant and weaker than E'

A matches Open Deep Research's final-writer input closely and preserves the
full conversational nuance for both the decision maker and final writer while
keeping workers isolated.

However, this repository already resolves history into `standalone_query`
before downstream work. Under A, the same Conversation history would be
processed once by Query Understanding and then exposed again to Coordinator and
Synthesis. This repeats tokens and attention context. More importantly, old
assistant prose reaches Synthesis even though it is not current-Run Evidence.
The marker gates reject invalid marker identities but, by design, do not prove
that every factual sentence is cited. A prompt saying "history is not Evidence"
cannot enforce the same boundary as not supplying the old prose.

### B — underspecified version of E'

B is safe for Specialists, but it can make Synthesis misunderstand a follow-up
such as "compare it to its benchmark" unless Synthesis receives either a
resolved current query/research brief or enough explicit Task/result context to
remove the ambiguity. The existing `standalone_query` supplies exactly this
missing piece, so B should be replaced by explicit E' rather than selected as
written.

### C — reject for this topology

C copies handoff/group-chat semantics into bounded workers. It needlessly
exposes irrelevant earlier answers, increases prompt size, and risks mixing
user-visible history with actor-local execution context. None of the reviewed
manager/subagent examples requires it.

### D — reject for a persistent Conversation product

D is operationally simple but contradicts the product's stated ability to
continue a Conversation. It moves reference resolution onto the user and makes
checkpointed Conversation messages mostly useless to model behavior.

### E' — recommend for the POC

E' gives every actor the smallest semantically appropriate input and reuses the
existing v2 request-understanding contract:

1. Query Understanding receives the selected complete Conversation pairs,
   current user query, and compact Intent Catalog descriptions.
2. Its non-empty `standalone_query` and validated selected Business Intent
   become the canonical current-Run intent input to Coordinator. Deterministic
   code resolves the selected Intent to a trusted, non-expanding Research Scope.
3. Coordinator delegates self-contained Task objectives; Specialists receive
   only those objectives and selected prior results.
4. Synthesis receives `standalone_query`, selected Business Intent, accepted
   current-Run Specialist Results, eligible Evidence, and valid Calculation
   Artifacts—not old assistant answers.

This is not deterministic proof that the LLM preserved every nuance while
rewriting the query. Neither A nor Open Deep Research can provide that semantic
guarantee. `V2ResolvedQuery` structurally guarantees a non-empty
`standalone_query`; whether it is actually self-contained remains a prompt and
evaluation property. The current default builder instruction is intentionally
thin, so Agent-mode adoption should define the follow-up-resolution instruction
explicitly. The appropriate POC control is an end-to-end follow-up test with a
referent that exists only in prior Conversation history, asserting the expected
standalone query, dispatch decision, and final report. If those evaluations
later show material intent loss, passing selected user-only history or a richer
typed request brief to Synthesis can be reconsidered with evidence.

## Recommended POC invariants

1. Persist only logical user and final publishable assistant Messages in
   `conversation_messages`.
2. Select only complete prior user/final-assistant pairs within a deterministic
   context budget.
3. Project those pairs only to the Query Understanding actor; never hand any
   downstream actor a raw prior PydanticAI run history.
4. Validate the selected Business Intent against the trusted Tenant catalog,
   resolve its non-expanding Research Scope, and pass the accepted non-empty
   `standalone_query` plus selected Intent to Coordinator and Synthesis.
5. Do not project Conversation history to a Specialist. Build its model input
   from the current Task objective and the accepted results selected by
   `context_task_ids`.
6. Keep each Specialist's Tool transcript inside that Specialist invocation.
   Return only the accepted `SpecialistResult` across the actor boundary.
7. Treat runtime dependencies and Graph state as code-visible by default and
   model-visible only through an explicit projection.
8. Only current-Run eligible Evidence and Calculation Artifact IDs can pass
   publication gates. Prior assistant prose is unavailable to downstream
   research actors, not merely labeled as non-Evidence.
9. Test a second-request follow-up whose referent appears only in the prior
   complete Conversation pair. Assert that Query Understanding receives the
   pair; Coordinator and Synthesis receive the resolved standalone query and
   selected Intent; and Specialist test doubles receive no Conversation
   messages.
