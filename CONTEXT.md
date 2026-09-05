# Agent Research Platform

This context defines the language for a tenant-isolated platform that produces evidence-backed research and decision support through persistent conversations.

## Language

**Tenant**:
The isolation boundary for an organization's data, configuration, credentials, conversations, and runs within one deployment.
_Avoid_: App, workspace, account

**Subject**:
The authenticated security principal identity within a Tenant, supplied by the
trusted authentication boundary and used to scope checkpoint access; distinct
from the User, the person and domain actor.
_Avoid_: User, caller, client identity

**User**:
A person within a Tenant who owns or participates in Conversations and initiates Runs.
_Avoid_: Caller, operator

**Conversation**:
A user-visible exchange identified by an opaque UUID that can contain multiple
Runs and be continued through checkpointed short-term state. The current Core
does not persist a product Conversation registry or History.
_Avoid_: Session, thread, chat session

**Request**:
A logical user request identified by the stable client `request_id`. Its user
Message and final assistant Message share that identity within one Conversation.
In the current Linear Core these Messages are short-term checkpoint state, not
a product History record.

**Conversation Summary**:
A future versioned compression of earlier Conversation context, used only to
construct model input. It is not implemented in the Linear Core.
_Avoid_: Long-term memory, Evidence, source of truth

**Run**:
One execution initiated by a user request within a Conversation.
_Avoid_: Request, job, workflow instance

**Business Intent**:
The Tenant-defined classification of the business outcome requested by the
current Run. It is selected from the Intent Catalog and remains fixed for that
Run. It is not a Tool, Skill, permission, or execution plan.
_Avoid_: Tool choice, workflow, user utterance

**Intent Catalog**:
The Tenant- and domain-controlled registry of selectable Business Intents,
also known as the enterprise Intent Library. An Agent may see compact intent
descriptions for classification; only trusted platform code may use the
associated execution constraints.
_Avoid_: Skill catalog, Tool registry, prompt library

**Research Scope**:
The immutable boundary on Specialists, Tools, data sources, and search
constraints for one Run. It is derived from Tenant policy and the selected
Business Intent, and may only narrow—not expand—the Tenant's authority.
_Avoid_: Intent, query, authorization grant

**Task**:
A bounded unit of delegated business work accepted for execution within a Run.
The Agent Graph owns its stable identity; references to completed earlier Tasks
select the Specialist Results materialized as context for multi-hop work.
_Avoid_: Step, node, subtask

**Task Outcome**:
The terminal success-or-failure value collected for one Task. A successful
outcome contains its Specialist Result; a failed outcome contains no Result.
Usage, retry history, failure classification, and diagnostics are separate
Graph or telemetry state.
_Avoid_: Specialist Result, execution log, error report

**Coordinator Decision**:
A bounded structured choice by the Coordinator Agent to dispatch the next batch
of Specialist work or finish research. It is proposed by an Agent and must be
accepted by deterministic validation before execution.
_Avoid_: Plan, workflow, Tool call

**Dispatch Batch**:
One accepted set of mutually independent Tasks that the Agent Graph may execute
concurrently. A Task may select context only from successful Tasks completed
before the batch.
_Avoid_: Plan, DAG, queue

**Coordination Round**:
One accepted Coordinator Decision together with its stable revision number and,
when dispatching, its Dispatch Batch. Completed rounds form the auditable record
of how research changed in response to results.
_Avoid_: Plan Revision, retry, graph superstep

**Evidence**:
A source-backed data item used to support a factual claim or calculation.
_Avoid_: Context, reference material, raw result

**Citation**:
A precise link from a factual claim or numerical result to the Evidence that supports it.
_Avoid_: Source list, bibliography entry

**Calculation Artifact**:
A reproducible numerical result together with its declared inputs, method,
precision, units, and execution record. Its complete record is internal; a
Research Report may render only the method, unit, relevant period or as-of time,
and material assumptions needed to interpret the value. It is derived support,
not source Evidence.
_Avoid_: LLM calculation, estimate

**Entry Zone**:
A method-derived price range used for research and scenario analysis, with its method, inputs, assumptions, and as-of time stated explicitly.
_Avoid_: Recommended buy point, trading instruction

**Research Report**:
A decision-support output that may analyze financial instruments but does not
execute trades. Its canonical prose may contain inline references whose
eligibility and identity are validated before publication; citation coverage
and semantic support remain advisory judgments.
_Avoid_: Trading signal, trade instruction

**Incomplete Research Report**:
A valid Research Report produced from accepted results when expected data is
unavailable, an accepted Specialist Task produces no result, or structural
execution limits stop further research. Code inserts a visible disclosure of
the observed limitation before publication. For the POC this status is
conservative and monotonic: later research does not erase an earlier accepted
limitation, and the absence of one is not proof of global completeness. It is
not an unvalidated partial draft.
_Avoid_: Partial output, failed report, draft

## Agent Operations

**Agent**:
A policy-bounded LLM actor defined by a model profile, instructions, activated
Skills, and approved Tools. It works autonomously on a delegated goal within an
orchestrated Run but is not itself the workflow or a model instance.
_Avoid_: LLM instance, workflow, bot

**Agent Graph**:
The LangGraph runtime that deterministically validates coordination decisions,
resolves the trusted Research Scope, dispatches work, collects outcomes,
enforces execution limits, and routes a Run to synthesis or termination. It is
not an Agent.
_Avoid_: Orchestrator, Coordinator Agent, workflow agent

**Query Understanding Agent**:
The Agent that uses the current query, bounded Conversation context, and
compact Intent Catalog descriptions to produce a standalone query and select
one Business Intent. It does not answer the user, retrieve data, choose Tools,
or grant authority.
_Avoid_: Coordinator Agent, Intent router, answer Agent

**Coordinator Agent**:
The Agent that decides the next bounded batch of Specialist work or ends further
coordination. Its `Finish` decision is not a deterministic claim that research
is globally complete. It does not execute business Tools, schedule work, enforce
technical limits, or write the final domain output.
_Avoid_: Orchestrator, Scheduler, Synthesis Agent

**Synthesis Agent**:
The Agent that turns accepted Specialist Results and eligible Evidence into the
final domain output after research coordination has finished. It does not plan
or dispatch Specialist work, consume technical Run state, or decide a
deterministic global research-completeness status. Its input is limited to the
standalone query, selected Business Intent, accepted bounded Specialist Results,
their eligible Evidence excerpts, and a bounded catalog of eligible Calculation
Artifacts used through code-rendered placeholders. It does not receive canonical
calculated values to retype. If there is no eligible Evidence, the Agent Graph
does not invoke it.
_Avoid_: Coordinator, Planner, report renderer

**Specialist Agent**:
A bounded Agent that performs one domain-specific responsibility delegated by
the Coordinator Agent through the Agent Graph.
_Avoid_: Sub-agent, worker agent

**Specialist Descriptor**:
A compact, prompt-visible statement of a Specialist Agent's identity and
capabilities that a Coordinator Agent uses for delegation. It is not a function
schema or the Specialist's Skill catalog.
_Avoid_: Specialist prompt, Agent schema, Skill list

**Specialist Result**:
The bounded accepted result of a Specialist Agent's delegated work. It contains
the Agent's research summary and Evidence references plus code-owned Data Gaps
derived from expected unavailable Tool outcomes in the accepted attempt. It
excludes the Specialist's internal messages and Tool-call history.
_Avoid_: Task Outcome, Agent transcript, raw Tool result

**Data Gap**:
A bounded, code-owned record of requested business data that an expected read,
fetch, or Calculation outcome could not establish. It carries minimal trusted
source and observation provenance, may appear in a successful Specialist Result,
and contains no raw exception, provider payload, Tool arguments, credentials,
retry history, or stack trace. It records an observed limitation rather than a
claim about global research completeness.
_Avoid_: Error, exception, Task failure

**Tool**:
A registered executable capability with an explicit input contract and
permission policy. A Specialist Agent may invoke a Tool only when Tenant
policy, the current Research Scope, and the Specialist's own allowlist all
permit it; a Skill cannot grant Tool authority. The platform's Tool scope is
limited to reading or fetching data and deterministic Calculation; Tools do not
mutate external business state, transfer assets, execute orders, or consume a
business budget or quota that requires atomic reservation. An expected inability
to provide requested data is returned as a typed, model-visible outcome so the
Specialist may continue; authorization, invariant, and programmer failures
remain exceptions.
_Avoid_: Function, plugin, action

**Skill**:
A package of instructions and reference material that an Agent may activate on
demand. It guides use of already-authorized capabilities and cannot grant a Tool.
_Avoid_: Tool, prompt template, plugin

**Eligible Skill**:
A Skill that trusted Tenant policy and either shared or Specialist-scoped
registration permit one Specialist Agent to discover and activate. Eligibility
does not mean its full instructions have been loaded and does not grant Tools.
_Avoid_: Available Skill, activated Skill

**Activated Skill**:
An Eligible Skill whose full instructions have been loaded for one Specialist
Agent invocation.
_Avoid_: Cached Skill, registered Skill

**Calculation Executor**:
The component that evaluates approved deterministic numerical operations and produces Calculation Artifacts.
_Avoid_: Calculator agent, code agent, sandbox
