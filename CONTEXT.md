# Agent Research Platform

This context defines the language for a tenant-isolated platform that produces evidence-backed research and decision support through persistent conversations.

## Language

**Tenant**:
The isolation boundary for an organization's data, configuration, credentials, conversations, and runs within one deployment.
_Avoid_: App, workspace, account

**Subject**:
The authenticated security principal identity within a Tenant, supplied by the trusted authentication boundary and used to authorize Conversation access; distinct from the User, the person and domain actor.
_Avoid_: User, caller, client identity

**User**:
A person within a Tenant who owns or participates in Conversations and initiates Runs.
_Avoid_: Caller, operator

**Conversation**:
A persistent, user-visible exchange that can contain multiple Runs and be continued over time.
_Avoid_: Session, thread, chat session

**Turn**:
One user Message and its resulting assistant Message or recovery identity within a Conversation; distinct from a Run execution.

**Conversation Summary**:
A versioned compression of earlier Messages within one Conversation, used only to construct model context while the original Messages remain authoritative.
_Avoid_: Long-term memory, Evidence, source of truth

**Run**:
One execution initiated by a user request within a Conversation.
_Avoid_: Request, job, workflow instance

**Task**:
A bounded unit of work within a Run with explicit inputs, outputs, dependencies, and completion status.
_Avoid_: Step, node, subtask

**Evidence**:
A source-backed data item used to support a factual claim or calculation.
_Avoid_: Context, reference material, raw result

**Citation**:
A precise link from a factual claim or numerical result to the Evidence that supports it.
_Avoid_: Source list, bibliography entry

**Calculation Artifact**:
A reproducible numerical result together with its declared inputs, method, precision, units, and execution record.
_Avoid_: LLM calculation, estimate

**Entry Zone**:
A method-derived price range used for research and scenario analysis, with its method, inputs, assumptions, and as-of time stated explicitly.
_Avoid_: Recommended buy point, trading instruction

**Research Report**:
An evidence-backed decision-support output that may analyze financial instruments but does not execute trades.
_Avoid_: Trading signal, trade instruction

## Agent Operations

**Orchestrator**:
The coordinator responsible for planning, scheduling, budgets, and aggregation within a Run.
_Avoid_: Master agent, manager agent

**Specialist Agent**:
A bounded agent that performs one domain-specific responsibility for an Orchestrator.
_Avoid_: Sub-agent, worker agent

**Tool**:
A registered executable capability with an explicit input contract and permission policy.
_Avoid_: Function, plugin, action

**Skill**:
An on-demand package of instructions and reference material that guides agent behaviour.
_Avoid_: Tool, prompt template, plugin

**Calculation Executor**:
The component that evaluates approved deterministic numerical operations and produces Calculation Artifacts.
_Avoid_: Calculator agent, code agent, sandbox
