# LangGraph Agent Patterns POC

Status: ready-for-agent

## Problem Statement

The Linear LangGraph baseline cannot express multi-source concurrency, multi-hop research, map-reduce, or Specialist Agent delegation. Professional users need these patterns without allowing an LLM to control permissions, calculations, provenance, or unbounded execution.

## Solution

Allow the compatible `/v2/query/stream` endpoint to run either the existing
`linear` Graph or the new `agent` Graph according to the authenticated Tenant's
server-side mode configuration. The client cannot select or override the mode.
Use a fixed LangGraph safety skeleton around a validated dynamic Task DAG,
PydanticAI actors, registered Tools, Agent Skills, deterministic calculations,
and publication gates. `agent` is one runtime mode, not a collection of
Tenant-selected pattern implementations: each Turn derives the execution shape
required by its query, and one plan may combine parallel fan-out, multi-hop
dependencies, map-reduce, and Specialist delegation.

## User Stories

1. As a researcher, I want market data, reports, and news fetched concurrently, so that one Research Report combines multiple sources.
2. As an analyst, I want later Tasks to depend on earlier results, so that the platform can perform bounded multi-hop research.
3. As an analyst, I want time ranges and comparisons executed as map-reduce, so that parallel results merge deterministically.
4. As a user, I want Specialist Agents for bounded domain work, so that complex research remains modular and auditable.
5. As a financial user, I want numerical outputs produced by registered functions, so that the LLM cannot invent calculations.
6. As a compliance reviewer, I want every publishable claim linked to eligible Evidence or a Calculation Artifact, so that the report is traceable.
7. As a Tenant administrator, I want models, Skills, Specialists, Tools, freshness, and budgets controlled by policy, so that agents cannot expand their own authority.

## Implementation Decisions

- Depend on the Linear Core's request-owned v2 SSE, Conversation/Turn identity,
  official shared PostgreSQL checkpoints and Tenant isolation. Keep
  `linear` as the initial Tenant default and enable `agent` only through trusted
  Tenant configuration.
- A Run remains the domain name for one execution, but it has no application-owned
  persistence model. LangGraph State and its official checkpoint are the durable
  execution authority; do not add a `runs` table, Run repository, duplicate
  checkpoint pointer, transport Event journal, or Redis recovery state.
- Resolve the Tenant's configured mode before starting a Turn. One Tenant has
  exactly one active mode, so a Conversation never mixes Linear and Agent Turns.
  Runtime mode is fixed for the current deployment configuration; config reload
  must reject a mode change.
- Within an Agent-mode Conversation, plan every Turn independently from that
  Turn's query and authorized capabilities. Consecutive Turns may use different
  execution shapes; do not store a Tenant-level or Conversation-level active
  Agent pattern.
- Share Query authorization and request-owned streaming mechanics, and inject
  the same official checkpointer. Keep each Graph's builder and execution policy
  inside that Graph's own module rather than adding Agent branches to the Linear
  Graph.
- Use one static Agent Graph: input/research policy → understand and select Skills → activate approved Skills → plan/replan → compile plan → schedule ready frontier → execute Tasks → assess progress → prepare Evidence → synthesize typed report → output gates → finalize. Query-specific patterns are validated plan topology inside this Graph, not separately compiled top-level Graphs.
- The LLM returns typed plans only. Deterministic code validates acyclicity, dependencies, required/optional inputs, allowed Tools/Skills, schemas, budgets, retries, and deadlines before execution.
- Every accepted `ValidatedPlan` has an immutable revision and content hash.
  Normal progress assessment may create a new validated revision. A changed
  `TaskSpec` must receive a new Task ID. A terminal outcome may be reused only
  when both its Task ID and spec hash match the active plan.
- Checkpoint lightweight Task control outcomes separately from request-local
  Evidence and raw Tool payloads. Expansion whose cardinality is learned from
  an outcome must pass through normal assessment and a newly validated plan
  revision.
- Retrieval, reranking, reduction, calculation, and Specialist work are Task kinds behind one `TaskSpec → TaskOutcome` execution seam. Map-reduce and multi-hop are dependency shapes, not separate runtimes.
- Schedule each ready frontier with LangGraph fan-out/fan-in and a barrier before progress assessment. Permit at most 32 Tasks, eight-way fan-out, three replans, two eligible retries, a 60-second Task timeout, and a 10-minute Run budget. Anchor the Run budget to the Turn's creation time and checkpoint its fixed deadline; retry never resets it.
- Keep Run, Task, and Specialist state separate inside LangGraph State. Branches return immutable outcomes; reducers merge stable-ID maps associatively and idempotently. Same ID with different content is a conflict.
- PydanticAI builds role-configured actors with model abstraction, activated Skill instructions, approved tool bindings, structured outputs, and usage. Planner actors do not execute business Tools.
- Specialist Agents are one-level, per-invocation subgraphs. They may use restricted read-only PydanticAI Tools, but bindings delegate to the platform Tool Executor for typed outcomes, Evidence, audit, and SSE events.
- Support Agent Skills discovery, activation, and references. Models propose Skills; deterministic Tenant policy approves and pins name, version, content hash, and Tool IDs. Skill scripts and assets are not executed.
- Provide mock registered Tools for instrument search, quotes, price series, reports, news, fund holdings, and sector membership. Preserve production-shaped provenance, time, unit, currency, status, retry, Evidence, and Calculation Artifact contracts.
- Provide simple registered calculations for returns, volatility, Sharpe ratio, drawdown, support levels, Entry Zones, time-series aggregation, and period comparison. The LLM selects only allowed versioned methods and never generates executable formulas.
- Enforce Evidence eligibility, freshness, conflicts, required/optional coverage, calculation prerequisites, Calculation Artifact validation, and claim/citation integrity in code. Groundedness remains advisory in this POC.
- Synthesis produces a typed draft whose claims explicitly reference Citation and Calculation Artifact IDs. Do not stream answer text until deterministic publication gates pass; render the verified draft to Markdown afterward.
- Stream planning, Task, Tool, gate, warning, citation, and final progress through existing SSE event types. Do not expose chain-of-thought.
- Use the Tenant-registered Specialist catalog with initial mock Specialists for market data research, document/news research, and cross-dimension analysis. Calculation is a Task, not a Specialist.
- Keep retrieved chunks, raw Tool payloads, and other large Evidence values in
  request-local untracked state. Checkpoint plans, fixed deadlines, Task control
  state, and stable identifiers only.

## Testing Decisions

- Use `/v2/query/stream` with an Agent-configured Tenant as the primary seam,
  with fake PydanticAI models, deterministic mock Tools, and checkpointed
  LangGraph State.
- Cover four successful external scenarios: multi-source fan-out/fan-in; fund → Entry Zone/support → holdings-sector multi-hop; time-series map-reduce; and cross-period market comparison.
- Cover missing, stale, conflicting, cross-Tenant, malformed, duplicated, and failed Tool/Evidence/calculation outcomes. Unsupported numerical claims must never be published.
- Verify bounded replanning, Task budgets, one-level Specialist permissions, approved Skill activation, deterministic reducer behavior, retry idempotency, and additive SSE compatibility.
- Verify that separate Linear-configured and Agent-configured Tenants use the
  same query route and cannot override mode through request input.
- Verify consecutive Turns in one Agent-mode Conversation can execute different
  shapes—for example, fan-out/fan-in followed by a combined multi-hop plan.
- Run the Agent scenarios within the shared 50-concurrent-stream harness using mock dependencies.
- Add focused unit/property tests for plan compilation, frontier selection, stable-ID reducers, Evidence gates, and registered calculation contracts.

## Out of Scope

- Real financial providers, production-grade financial methodology, arbitrary code execution, or an isolated sandbox.
- Skill scripts/assets, recursive Specialist creation, unconstrained ReAct loops, automatic trading, or order execution.
- Long-term Memory implementation, frontend changes, LangGraph Platform/Cloud, or removal of the legacy engine.
- Checkpoint Resume, execution recovery, and human-in-the-loop interrupts.

## Further Notes

- This spec starts after the `langgraph-orchestration-core` contracts and Linear graph are available, but Agent implementation may begin in parallel once those shared contracts stabilize.
- The root glossary and ADR-0001 through ADR-0004 are normative.
