# LangGraph Agent Patterns POC

Status: ready-for-agent

## Problem Statement

The Linear LangGraph baseline cannot express multi-source concurrency, multi-hop research, map-reduce, or Specialist Agent delegation. Professional users need these patterns without allowing an LLM to control permissions, calculations, provenance, or unbounded execution.

## Solution

Add `agent` mode to the compatible `/v2/query/stream` endpoint. Use a fixed LangGraph safety skeleton around a validated dynamic Task DAG, PydanticAI actors, registered Tools, Agent Skills, deterministic calculations, and publication gates.

## User Stories

1. As a researcher, I want market data, reports, and news fetched concurrently, so that one Research Report combines multiple sources.
2. As an analyst, I want later Tasks to depend on earlier results, so that the platform can perform bounded multi-hop research.
3. As an analyst, I want time ranges and comparisons executed as map-reduce, so that parallel results merge deterministically.
4. As a user, I want Specialist Agents for bounded domain work, so that complex research remains modular and auditable.
5. As a financial user, I want numerical outputs produced by registered functions, so that the LLM cannot invent calculations.
6. As a compliance reviewer, I want every publishable claim linked to eligible Evidence or a Calculation Artifact, so that the report is traceable.
7. As a Tenant administrator, I want models, Skills, Specialists, Tools, freshness, and budgets controlled by policy, so that agents cannot expand their own authority.

## Implementation Decisions

- Depend on the Linear Core's v2 SSE, Run persistence, Conversation identity, checkpoints, recovery, Tenant isolation, and OpenTelemetry seams. Keep `linear` as the initial Tenant default; `mode: agent` is optional and additive.
- Use a static graph: input/research policy → understand and select Skills → activate approved Skills → plan/replan → compile plan → schedule ready frontier → execute Tasks → assess progress → prepare Evidence → synthesize typed report → output gates → finalize.
- The LLM returns typed plans only. Deterministic code validates acyclicity, dependencies, required/optional inputs, allowed Tools/Skills, schemas, budgets, retries, and deadlines before execution.
- Retrieval, reranking, reduction, calculation, and Specialist work are Task kinds behind one `TaskSpec → TaskOutcome` execution seam. Map-reduce and multi-hop are dependency shapes, not separate runtimes.
- Schedule each ready frontier with LangGraph fan-out/fan-in and a barrier before progress assessment. Permit at most 32 Tasks, eight-way fan-out, three replans, two eligible retries, a 60-second Task timeout, and a 10-minute Run budget.
- Keep Run, Task, and Specialist state separate. Branches return immutable outcomes; reducers merge stable-ID maps associatively and idempotently. Same ID with different content is a conflict.
- PydanticAI builds role-configured actors with model abstraction, activated Skill instructions, approved tool bindings, structured outputs, and usage. Planner actors do not execute business Tools.
- Specialist Agents are one-level, per-invocation subgraphs. They may use restricted read-only PydanticAI Tools, but bindings delegate to the platform Tool Executor for typed outcomes, Evidence, audit, and SSE events.
- Support Agent Skills discovery, activation, and references. Models propose Skills; deterministic Tenant policy approves and pins name, version, content hash, and Tool IDs. Skill scripts and assets are not executed.
- Provide mock registered Tools for instrument search, quotes, price series, reports, news, fund holdings, and sector membership. Preserve production-shaped provenance, time, unit, currency, status, retry, and Artifact contracts.
- Provide simple registered calculations for returns, volatility, Sharpe ratio, drawdown, support levels, Entry Zones, time-series aggregation, and period comparison. The LLM selects only allowed versioned methods and never generates executable formulas.
- Enforce Evidence eligibility, freshness, conflicts, required/optional coverage, calculation prerequisites, Calculation Artifact validation, and claim/citation integrity in code. Groundedness remains advisory in this POC.
- Synthesis produces a typed draft whose claims explicitly reference Citation and Calculation Artifact IDs. Do not stream answer text until deterministic publication gates pass; render the verified draft to Markdown afterward.
- Stream planning, Task, Tool, gate, warning, citation, and final progress through existing SSE event types. Do not expose chain-of-thought.
- Use the Tenant-registered Specialist catalog with initial mock Specialists for market data research, document/news research, and cross-dimension analysis. Calculation is a Task, not a Specialist.

## Testing Decisions

- Use `/v2/query/stream` in `agent` mode as the primary seam with fake PydanticAI models, deterministic mock Tools, and persisted Run state.
- Cover four successful external scenarios: multi-source fan-out/fan-in; fund → Entry Zone/support → holdings-sector multi-hop; time-series map-reduce; and cross-period market comparison.
- Cover missing, stale, conflicting, cross-Tenant, malformed, duplicated, and failed Tool/Evidence/calculation outcomes. Unsupported numerical claims must never be published.
- Verify bounded replanning, Task budgets, one-level Specialist permissions, approved Skill activation, deterministic reducer behavior, replay idempotency, and additive SSE compatibility.
- Run the Agent scenarios within the shared 50-concurrent-stream harness using mock dependencies.
- Add focused unit/property tests for plan compilation, frontier selection, stable-ID reducers, Evidence gates, and registered calculation contracts.

## Out of Scope

- Real financial providers, production-grade financial methodology, arbitrary code execution, or an isolated sandbox.
- Skill scripts/assets, recursive Specialist creation, unconstrained ReAct loops, automatic trading, or order execution.
- Long-term Memory implementation, frontend changes, LangGraph Platform/Cloud, or removal of the legacy engine.

## Further Notes

- This spec starts after the `langgraph-orchestration-core` contracts and Linear graph are available, but Agent implementation may begin in parallel once those shared contracts stabilize.
- The root glossary and ADR-0001 through ADR-0004 are normative.
