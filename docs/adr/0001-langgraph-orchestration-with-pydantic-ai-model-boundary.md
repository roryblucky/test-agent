# Use LangGraph for orchestration and PydanticAI for LLM actors

Status: partially superseded by ADR-0006

ADR-0006 supersedes this record's Agent-mode requirements for Planner-authored
complete Task DAGs, top-level Calculation Tasks, and related scheduling. The
LangGraph/PydanticAI ownership boundary below remains accepted.

All new orchestration is implemented with self-hosted LangGraph behind FastAPI. LangGraph owns graph state, routing, fan-out/fan-in, map-reduce, replanning, Specialist composition, short-term thread state, and checkpoints, while preserving future seams for long-term memory, human interrupts, and recovery.

Superseded Agent-mode detail, retained only as decision history:
PydanticAI constructs the LLM-facing actors with model-provider abstraction, instructions, activated Skills, structured outputs, usage tracking, and approved tool bindings. Planner actors return typed Task DAGs instead of executing business tools. LangGraph schedules top-level Retrieval, Rerank, Reduction, Calculation, and Specialist Tasks; a Specialist may use a restricted read-only PydanticAI toolset internally, but every binding delegates to the platform Tool Executor for typed outcomes, Evidence, audit, and events. Deterministic Calculation remains an explicit LangGraph Task rather than a hidden free-form tool call.

The accepted actor boundary is version-independent: LangGraph and application
adapters own coordination, technical retry, and repair; actor configuration
therefore makes Tool/output retry, end strategy, Tool execution ordering, MCP
execution location, and provider selection explicit rather than inheriting
library defaults. PydanticAI dependency migration and version-specific
compatibility are owned by a separate migration spec. POC Tools do not use
PydanticAI model-adaptation exceptions for expected failures.
Registered Tool bindings return expected read/fetch/Calculation unavailability
as bounded typed values, allowing the same Specialist run to continue multi-hop
or collect partial internal fan-out results. Fatal authorization, invariant,
and programmer failures remain exceptions.
