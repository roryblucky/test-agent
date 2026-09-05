# PydanticAI V2 Migration

Status: ready-for-agent

Last researched: 2026-09-05

## Problem Statement

The repository declares `pydantic-ai>=1.93.0`, while `uv.lock` currently resolves
`pydantic-ai`, `pydantic-ai-slim`, `pydantic-evals`, and `pydantic-graph` to
`1.93.0`. A fresh dependency upgrade may therefore cross the major-version
boundary before the application has migrated its APIs and reviewed V2's silent
behavior changes.

The application depends on PydanticAI at several runtime boundaries: Agent
construction, structured and streamed output, usage accounting, message-history
serialization, history processing, custom capabilities, function toolsets, MCP,
and Azure/Google model providers. The migration must move those boundaries to
the latest stable PydanticAI V2 without redesigning the LangGraph Agent Patterns
or changing the public API.

## Goal

Upgrade the completed application from PydanticAI `1.93.0` to the latest stable
V2 release through the official latest-V1 bridge, preserving application-owned
behavior and proving compatibility with deterministic tests before any
real-provider canary.

At the research date:

- the latest stable V2 release is `2.39.0`;
- the latest V1 bridge release is `1.107.5`;
- PydanticAI's official upgrade path is latest V1, resolve every deprecation
  warning, then upgrade to V2 and review changes that cannot produce warnings.

The implementer must re-check the official release page when implementation
starts. If a newer stable V2 exists, update the exact target in this spec, review
every intervening release note, and use that version. Do not silently implement
against a newer version than the reviewed spec.

`1.107.0` was the V1 code line from which the final V2 beta was forked, but it
was not the last V1 maintenance release. PyPI and the signed GitHub release show
that `1.107.5` followed it; the bridge uses the latest available V1 patch so all
backported fixes and deprecation guidance are present.

## Preconditions and Sequence

- Implement this migration only after the LangGraph Agent Patterns work is
  complete and green on the current PydanticAI V1 baseline. The migration then
  covers the final merged call-site inventory, including its Coordinator,
  Specialist, Synthesis, Tool-return, retry, and Pydantic Evals contracts.
- Keep the latest-V1 and V2 transitions as two verifiable changes on the
  migration branch. The latest-V1 state is a migration checkpoint, not a new
  production architecture or a long-lived compatibility target.
- Do not create a V1/V2 compatibility abstraction, conditional imports, or
  version checks. Once the V2 phase starts, production code and test doubles use
  only V2 APIs.
- Dependency, source, tests, and `uv.lock` move together. Do not merge or deploy
  a dependency-only V2 bump.
- This spec does not create implementation tickets. Tickets are planned only
  after this spec is approved.

## Compatibility Contract

The migration preserves these application-owned contracts:

- `/v2/query/stream` request fields, SSE event names and framing, token order,
  `done.answer`, citations, clarification behavior, and checkpoint-before-
  publication ordering;
- the legacy endpoint contracts while those endpoints remain in the repository;
- Tenant model selection, Azure Chat Completions behavior, Google Cloud/Vertex
  behavior, model settings, structured output schemas, prompts, Tool allowlists,
  Skill discovery/activation, and MCP tool prefixes;
- the LangGraph/PydanticAI ownership boundary and all accepted Agent Patterns
  limits, retry ownership, partial-result, Evidence, Calculation, and publication
  rules;
- public token-usage field names such as `request_tokens` and
  `response_tokens`, even though PydanticAI's internal V2 names are
  `input_tokens` and `output_tokens`.

The migration may correct history processing where the existing implementation
does not meet its stated contract: Tool call/return pairs must remain complete,
and filtering a `ModelResponse` must preserve its other fields.

## Version and Dependency Strategy

### Phase 0: Freeze the baseline

- Record the current `1.93.0` test result before changing dependencies.
- Generate and commit one representative V1 message-history fixture with
  `ModelMessagesTypeAdapter`. It must cover the message shapes the application
  persists, including user and assistant text plus one function Tool call/return
  pair. Include Thinking or usage data when those shapes are persisted by the
  completed Agent Patterns implementation.
- Re-run the PydanticAI call-site inventory after Agent Patterns lands. The
  inventory must include imports, Agent constructors, stream/result accessors,
  capabilities, toolsets, MCP, providers, message serialization, and
  `pydantic_evals`.

### Phase 1: Latest V1 bridge

- Replace the open lower bound with the exact bridge pin
  `pydantic-ai==1.107.5` and regenerate `uv.lock`.
- If application or test code imports `pydantic_evals` directly, declare it as
  a direct development dependency at the matching exact release instead of
  relying on a transitive extra.
- Run application startup, Agent warm-up/construction, and the full deterministic
  suite with `PydanticAIDeprecationWarning` treated as an error.
- Apply every deprecation-guided migration that V1 supports, including
  capabilities, result/usage accessors, provider names, retry arguments, and
  Pydantic Evals call signatures.
- The bridge phase is complete only when no PydanticAI deprecation warning is
  ignored or filtered.

### Phase 2: Stable V2

- Re-check the current stable release. With the research snapshot, pin
  `pydantic-ai==2.39.0`; never restore an open `>=` major-version range.
- Regenerate `uv.lock` from the declared dependency and verify that the OpenAI,
  Google, MCP, Evals, and retry functionality used by this repository is
  installed. Do not add unused provider extras.
- Treat the transitive OpenAI Python SDK change as part of this migration. At
  the research snapshot, V2 `2.39.0` requires OpenAI SDK `>=3.8.0`, while the
  current lock contains `2.36.0`; record and verify the final resolved 3.x
  version rather than treating it as an incidental lockfile change.
- Because production code will create a PydanticAI-only `httpx2` client,
  declare `httpx2` as a direct runtime dependency using the compatibility range
  required by the reviewed PydanticAI release and let `uv.lock` fix its exact
  version. Do not rely on an undeclared transitive import.
- Remove bridge-only compatibility code and deprecated names. The final source
  tree has one V2 code path.
- Run all deterministic gates before Azure or Google network calls. Real-provider
  canaries validate the final artifact; they are not a way to discover ordinary
  import, schema, or control-flow failures.

## Repository Migration Matrix

| Boundary | Current repository use | Required V2 result |
| --- | --- | --- |
| Dependency resolution | `pydantic-ai>=1.93.0`, lock at `1.93.0` | Exact reviewed V2 pin and regenerated lock; no accidental major upgrade |
| OpenAI SDK closure | Direct lower bound `openai>=2.20.0`, lock at `2.36.0` | Resolve the PydanticAI-required OpenAI SDK 3.x version and validate Azure behavior explicitly |
| Dependency-free Agents | `Agent[None, ...]` and ModelRegistry overloads return `None` deps | Use `Agent[object, ...]` and the V2 `object` default where no dependency object is required |
| History processing | `Agent(history_processors=[...])` in Agent factories | Register ordered `ProcessHistory(...)` capabilities |
| History transformation | `trim_history` and `filter_thinking` rebuild or slice messages | Preserve correct `ModelResponse(ToolCallPart)` → `ModelRequest(ToolReturnPart)` pairs and use replacement/copy semantics that retain all untouched fields |
| Stream usage | `stream.usage()` in handlers and citation extraction | Read the V2 `stream.usage` property; update fakes and stable usage adapters |
| Usage normalization | `model_usage_payload()` only accepts a callable `usage` | Read a V2 `RunUsage` value directly and emit the existing application-owned usage mapping |
| Stream/output API | Structured streams use stable methods; `LLMHandler` joins cumulative `stream_text()` snapshots as if they were deltas | Keep unchanged V2 APIs unchanged, but fix the snapshot/delta bug with suffix publication and last-snapshot finalization |
| Direct MCP clients | `MCPServerSSE`, `MCPServerStreamableHTTP`, `MCPServerStdio` | Use `MCPToolset`; preserve URL transport selection, stdio args/env, prefixing, lifecycle, and the route's existing effective retry/timeout behavior |
| Adaptive MCP capability | V1 `MCP(url=...)` native behavior in `AgentHandler` | Make provider-native intent explicit with `native=True`; do not accept V2's local-by-default flip accidentally |
| Function toolsets | Context-aware callables passed to `FunctionToolset` | Keep `FunctionToolset`; prove every `.tool()`/registered context-aware callable has `RunContext` first and use `tool_plain()` only for truly context-free functions |
| Skills capability | Custom `SkillsCapability(AbstractCapability)` | Keep the custom capability; prove instructions, Tool registration, per-run state, and capability IDs still behave under V2 |
| Google model provider | `GoogleProvider(project=..., http_client=...)` selects Vertex | Use `GoogleCloudProvider(project=..., http_client=...)` with the existing `GoogleModel` and settings |
| Azure model provider | Explicit `OpenAIChatModel` with `AzureProvider` | Keep Chat Completions explicitly; do not switch to the Responses API as part of this migration |
| Provider HTTP client | Shared legacy `httpx.AsyncClient` is passed into PydanticAI providers | Add one PydanticAI-only cached `httpx2.AsyncClient` seam; leave the existing client pool consumers unchanged and close both client families at shutdown |
| Agent termination | V1 default `end_strategy='early'` | Declare `early` explicitly for existing actors that relied on the V1 default; preserve any already-explicit Agent Patterns policy |
| Agent retries | V1 defaults and any merged `tool_retries`/`output_retries` | Preserve each actor's accepted policy using V2 `retries={'tools': ..., 'output': ...}`; do not invent a new global budget or hidden recovery loop |
| Captured messages/events | Agent Patterns tests may assert exact trajectories | Account for V2 interrupted request/response capture and dedicated output Tool events; do not discard interrupted state to keep an old assertion |
| Pydantic Evals | Added by the completed Agent Patterns work | Use keyword-only calls, required Dataset names, and V2 evaluator accessor methods where applicable |
| Serialized history | The legacy store reads with `ModelMessagesTypeAdapter` but writes with generic `pydantic_core.to_json()`; production may select `RedisSessionStore` | Write with `ModelMessagesTypeAdapter.dump_json()` and prove V1 fixture compatibility; LangGraph checkpoints remain application-owned |
| Pytest discovery | Root `test_stream_union.py` and `test_union2.py` execute experiments during import | Put execution behind `if __name__ == "__main__":` or move them outside test discovery, then prove root-level collection succeeds |

## Implementation Decisions

### Agent construction and defaults

- Keep `ModelRegistry` as the narrow construction seam. It may normalize common
  V2 defaults, but it must not become a generic compatibility wrapper or hide
  actor-specific retry policy.
- Convert dependency-free factory annotations and overloads from `None` to
  `object`. Agents with real dependencies such as `AgentDeps`, `RAGAgentDeps`,
  and `SharedState` keep those types.
- Register each existing history processor as its own `ProcessHistory`
  capability in the same order: trim first, Thinking filtering second. Merge
  these with existing capabilities rather than replacing Skill or MCP
  capabilities.
- Preserve the V1 termination behavior by explicitly setting
  `end_strategy='early'` on existing Agent constructors that have no stronger
  actor-owned setting. A deterministic trajectory test must return a valid
  structured output alongside a function Tool call and prove the Tool is not
  executed after the output is accepted.
- Do not globally change retry counts. Translate deprecated retry arguments to
  the V2 `retries` dictionary with the same effective values. In particular,
  retain the Agent Patterns spec's separation between adapter-owned repair,
  Specialist outer attempts, and disabled hidden Tool/output retries.

### History processing and persistence

- Update the history processors while migrating them to `ProcessHistory`:
  Tool calls live in `ModelResponse`; their Tool returns live in a following
  `ModelRequest`. Trimming must never leave either side orphaned, even when a
  complete pair slightly exceeds the requested message count.
- When removing `ThinkingPart`, create a replacement `ModelResponse` that changes
  only `parts`. Preserve model name, timestamp, usage, provider metadata,
  finish/state fields, run identity, and future fields supplied by the installed
  V2 type. Do not mutate existing history objects in place.
- Keep `new_messages()` as the per-run persistence boundary. Tests must prove
  that `ProcessHistory` does not cause prior messages to enter
  `new_messages()` or remove the current run's visible user/final-assistant
  messages.
- PydanticAI officially guarantees that history serialized through the V1
  `ModelMessagesTypeAdapter` deserializes under V2. Use that guarantee and the
  golden fixture; do not add a data converter, version column, or dual reader.
- Make the write boundary match the official serializer: replace generic
  `pydantic_core.to_json(messages)` in `BaseSessionStore.save()` with
  `ModelMessagesTypeAdapter.dump_json(messages)`. This applies equally to the
  in-memory and Redis implementations without changing either backend.
- The new LangGraph runtime stores application-owned Conversation Messages, not
  PydanticAI run history, so this migration adds no PostgreSQL/Alembic change.

### Results, streaming, and usage

- Replace every PydanticAI result/stream `usage()` call with the V2 `usage`
  property. Keep `new_messages()`, `get_output()`, and `stream_output()` as
  methods because those APIs remain methods.
- Make the PydanticAI boundary produce one internal mapping based on V2
  `RunUsage`: `requests`, `input_tokens`, `output_tokens`, `tool_calls`, and any
  already-supported detail fields. Callers do not inspect `__dict__` or test
  whether `usage` is callable.
- Preserve the public compatibility translation to `request_tokens` and
  `response_tokens`. Those response names are an application API decision, not
  a reason to retain V1 PydanticAI accessors internally.
- Preserve existing stream cancellation and close/drain behavior. This migration
  does not add SSE replay, background execution, or a Run-wide deadline.
- Correct the existing `LLMHandler` text path without changing its API:
  `stream_text()` keeps the default cumulative-snapshot mode so final messages
  remain available; publish only the suffix relative to the previous snapshot,
  and set the final answer to the last complete snapshot. A deterministic
  `h` → `he` → `hello` test must emit `h`, `e`, `llo` and finalize exactly
  `hello`.

### Providers

- Keep Azure on the explicitly constructed `OpenAIChatModel` and
  `OpenAIChatModelSettings`. The V2 bare `openai:` prefix now means Responses,
  but this repository does not use that shorthand and must not change API
  families during a dependency migration.
- Replace the Google Vertex construction with `GoogleCloudProvider` from
  `pydantic_ai.providers.google_cloud`. Preserve the configured project, shared
  model name, `GoogleModel`, and `GoogleModelSettings` including
  Thinking settings.
- PydanticAI V2 accepts the existing `httpx.AsyncClient` with a V3 deprecation
  warning. Avoid creating that known debt: add a narrowly named cached
  PydanticAI client getter to `HttpClientPool` using `httpx2.AsyncClient`, with
  the same timeout, connection-limit, proxy, reuse, and shutdown behavior as the
  existing getter. Only Azure and Google PydanticAI builders use it; retrievers,
  rankers, moderation, and other HTTP consumers stay on the existing client.
- Add construction tests for Azure and Google that make no network request and
  assert the selected model/provider class and relevant settings. Keep separate,
  opt-in Azure and Google canaries for authentication and real structured
  streaming after deterministic tests pass.
- Do not add a provider, change a model identifier, migrate Azure to its newer
  endpoint form, or tune generation parameters in this spec.

### MCP, Tools, and Skills

- Replace the legacy per-transport MCP classes in `app/core/mcp.py` with
  `MCPToolset`. A URL ending in `/sse` remains legacy SSE; other URLs remain
  Streamable HTTP. Preserve stdio command, args, and env through the supported
  FastMCP stdio transport, and apply the existing server name through the V2
  prefixed-toolset API.
- Preserve the legacy builder's effective timeout and retry behavior explicitly
  because V2's `MCPToolset` defaults differ. For the existing legacy builder,
  preserve the latest-V1 values: all transports use a 5-second initialization
  timeout and a 300-second read timeout; MCP Tool errors use
  `tool_error_behavior="retry"` and `max_retries=1`. Here `max_retries` is the
  number of model-visible `ModelRetry` attempts for that Tool call, not a
  transport reconnection retry. Tests must prove a completed server `ToolError`
  or protocol `McpError` raised during a Tool call gets one correction attempt
  and then fails the run; connection-establishment and other non-Tool/non-MCP
  exceptions propagate directly. None of these paths may be presented as a
  successful Tool result. These legacy
  compatibility values do not apply to Agent Patterns actors; those actors keep
  their accepted zero-hidden-retry adapter boundaries.
- In `AgentHandler`, explicitly select native MCP behavior to preserve the V1
  `MCP(url=...)` meaning. Moving these servers to local execution may be a valid
  later security/observability decision, but it is a separate behavior change.
- Keep expected POC Tool unavailability as the Agent Patterns spec defines it.
  The migration does not start using `ModelRetry` or `ToolFailed` for those
  outcomes merely because V2 exposes richer failure APIs.
- Exercise `SkillsCapability` with a V2 Agent and deterministic model: an empty
  registry exposes no activation Tools, a populated registry exposes the two
  approved Tools, instructions are refreshed per run, and per-run activated
  state does not leak between runs.

### Tests and model isolation

- Ordinary unit and integration tests must set
  `pydantic_ai.models.ALLOW_MODEL_REQUESTS=False`. Explicit real-provider
  canaries opt out in their isolated process or test configuration.
- Use the official `TestModel`, `FunctionModel`, `Agent.override()`, and
  `capture_run_messages()` where they expose a PydanticAI contract more directly
  than a hand-written fake. Keep small application fakes where the test is about
  LangGraph or HTTP behavior rather than PydanticAI internals.
- Update stream fakes to expose `usage` as a property/value. Do not make test
  doubles accept both a method and a property, since that would hide incomplete
  migration.
- Make the repository's default test discovery safe. The two root union-output
  experiments must not execute at import time; guard or relocate them, then run
  root-level `pytest --collect-only` in addition to the actual `tests/` suite.
- A dependency migration is not accepted on import success alone. Tests must
  cover exact model-request trajectories, structured output, streaming usage,
  history processing, Tool registration, MCP construction, provider
  construction, and the Agent Patterns multi-hop/internal fan-out failure paths.

## Verification Gates

### Latest-V1 bridge gate

Run, at minimum:

```shell
uv lock --check
uv sync --frozen
uv run ruff check app tests alembic/versions/0018_drop_conversation_registry.py
uv run pyright --pythonpath .venv/bin/python
PYTHONPATH=. uv run pytest -W error::pydantic_ai.agent.PydanticAIDeprecationWarning tests
PYTHONPATH=. uv run pytest --collect-only
```

The PostgreSQL-backed tests use the disposable database contract documented in
`docs/testing.md`. Startup, Agent construction/warm-up, and representative run
paths must all execute under the warning-as-error setting.

### V2 deterministic gate

- The lock and installed metadata resolve the reviewed exact V2 version.
- The lock also resolves an OpenAI SDK version satisfying the reviewed
  PydanticAI 2.39.0 requirement (`>=3.8.0` at the research snapshot) and the
  declared compatible `httpx2` dependency.
- Ruff, strict Pyright, and the complete deterministic pytest suite pass.
- A removed-symbol scan has no production references to the V1 MCP server
  classes, `history_processors=`, method-style PydanticAI `usage()`, V1 provider
  names, or deprecated retry arguments. Public response fields and comments are
  excluded only when they intentionally describe the stable application API.
- The V1 golden history fixture deserializes through the installed V2
  `ModelMessagesTypeAdapter` and preserves semantically equivalent messages,
  Tool correlation IDs, text, and usage.
- History tests prove Tool call/return pairing, field-preserving Thinking
  filtering, processor order, and the `new_messages()` boundary.
- Session-store tests prove `ModelMessagesTypeAdapter.dump_json()` is the only
  message-history write serializer and both in-memory and Redis-backed stores
  round-trip the same fixture.
- Provider tests prove Azure remains Chat Completions and GCP remains Google
  Cloud/Vertex with the shared HTTP-client seam.
- MCP tests prove SSE/Streamable HTTP/stdio selection, args/env, prefixes,
  Tool allowlists, and the intended retry/error policy without contacting an
  external server.
- PydanticAI-native contract tests prove structured output, usage extraction,
  explicit termination/retry behavior, Tool metadata isolation, and exact
  request counts.
- The cumulative text-stream counterexample emits suffixes only and preserves
  both the exact final answer and the current-run message history.
- Root-level pytest collection completes without executing experimental network
  or Agent runs.
- All LangGraph Linear and Agent Pattern tests pass unchanged at their public
  boundaries, including final checkpoint-before-publication and partial-result
  behavior.

### Real-provider canary gate

- After deterministic gates pass, run one bounded Azure structured-streaming
  canary and one bounded Google Cloud structured-streaming canary against the
  exact configured deployments.
- Canaries assert provider identity, valid structured output, non-empty usage,
  and clean stream completion. They use strict request/tool limits and do not
  become ordinary PR blockers.
- Do not run MCP canaries unless an owned test server and credentials already
  exist. Local construction and protocol-contract tests are sufficient for the
  migration POC.

## Rollout and Rollback

- Deploy source and `uv.lock` as one artifact. Do not run mixed V1/V2 workers
  against the same legacy session-history keys until reverse compatibility has
  been tested.
- Before rollout, add a reverse fixture check: serialize a
  representative history with V2 and attempt to read it with the latest-V1
  bridge environment. PydanticAI guarantees V1-to-V2 compatibility, not the
  reverse. Failure does not block the migration itself, but it changes rollout
  and rollback handling.
- If reverse compatibility passes, rollback restores source and lock together.
  If it fails and `SESSION_STORE_URL` selects Redis, take a raw namespace backup
  before cutover and restore that backup with the V1 artifact during rollback;
  canary first with dedicated session IDs. Do not clear user history, add
  dual-read/write logic, or assume V2-written keys are distinguishable. If no
  external session store is configured, no backup step is needed. Do not delete
  PostgreSQL LangGraph checkpoints; they contain application-owned messages and
  are outside this serialization risk.
- Restoring a pre-cutover Redis backup intentionally discards legacy session
  history created or updated during the V2 traffic window. This is an accepted,
  bounded rollback loss for TTL-based conversation memory, not a loss of
  LangGraph checkpoints. General traffic may start after canaries only when the
  operator explicitly accepts that fallback; otherwise remain on V1 until
  reverse compatibility or another recovery approach is proven.
- No Alembic migration, checkpoint rewrite, bulk Redis rewrite, feature flag,
  or dual-version runtime is introduced. Rollback before V2 traffic is simply
  the previous source plus previous lock artifact.

## Acceptance Criteria

1. The final dependency is the latest stable, explicitly reviewed PydanticAI V2
   release and is exact-pinned in both project metadata and `uv.lock`.
2. The latest-V1 bridge passes with zero ignored PydanticAI deprecation
   warnings, and the final tree contains no V1 compatibility path.
3. Every production PydanticAI call site in the post-Agent-Patterns repository
   is represented by the migration matrix or proven unchanged by a contract
   test.
4. Azure remains on explicit Chat Completions, Google remains on Google
   Cloud/Vertex, and both deterministic provider-construction tests pass.
5. Streaming produces the same public output and non-empty normalized usage;
   no PydanticAI result is treated as supporting both method- and property-style
   `usage`.
6. V1 persisted history loads under V2, history processors preserve Tool pairs
   and untouched message fields, all history writes use the strict adapter, and
   no database migration is required.
7. MCP transports, stdio arguments/environment, Tool prefixes/allowlists, and
   intended failure/retry behavior are preserved explicitly.
8. Existing and newly implemented Agent Patterns keep their accepted
   termination, retry, partial-result, Tool-failure, Evidence, Calculation, and
   publication semantics.
9. Ordinary tests cannot make real model requests; complete deterministic,
   static-analysis, disposable-PostgreSQL, and bounded provider-canary gates
   pass in that order.
10. Provider construction uses the PydanticAI-only `httpx2` client seam without
    changing non-PydanticAI HTTP consumers or leaving the known V3 deprecation
    warning.
11. There are zero unresolved review comments. Every review finding is fixed or
    recorded with evidence as a false positive or explicitly owned follow-up.

## Out of Scope

- Redesigning LangGraph Agent Patterns, their state, or their business
  contracts.
- Adopting PydanticAI Harness, deferred capabilities, durable execution, or a
  new Agent framework.
- Moving Azure from Chat Completions to Responses, changing cloud endpoints,
  adding models/providers, or tuning prompts and generation settings.
- Changing expected POC Tool-unavailability semantics to `ModelRetry` or
  `ToolFailed`, adding hidden transport retries, or introducing a Run-wide
  atomic business budget.
- A Run hard deadline, whole-Specialist timeout, global Tool-call accounting,
  or externally mutating Tools.
- SSE replay, background execution, request recovery, human-in-the-loop Resume,
  or same-thread concurrency admission.
- A PostgreSQL schema migration, checkpoint conversion, broad Redis migration,
  or deletion of the legacy Flow Engine.
- New instrumentation or span-schema work unless the post-Agent-Patterns
  Phase-0 inventory finds application code that actually consumes it.
- Expanding the Pydantic Evals corpus or changing its business scoring policy;
  this migration only makes the accepted suite run correctly on V2.

## Review Record

The first independent review reported seven findings. Their dispositions are:

1. The requested change from V1 `1.107.5` to `1.107.0` was rejected as a false
   positive: official PyPI and GitHub releases confirm that `1.107.5` is the
   later V1 maintenance release. The spec now explains the distinction.
2. MCP timeout and `max_retries` semantics were corrected and made testable.
3. The history serializer and rollback plan were corrected; the repository's
   selectable `RedisSessionStore` is now handled by backup/restore, not deletion.
4. A narrow PydanticAI-only `httpx2` client seam now avoids known V3 debt.
5. Root experimental scripts are included in the pytest collection gate.
6. The cumulative-snapshot streaming bug has an explicit fix and counterexample.
7. Speculative instrumentation migration was removed from scope.

The second review's four findings were also closed: MCP error classes and
retry behavior are now distinguished; Redis rollback states its bounded data
loss explicitly; the research record uses `1.107.5` as the bridge throughout
and mentions `1.107.0` only as the historical beta fork baseline (the review
read a stale revision); and the OpenAI SDK 2.x→3.x dependency change is now a
first-class lock and canary gate.

Final independent review: **PASS** (`P0=0`, `P1=0`, `P2=0`, unresolved `0`).

## Primary References

- [PydanticAI latest release](https://github.com/pydantic/pydantic-ai/releases/latest)
- [PydanticAI latest V1 release](https://github.com/pydantic/pydantic-ai/releases/tag/v1.107.5)
- [PydanticAI V2 upgrade guide](https://pydantic.dev/docs/ai/project/changelog/)
- [V1 to V2 migration map](https://pydantic.dev/docs/ai/overview/migration/)
- [PydanticAI version policy](https://pydantic.dev/docs/ai/project/version-policy/)
- [PydanticAI testing guide](https://pydantic.dev/docs/ai/guides/testing/)
- [PydanticAI retries](https://pydantic.dev/docs/ai/core-concepts/retries/)
- [PydanticAI MCP client](https://pydantic.dev/docs/ai/mcp/client/)
- [PydanticAI message history](https://pydantic.dev/docs/ai/core-concepts/message-history/)
- [PydanticAI Google provider](https://pydantic.dev/docs/ai/models/google/)
- [PydanticAI OpenAI/Azure provider](https://pydantic.dev/docs/ai/models/openai/)

## Further Notes

- The version numbers above are a dated research snapshot, while the version
  selection rule is normative: re-check and pin the latest stable V2 at
  implementation start.
- The separate research record
  `docs/research/pydantic-ai-v2-migration-2026-09-05.md` contains the source
  evidence and repository call-site audit behind this spec.
- Approval of this spec does not itself upgrade dependencies or create
  implementation tickets.
