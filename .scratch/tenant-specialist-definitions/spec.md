# Tenant-authored Specialist Definitions

Status: ready-for-agent

This is the English counterpart of `spec.zh-CN.md`; both define the same scope.

## Problem

Tenant administrators cannot add or revise a Specialist without editing Python
registration code. Specialist identity, description, instructions, model
profile, and Skills are declarative content and should not require an application
deployment.

The goal is to borrow VS Code's Markdown Agent authoring approach while reusing
the repository's existing physical three-tier Agent Skills implementation. The
change must preserve the Agent Graph Task, Result, Evidence, Calculation, SSE,
and Tenant-isolation contracts. It does not change model-produced Intent Result
or Intent Catalog schemas.

## Authoring and storage

At startup, load definitions from a Tenant-scoped tree. Development uses a local
root; production uses the same relative layout under a GCS prefix:

```text
tenants/{kmsAppId}/
├── agents/
│   └── {specialist-id}.agent.md
└── skills/
    └── {skill-name}/
        ├── SKILL.md
        └── references/
```

`kmsAppId` is the Tenant ID. Tenant identity comes from the trusted path and
request context, never Markdown. Local and GCS loaders implement one catalog
contract.

`.agent.md` frontmatter contains only `id`, `description`, `model-profile`, and
`skills`; its body is the Specialist instructions. Do not add a role, Intent
allowlist, direct Tool allowlist, or custom input/output schema. The model
profile must resolve through the approved platform registry. Instructions are
limited to 30,000 characters.

The Coordinator sees every valid current-Tenant `id` and `description`. Intent
continues to define trusted data scope. It does not gain
`allowed_specialist_ids` and does not filter Skills.

## Agent Skills

Specialists use the existing Agent Skills loader and `TenantSkillRegistry`:

1. startup discovery loads metadata summaries only;
2. `activate_skill` loads full `SKILL.md` instructions on demand through the
   shared process-local Tier 2 definition cache;
3. `load_reference` reads current direct files under `references/` on every
   call and does not cache reference listings or contents.

Eligible Skills are exactly the Specialist-declared names present in the current
Tenant Skill Catalog, in declaration order. An invocation may
activate zero or more Skills and interleave activation with business Tool calls.
Repeated activation is idempotent and remains effective for that invocation.

There is no `required-skills`, `max_activated_skills`, runtime precedence,
semantic conflict detector, or automatic fallback. Skill conflicts are Tenant
content-quality defects handled by publishing review and evals.

`load_reference` accepts only a Skill activated in the current Tenant and
invocation. A missing or empty directory succeeds with an empty set. Listing or
read failure returns a bounded failure with no partial contents. Edits are
visible on the next call.

## Tool authority

Tool definitions come only from the global platform Registry. A Markdown
Specialist's bound business Tools are:

```text
Global Tool Registry
∩ Tenant Tool Policy
∩ Research Scope
∩ union(allowed-tools of the Specialist's declared valid Skills)
```

`allowed-tools` is a restrictive ceiling, not an authority grant. Every name
must exist in the global registry or the Skill is skipped at startup. A Tool
omitted from the declared Skills is not bound even when broader policy permits
it.

The Graph Specialist runtime does not consume `required-tools` as a second
eligibility policy. That legacy field remains part of the existing FlowEngine
Agent Skills implementation. The Graph path ignores it rather than creating a
second Skill schema.

Activation and reference loading never rebind business Tools. Bundled Skill
scripts, binaries, and assets are not executed.

## Instruction precedence

Platform Specialist instructions and fixed security guards precede Tenant
Specialist instructions. Activated Skill instructions enter later through the
activation result. Text ordering is not the security boundary: code owns Tenant
identity, Research Scope, Tool bindings, Evidence and Calculation validation,
coordination limits, and graph routing.

## Graph contracts and state

Retain the existing trust boundaries:

- the model emits `TaskProposal`; code validates it into `AcceptedTask`;
- a Specialist receives `SpecialistTaskInput` and returns `SpecialistAttempt`;
- deterministic acceptance creates `SpecialistResult` and `TaskOutcome`;
- the batch barrier atomically promotes the exact manifest to `AcceptedBatch`.

The accepted Task manifest is stored only in `CoordinationRound.tasks`. An
unaccepted final dispatch round is the pending batch; there are no separate
`ActiveBatch` or `CoordinationTask` models.

Checkpoint state stores accepted facts rather than derived mirrors.
Coordination completion is derived from a terminal round or bounded stop reason.
Public `completion_status` and `termination_reason` remain response metadata but
are derived during finalization from `incomplete_research`; they are not
checkpoint channels. Invocation-only failed-attempt diagnostics are not stored.

The runtime does not compute or persist Specialist Definition Pins or Skill
Pins. They enforce no authority and cannot provide exact replay when execution
uses the current process catalog and references are live. Task outcomes, usage,
startup logs, and normal tracing supply current observability.

The PydanticAI version remains fixed by `pyproject.toml` and the lockfile; the
runtime does not duplicate that dependency contract with version checks.

## Startup, recovery, and failures

Agent and Skill definition edits take effect after restart; one process uses its
loaded catalog. References remain live.

Invalid entries are skipped individually and logged with Tenant, source
identity, definition kind, and a stable reason. Valid siblings continue. Logs do
not include full instructions, references, or secrets.

After recovery, the Coordinator reads the current Tenant catalog. The Run does
not persist a historical Specialist ID snapshot, while its original Tool,
source, query, freshness, and other data constraints remain. A legitimately
accepted Task whose current definition is missing produces its own failed
outcome; valid siblings continue. Damaged identities or accepted decisions
remain invariant failures.

## Acceptance criteria

- Local startup drives a Markdown Specialist through existing HTTP/SSE
  selection, dispatch, execution, and publication.
- Local and GCS share loader tests for layout, Tenant isolation, skip/log,
  live references, deterministic order, and no partial success.
- Different Intents receive the same current Tenant Specialist descriptors while
  preserving different data scopes.
- Skill tests cover zero, one, multiple and repeated activations, unknown names,
  and activation interleaved with business Tools.
- `allowed-tools` actually narrows the Specialist's bindings and cannot expand
  registry, Tenant, or Research Scope authority.
- `required-tools` does not affect Graph Skill eligibility; FlowEngine retains
  its existing contract.
- Checkpoints contain no duplicate Task manifest, Pins, derived completion
  channels, or failed-message diagnostics. Public completion metadata is stable.
- Code and Markdown adapters share one input, output, Evidence, Calculation, and
  batch-acceptance contract during migration.
- Focused unit/eval and affected PostgreSQL integration tests, Ruff, Pyright,
  and `git diff --check` pass.

## Out of scope

- Markdown Coordinator, Query Understanding, or Synthesis actors.
- Exact VS Code custom-agent compatibility.
- Intent Agent/Skill allowlists or changes to Intent schemas.
- `required-skills`, `max_activated_skills`, or runtime conflict resolution.
- Agent/Skill hot reload, reference caching, historical snapshots, or exact
  replay.
- Atomic all-or-nothing Tenant catalog loading.
- Executing code from Skill packages.
- Tenant publishing, approval, or eval-authoring UI.
- Refactoring the legacy FlowEngine request path.
