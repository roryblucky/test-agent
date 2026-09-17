# Load Tenant Specialist definitions from Markdown

Status: accepted

Tenant administrators need to add or revise Specialist Agents without changing
Python registration code. We load Tenant-owned Specialist Definitions from
Markdown, reuse the repository's Agent Skills loader and registry, and adapt
Markdown-defined Specialists to the existing Graph execution contracts. The
authoring format borrows useful conventions from VS Code agent files and the
open Agent Skills format without claiming exact VS Code compatibility.

## Definition and storage layout

Each Tenant uses its `kmsAppId` as its Tenant ID and owns one relative tree:

```text
tenants/{kmsAppId}/
├── agents/
│   └── {specialist-id}.agent.md
└── skills/
    └── {skill-name}/
        ├── SKILL.md
        └── references/
```

Development reads this tree from a local root. Production reads the same
relative structure from a configured GCS prefix. Storage-specific behavior
stays behind loaders and both environments build the same runtime catalog.

A `.agent.md` file has YAML frontmatter containing only `id`, `description`,
`model-profile`, and `skills`. Its Markdown body is the Specialist's
instructions. The owning path supplies Tenant identity. The runtime supplies
the common `SpecialistTaskInput -> SpecialistResult` contract, platform
instructions, security guards, and Tool bindings. A definition cannot declare
its own Tenant, Intent allowlist, Tool allowlist, role, or output schema.

`model-profile` must resolve to a platform-approved profile. Specialist
instructions are limited to 30,000 characters. An invocation exposes all
eligible Skill summaries in declaration order. This phase applies only to
Specialists; Coordinator and Synthesis remain platform-defined.

## Catalog and selection

Startup validates each Tenant's Specialist definitions and discovers Skill
metadata. Invalid entries are skipped and logged without rejecting valid
siblings. Agent and Skill definition changes take effect on restart. Reference
files are the exception: they are read live on each `load_reference` call.

The Coordinator sees every valid `id` and `description` in the current Tenant
catalog. Intent does not contain an Agent or Skill allowlist. Intent policy
continues to define the data scope: Tools, sources, queries, freshness, and
other business constraints.

The model proposes a Task and deterministic code accepts it as an
`AcceptedTask`. The accepted Task is stored once in its `CoordinationRound`.
An unaccepted final dispatch round is the pending batch; no separate active
batch checkpoint exists. The batch barrier still validates and atomically
promotes the exact set of Task outcomes; these two trust boundaries remain.

## Skill activation and references

Skills retain the repository's physical three-tier behavior:

1. startup discovery exposes metadata summaries;
2. `activate_skill` loads full `SKILL.md` instructions on demand through the
   shared `TenantSkillRegistry`, whose Tier 2 definition cache is process-local;
3. `load_reference` reads the current direct files under `references/` every
   time and does not cache their contents.

A Specialist's eligible Skills are exactly its declared names that exist in
the current Tenant Skill Catalog, in declaration order. An
invocation may activate zero or more of them, including between business Tool
calls. Repeated activation is idempotent. There is no `required-skills` or
`max_activated_skills` setting. Skill conflicts are Tenant content-quality
problems handled by publishing review and evals, not runtime precedence logic.

An absent or empty references directory returns a successful empty result. A
listing or read failure returns a bounded failure and no partial contents.
Neither full instructions nor references are persisted in graph state.

## Tool authority and instruction precedence

Tools are registered globally by the platform. For a Markdown Specialist, the
business Tools bound to an invocation are:

```text
Registered Tools
∩ Tenant Tool Policy
∩ Research Scope
∩ union(allowed-tools of the Specialist's declared valid Skills)
```

`allowed-tools` is a restrictive ceiling, not a grant. Every listed name must
exist in the global registry or the Skill is skipped at startup. A Tool omitted
from the declared Skills is not bound even if broader platform policy permits
it. Graph Specialist eligibility does not consume `required-tools`; that legacy
field remains available to the FlowEngine Skill implementation and is not a
second Graph policy.

Executable capability always comes from the Tool Registry; bundled Skill
scripts are not executed. Platform Specialist instructions and fixed security
guards precede Tenant Specialist instructions. Activated Skill instructions
arrive later through the activation result. Instructions cannot widen Tenant
identity, Research Scope, Tool bindings, validation, or graph routing.

## Runtime state and observability

The runtime does not compute or persist Specialist Definition Pins or Skill
Pins. They did not enforce authority or provide exact replay because definitions
are loaded from the current process catalog and references are live. Existing
Task outcomes, bounded usage accounting, startup logs, and normal traces provide
the required operational observability without hashing authoring content into
checkpoint state.

Graph state stores accepted facts, not derived mirrors. Coordination completion
is derived from a terminal `CoordinationRound` or a bounded stop reason. Public
`completion_status` and `termination_reason` remain in response metadata but are
derived during finalization from `incomplete_research`; they are not separate
checkpoint channels.

The reviewed PydanticAI version is already fixed by `pyproject.toml` and the
lockfile. The runtime does not duplicate dependency resolution with a startup
or per-attempt version check.

## Migration and consequences

Code-defined and Markdown-defined adapters temporarily share the existing
Specialist input/output contract for comparison. Markdown definitions are the
production target; Graph code does not gain a general plugin framework.

Tenant isolation follows the `kmsAppId`-scoped storage path, catalog, trusted
request context, and Tool binding. Descriptions and Skill content require
Tenant publishing review and evals. Definitions are stable for a process
lifetime, while references intentionally remain live.

Rejected alternatives include Intent Agent allowlists, historical definition
snapshots, hot reload of Agent or Skill instructions, atomic all-or-nothing
Tenant catalog loading, runtime Skill-conflict resolution, bundled code
execution, and a second Graph-specific Agent Skills implementation.
