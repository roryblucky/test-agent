# Load Tenant Specialist definitions from Markdown

Status: accepted

Tenant administrators need to add or revise Specialist Agents without changing
Python registration code. We will load Tenant-owned Specialist Definitions from
Markdown, use Agent Skills for reusable instructions, and adapt both Markdown-
defined and existing code-defined Specialists to the same runtime contracts. The
design deliberately borrows portable conventions from VS Code agent files and
the open Agent Skills format without promising exact VS Code compatibility.

## Definition and storage layout

Each Tenant uses its `kmsAppId` as its Tenant ID and owns one relative definition
tree:

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
relative structure from a configured GCS prefix. Storage-specific code remains
behind loaders so the Specialist runtime consumes one common catalog model.

A `.agent.md` file has YAML frontmatter containing only:

- `id`
- `description`
- `model-profile`
- `skills`

Its Markdown body is the Specialist's instructions. It does not declare a role,
Tenant ID, Intent allowlist, Tool allowlist, or output schema. The owning path
supplies the Tenant identity, and the runtime supplies common input and output
contracts. `model-profile` must resolve to a platform-approved model profile.
Specialist instruction bodies are limited to 30,000 characters, and one
Specialist exposes at most 20 eligible Skill summaries.

The first phase applies only to Specialist Agents. Coordinator and Synthesis
actors remain platform-defined.

## Catalog and selection

Startup parses and validates each Tenant's Specialist and Skill definitions. It
loads complete `.agent.md` and `SKILL.md` content into an in-memory Tenant
catalog, but exposes only compact Specialist descriptors and Skill summaries to
models. A malformed definition is skipped and logged without rejecting the
Tenant's entire catalog.

The Coordinator receives every valid Specialist descriptor in the current
Tenant and chooses a Specialist from its `id` and `description`. Business Intent
does not carry `allowed_specialist_ids` and does not directly filter the
Specialist or Skill catalogs. Intent policy instead produces the current
Research Scope, including permitted sources and Tools.

Agent and `SKILL.md` changes become effective only after process restart. Every
Coordinator decision, including after checkpoint resume, uses all valid
Specialist descriptors in the current process's Tenant catalog. The Run retains
its original Tool, source, query, freshness, and other data constraints. A
legitimately dispatched Task whose Specialist Definition has since disappeared
produces a failure outcome for that Task without preventing valid sibling Tasks
from running; a forged Task, invalid identity, or damaged Batch manifest remains
an invariant failure. An existing ID uses the definition loaded by the current
process, and the accepted Specialist attempt or outcome records the content hash
of the definition it actually used.

## Skill activation and references

Skills follow the open Agent Skills progressive-disclosure model: discovery
shows only metadata, and activation adds the cached full `SKILL.md` instructions
to the current Specialist invocation. This is progressive disclosure to the
model, not lazy storage access.

A Specialist invocation computes its eligible Skill set once from the loaded
definitions and frozen effective Tools. It may progressively activate zero or
more members of that fixed set before or after business Tool calls. There is no
separate `max_activated_skills` setting, each activation remains effective for
the rest of the invocation, and there is no `required-skills` dependency
mechanism. A Skill Pin contains the Skill name, its content hash, and the optional
`metadata.version` value when the author supplied one. The content hash identifies
the actual cached definition when no version is declared; version is not a
required authoring field.

Activated Skills are peers. The runtime does not assign precedence, detect
semantic conflicts, reorder instructions, degrade behavior, or provide a code-
level fallback for conflicting Skills. Such conflicts are Tenant-authored
content defects handled through publishing review and evaluations.

Reference contents are not loaded or cached at startup. When the model needs
them, `load_reference` reads and returns all files in that Skill's `references/`
directory. Each call sees the latest stored contents without requiring restart;
the runtime does not pin or preserve a historical reference version. An absent or
empty directory returns an empty result. A listing or file-read failure returns an
explicit Tool failure and no partial contents, so incomplete data is never
reported as a complete reference set.

## Tool authority and instruction precedence

Tools are registered in the platform's global Tool Registry. A Specialist does
not own a separate Tool allowlist. Its effective Tools are the intersection of
Tenant Tool policy and the current Research Scope:

```text
Effective Tools = Tenant Tool Policy ∩ Research Scope
```

`required-tools` declares the hard Tool dependencies used to decide whether a
Skill is eligible for an invocation. `allowed-tools` is portable usage guidance
for Tools the Skill may discuss when they are otherwise available; it neither
grants Tools nor makes them prerequisites. Every Tool name in either field must
resolve in the global Tool Registry or the Skill is invalid and skipped at
startup. If a `required-tools` Tool exists but the current Research Scope does
not permit it, that Skill is omitted from the Specialist's eligible summaries.
Executable capability must come from the Tool Registry; the runtime does not
execute bundled Skill scripts.

Platform instructions and fixed security guards have authority over Specialist
and Skill instructions. Specialist instructions are appended to the platform
instructions, and activated Skill instructions are subsequently introduced into
the invocation. Neither can widen Tenant authority, Research Scope, Tool
bindings, or graph routing. We do not add a semantic instruction-conflict
detector; the platform instruction boundary remains authoritative.

## Runtime contract and migration

Both implementation styles use the existing common
`SpecialistTaskInput -> SpecialistResult` contract. During migration, code-
defined and Markdown-defined adapters remain available and must pass the same
contract tests. The production target is Markdown-defined Specialists only;
the code-defined adapter exists to compare behavior and de-risk the transition,
not as a permanent second authoring model.

## Considered alternatives

- Keeping Specialists registered only in Python was rejected because Tenant
  administrators would still require a code change and deployment to author one.
- Exact VS Code agent-file compatibility was rejected because this runtime has
  different Tenant, Tool, Intent, and execution contracts.
- Filtering Specialists through Intent `allowed_specialist_ids` was rejected;
  the Coordinator should select from all valid Tenant Specialist descriptors,
  while Intent policy controls data and Tool scope.
- Preserving or intersecting a historical Specialist ID snapshot on resume was
  rejected because every Coordinator decision should use the current Tenant
  catalog; the original Run's Tool, source, query, and freshness constraints
  continue to prevent data-authority expansion.
- Hot-reloading Agent and Skill instructions was rejected in favor of a stable
  process-lifetime definition snapshot. References intentionally remain live.
- Atomic all-or-nothing Tenant catalog loading was rejected in favor of skipping
  and logging individual invalid definitions.
- A Specialist-specific Tool allowlist was rejected because Tenant policy and
  Research Scope already own Tool authority, while Skills only express
  dependencies and usage guidance.

## Consequences

Tenant isolation follows the `kmsAppId`-scoped path, catalog, and Tool context.
Description quality becomes important because Coordinator and Skill selection
are model-driven, so publishing review and evaluations are required operational
controls. Definitions are stable for a process lifetime, but live references
and latest-definition retries mean historical execution is observable by hashes
rather than exactly replayable from persisted definition contents.
