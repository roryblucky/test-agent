# Tenant-authored Specialist Definitions

Status: ready-for-agent

## Problem Statement

Tenant administrators currently cannot add or revise a Specialist Agent without
changing Python registration code. That makes a Tenant-owned content decision
depend on an application code change and deployment, even though the Specialist's
identity, description, instructions, model profile, and Skills are declarative.

The existing runtime also has two different Skill implementations and a
code-defined Specialist registration model whose Specialist, Skill, and Tool
allowlists overlap with authority already owned by Tenant policy and Research
Scope. This makes it harder to explain which layer can grant execution authority
and harder to compare a Markdown-defined Specialist with the existing
code-defined implementation.

The feature must let a Tenant administrator author a Specialist Definition and
its Skills as Markdown while preserving the existing Agent Graph execution and
result contracts. It must enforce Tenant isolation through `kmsAppId`, preserve
platform security instructions, keep Tool authority in trusted platform policy,
and avoid changing the current model-produced Intent Result or Intent Catalog
structure.

## Solution

Load each known Tenant's Specialist Definitions and Skills at application startup
from a Tenant-scoped definition tree. Development uses a local directory;
production uses the same relative tree below a configured GCS prefix. Both
storage adapters produce the same in-memory Specialist Catalog and Skill Catalog.

A `.agent.md` document provides a Specialist's identity, description, approved
model profile, named Skills, and instruction body. Its owning path supplies the
Tenant ID. A standard `SKILL.md` document provides Skill discovery metadata,
instructions, and Tool dependencies. Complete Agent and Skill definitions are
validated and cached for the process lifetime, while Skill reference files remain
live and are read only when an Activated Skill calls `load_reference`.

The Coordinator Agent sees every valid Specialist Descriptor in the current
Tenant's Specialist Catalog and selects by ID and description. Business Intent
continues to select trusted scope constraints, but does not select or authorize
Specialists and does not gain an `allowed_specialist_ids` field. At execution,
the runtime adapts Markdown-defined and code-defined Specialists to the existing
Specialist actor and Agent Graph contracts. Both receive the same
Specialist Task Input, produce the same Specialist Attempt, and pass through the
same deterministic acceptance step to the same Specialist Result and Task
Outcome.

Specialists may activate zero or more Eligible Skills progressively and in any
interleaving with business Tool calls. Skills guide already-authorized Tool use;
they do not grant Tool authority. Effective Tools remain the intersection of the
platform Tool Registry, Tenant Tool policy, and the current Research Scope.
Platform instructions and fixed security guards always take precedence over
Tenant-authored Specialist and Skill instructions.

## User Stories

1. As a Tenant administrator, I want to define a Specialist Agent in Markdown, so that I can add domain expertise without editing Python registration code.
2. As a Tenant administrator, I want a Specialist Definition to contain an ID, description, approved model profile, Skill names, and instructions, so that its runtime behavior is understandable from one document.
3. As a Tenant administrator, I want the Specialist Definition's Markdown body to be its instructions, so that I can author guidance in a familiar format.
4. As a Tenant administrator, I want to reuse one Skill across multiple Specialist Definitions, so that shared guidance is maintained once.
5. As a Tenant administrator, I want to list multiple Skills for one Specialist Agent, so that it can progressively combine relevant guidance for a Task.
6. As a Tenant administrator, I want a Specialist Agent to work without any Skills, so that simple responsibilities do not require an artificial Skill package.
7. As a Tenant administrator, I want Skill conflicts treated as a publishing-quality defect, so that I can resolve them through review and evaluations rather than hidden runtime precedence.
8. As a Tenant administrator, I want malformed definitions to be skipped with useful logs, so that one content error does not prevent every valid Specialist in my Tenant from loading.
9. As a Tenant administrator, I want Agent and Skill changes to take effect after restart, so that one process uses a stable definition snapshot.
10. As a Tenant administrator, I want reference-file edits to be visible on the next `load_reference` call, so that supporting material can be corrected without restarting the runtime.
11. As a Tenant administrator, I want Specialist descriptions to be shown to the Coordinator Agent, so that delegation can be driven by declared responsibilities.
12. As a Tenant administrator, I want the platform to reject unknown model profiles, so that Markdown cannot select an unapproved model deployment.
13. As a Tenant administrator, I want a clear instruction-length limit, so that a single Specialist Definition cannot consume an unbounded prompt budget.
14. As a Tenant administrator, I want at most twenty Eligible Skill summaries exposed for a Specialist invocation, so that discovery remains bounded.
15. As a Tenant administrator, I want every Tool named by a Skill validated against the platform Tool Registry, so that broken Skill packages are detected at startup.
16. As a Tenant administrator, I want a Skill whose required Tool is outside the current Research Scope omitted from discovery, so that the model is not invited to activate unusable guidance.
17. As a Tenant administrator, I want Skill reference contents excluded from startup loading, so that large supporting documents do not inflate startup memory or prompt context.
18. As a Coordinator Agent, I want all valid Specialist Descriptors from the current process's Tenant Catalog, including after checkpoint resume, so that I can choose the best Specialist independently of the selected Business Intent.
19. As a Coordinator Agent, I want only IDs and descriptions during Specialist selection, so that Specialist instructions and Skill details do not pollute coordination context.
20. As a Coordinator Agent, I want an invalid or cross-Tenant Specialist ID rejected by deterministic validation, so that model output cannot escape the current Tenant's Specialist Catalog.
21. As a Specialist Agent, I want the same Specialist Task Input regardless of how I was defined, so that Markdown authoring does not create a second execution protocol.
22. As a Specialist Agent, I want to produce the existing Specialist Attempt and Specialist Result shapes, so that the Agent Graph, synthesis, and clients do not need Markdown-specific handling.
23. As a Specialist Agent, I want to see only compact summaries of my Eligible Skills initially, so that full instructions are disclosed only when useful.
24. As a Specialist Agent, I want to activate zero or more Eligible Skills, so that I can match the amount of guidance to the Task.
25. As a Specialist Agent, I want to activate another Eligible Skill after using a business Tool, so that discovery and execution can be interleaved naturally.
26. As a Specialist Agent, I want a repeated activation of the same Skill to be idempotent, so that retries do not duplicate its effective instructions or pins.
27. As a Specialist Agent, I want each Activated Skill to remain effective for the rest of my invocation, so that I can follow combined guidance consistently.
28. As a Specialist Agent, I want `load_reference` to return every reference file or an explicit failure, so that I do not mistake partial content for complete supporting context.
29. As a Specialist Agent, I want `load_reference` to read current storage on every call, so that I never receive a stale process cache of reference content.
30. As a Specialist Agent, I want an ineligible or unknown Skill activation rejected, so that I cannot discover arbitrary Tenant content by guessing names.
31. As a Specialist Agent, I want only effective business Tools bound into my invocation, so that prompt instructions cannot expand executable authority.
32. As a Specialist Agent, I want expected Tool unavailability to continue through the existing typed outcome path, so that Markdown authoring does not change Data Gap behavior.
33. As a platform security owner, I want `kmsAppId` to identify both the Tenant and its definition-tree prefix, so that one Tenant's Agent cannot load another Tenant's definitions.
34. As a platform security owner, I want Tenant identity to come from trusted request and storage context rather than Markdown frontmatter, so that content cannot self-assign a Tenant.
35. As a platform security owner, I want Specialist and Skill instructions appended below fixed platform instructions, so that Tenant content cannot replace security guards.
36. As a platform security owner, I want Agent and Skill content unable to widen Research Scope, Tool bindings, Tenant authority, or Agent Graph routing, so that instructions remain guidance rather than policy.
37. As a platform security owner, I want executable capabilities to come only from the Tool Registry, so that a Skill package cannot execute bundled scripts.
38. As a platform security owner, I want no Specialist-specific Tool allowlist, so that authority has one explainable source in Tenant Tool policy and Research Scope.
39. As a platform security owner, I want Business Intent to remain free of `allowed_specialist_ids`, so that Business Intent classifies business purpose rather than acting as an Agent permission list.
40. As a platform operator, I want local and GCS storage adapters to share one definition-loading interface, so that environment selection does not change runtime behavior.
41. As a platform operator, I want all Agent and Skill definitions loaded before the application accepts traffic, so that a process does not serve a partially initialized catalog.
42. As a platform operator, I want invalid entries logged with Tenant, source, definition kind, and validation reason, so that content owners can repair them without exposing file contents or secrets.
43. As a platform operator, I want a Tenant-scoped count of loaded and skipped Specialists and Skills, so that startup health is observable.
44. As a platform operator, I want a storage or validation failure to fail closed for the affected definition, so that the runtime never borrows a definition from another Tenant or silently grants broader behavior.
45. As an auditor, I want an accepted Specialist execution to record the Specialist Definition Pin it used, so that I can correlate a Task Outcome with the observed configuration.
46. As an auditor, I want Activated Skill Pins recorded in first-activation order with an optional author-declared version and a required content hash, so that versionless Skills still identify the exact cached definitions that guided an accepted attempt.
47. As an auditor, I want live reference content excluded from definition pins, so that a pin is not misrepresented as a guarantee of exact historical replay.
48. As an application developer, I want code-defined and Markdown-defined Specialist Adapters to pass the same contract suite, so that migration behavior can be compared directly.
49. As an application developer, I want the existing code-defined Adapter retained during migration, so that the Markdown path can be introduced without a flag-day replacement.
50. As an application developer, I want Markdown-defined Specialists to be the production target, so that the transitional code Adapter does not become a permanent second authoring system.
51. As an application developer, I want one Specialist Catalog interface to hide parsing, validation, storage, and indexing details, so that the Agent Graph does not learn storage-specific behavior.
52. As an application developer, I want to reuse the existing Agent Skills schema and loader behavior where their semantics match, so that the Specialist runtime does not invent a second `SKILL.md` dialect.
53. As an application developer, I want the existing model-produced Intent Result and Intent Catalog item schemas unchanged, so that query understanding remains backward compatible.
54. As an application developer, I want a resumed or retried Task to resolve through the current process catalog, so that runtime recovery does not require persisting full historical Markdown.
55. As an evaluator, I want deterministic scenarios for ambiguous descriptions, multiple Skill activations, and Tool-scope restrictions, so that Tenant content quality can be measured before publishing.
56. As a platform operator, I want a legitimately dispatched Task whose Specialist Definition was removed to fail independently, so that valid sibling Tasks in the same Batch can still complete.

## Implementation Decisions

### Authoring and storage contract

- A Tenant is identified by `kmsAppId`. The platform enumerates known Tenants
  from trusted configuration and loads only their corresponding definition
  prefixes; it does not discover new Tenants from storage contents.
- The storage contract is the relative tree
  `tenants/{kmsAppId}/agents/{specialist-id}.agent.md` and
  `tenants/{kmsAppId}/skills/{skill-name}/SKILL.md`, with optional files below
  each Skill's `references` directory.
- Development selects a Local Adapter rooted at a configured directory.
  Production selects a GCS Adapter rooted at a configured bucket and prefix.
  Both satisfy one Loader interface and return the same source documents and
  source identities to the catalog implementation.
- Storage selection is environment configuration, not a field in Tenant-authored
  Markdown. Runtime modules consume catalogs and do not branch on Local versus
  GCS.
- A Specialist Definition's YAML frontmatter contains only `id`, `description`,
  `model-profile`, and `skills`. Its Markdown body is the instruction content.
  There is no `role`, Tenant ID, Intent allowlist, Tool allowlist, output schema,
  or `required-skills` field.
- `model-profile` resolves through the Tenant's platform-approved Model Registry.
  A Specialist Definition whose profile cannot be resolved is invalid.
- A Specialist instruction body is limited to 30,000 characters. Empty or
  structurally invalid definitions are skipped and logged.
- Skills retain the repository's Agent Skills-compatible frontmatter and
  Markdown format. `required-tools` declares hard dependencies used for
  invocation eligibility. `allowed-tools` is portable usage guidance for Tools
  the Skill may discuss when they are otherwise available; it neither grants nor
  filters the Specialist's effective Tools, and it does not make a Tool required.
  Every Tool name in either field must resolve in the platform Tool Registry.
- Bundled Skill scripts are not loaded or executed. References are data read by
  `load_reference`, not executable capabilities.

### Catalog module and startup lifecycle

- Build one deep catalog module whose interface returns the current Tenant's
  Specialist Catalog, Skill Catalog, compact Specialist Descriptors, and
  invocation-specific Eligible Skill summaries. Parsing, validation, hashing,
  indexing, and invalid-entry logging remain implementation details behind that
  interface.
- Application startup loads every known Tenant after trusted Tenant
  configuration and Model Registries exist and before request traffic is
  accepted. The startup composition installs Tenant-scoped catalogs where the
  Agent runtime can resolve them from the trusted request context.
- Complete `.agent.md` and `SKILL.md` definitions are read, parsed, validated,
  and cached at startup. Progressive disclosure refers only to what the model
  sees; it is not lazy loading of Skill definitions from storage.
- Reference file contents are neither read nor cached at startup. The catalog
  retains only enough trusted source identity to locate the activated Skill's
  reference directory later.
- One invalid Specialist or Skill is skipped without rolling back other valid
  definitions for that Tenant. Errors are logged with bounded identifying
  metadata and no full instruction or reference contents.
- A Skill with any Tool name in `required-tools` or `allowed-tools` absent from
  the platform Tool Registry is invalid and excluded from the Skill Catalog. A
  Specialist Definition that names a Skill absent from the
  successfully loaded Skill Catalog is invalid and excluded from the Specialist
  Catalog; the runtime does not silently rewrite the authored Skill list.
- Catalog membership is immutable for the process lifetime. Agent or Skill edits
  require restart. A storage failure cannot reuse another Tenant's catalog or
  expand catalog membership.
- Existing Agent Skills parsing and Local/GCS storage implementations are prior
  art and should be reused or deepened where their contracts match. The Agent
  Graph must not keep a separate persisted Skill schema or parser. The existing
  FlowEngine Agent Handler remains a separate caller during this phase and must
  not dictate incompatible Specialist invocation semantics.

### Intent, scope, and Coordinator selection

- The model-produced Intent Result and Intent Catalog item structures remain
  unchanged. Do not add `allowed_specialist_ids`, Skill IDs, Tool IDs, or Agent
  routing fields to model output.
- Intent Policy remains trusted Tenant configuration for Tools, sources, search
  constraints, Evidence freshness, and other non-Agent Research Scope limits.
  It does not filter the Specialist Catalog or Skill Catalog.
- Specialist descriptors currently carried as Intent-specific policy data cease
  to be an authority source. Whenever the Coordinator runs or a Task is
  dispatched, including after checkpoint resume, the runtime projects every
  valid descriptor and executable definition from the current process's Tenant
  Specialist Catalog. A newly added Specialist is therefore available to a
  resumed Run, while a removed Specialist is absent.
- Checkpoint resume preserves the original Run's Tool, source, query, freshness,
  and other data constraints. Any persisted Specialist descriptors are a stale
  projection and must not restrict or expand the current catalog. When an
  existing ID has changed, the current description and definition are used and
  the attempt records the current definition pin.
- Intent-specific `allowed_skill_names` likewise ceases to filter Skill
  discovery. Skill eligibility comes from the Specialist Definition, valid Skill
  Catalog membership, and `required-tools` availability in the current invocation.
- The Coordinator prompt receives compact Specialist Descriptors only. It never
  receives Specialist instruction bodies, model profiles, or Skill lists during
  selection.
- Existing deterministic Coordinator-decision validation continues to reject an
  ID outside the supplied Tenant Specialist Catalog. Model choice never bypasses
  registry and Tenant checks.

### Common Specialist runtime contract

- Preserve the existing Specialist execution seam. A Specialist Actor receives
  `SpecialistTaskInput` and returns `SpecialistAttempt`; the Agent Graph's
  deterministic acceptance step produces the existing `SpecialistResult` inside
  the `TaskOutcome`. No Markdown-specific Task, attempt, result, or SSE schema is
  introduced.
- The runtime catalog resolves a selected Specialist into the descriptor,
  approved model profile, instructions, Skill names, definition pin, and actor
  factory inputs needed for one invocation. The Agent Graph does not parse files
  or construct storage paths.
- A Markdown Adapter creates the existing PydanticAI Specialist Actor using the
  resolved Tenant Model Registry, platform Specialist instructions, fixed
  security guards, the Tenant-authored Specialist instructions, the frozen Tool
  bindings, and an invocation-local Skill activation interface.
- A code-defined Adapter remains available during migration and supplies the
  same runtime information and actor interface from trusted code. Its
  registrations must carry a prompt-visible description and a stable definition
  identity rather than depending on per-Intent descriptors.
- Direct test actors remain supported through the code Adapter, but production
  composition targets Markdown definitions. Do not create a generic framework
  for Coordinator or Synthesis actors as part of this work.
- A Specialist Definition Pin is a SHA-256 content hash of a canonical
  representation of the parsed Specialist metadata and instruction body. It
  excludes storage location, reference contents, and mutable runtime state.
- Each Specialist attempt carries the pin used to create its actor. Acceptance
  copies the pin into the accepted Specialist Result or Task Outcome so it is
  durable with the Task record. Full historical Specialist content is not stored.
- A resumed or retried Task resolves its selected ID through the current
  process's catalog. It does not restore an earlier process's Definition body;
  if the ID still exists, the recorded pin makes any content change observable.
- Separate Batch integrity from current Specialist availability. Batch-level
  validation still rejects invalid Batch or Task identities, malformed
  objectives or context, forged Specialist IDs, and Tasks that do not match the
  accepted Coordinator decision. Once a Task is proven to have been legitimately
  dispatched, resolve only that Task's Specialist against the current catalog.
  If its definition has disappeared, produce a failed Task Outcome without a
  model call and allow other valid Tasks in the same Batch to continue. Do not
  substitute another Specialist.

### Skill discovery, activation, and references

- For one Specialist invocation, an Eligible Skill must be named by that
  Specialist Definition, be present in the Tenant Skill Catalog, and have every
  `required-tools` dependency present in the invocation's effective Tool set.
- If a `required-tools` dependency exists globally but is excluded by Tenant Tool
  policy or Research Scope, omit that Skill from the invocation's Eligible Skill
  summaries. An unavailable `allowed-tools` entry does not make the Skill
  ineligible. Do not expose a Skill with an unmet hard dependency and then rely
  on activation-time failure as the normal path.
- Preserve declaration order when projecting Eligible Skill summaries and expose
  at most the first twenty. Full Skill instructions are absent from the initial
  Specialist prompt.
- Compute and freeze the Eligible Skill set when the Specialist invocation
  begins. The invocation may progressively activate zero or more members of that
  set before or after any business Tool call; remove the current single-activation
  restriction.
- Re-activating the same Skill is idempotent. Activated Skill Pins retain
  first-activation order without duplicates, and each Activated Skill remains
  effective until that Specialist invocation ends.
- A Skill Pin contains the Skill name, an optional version, and the required
  content hash of the cached Skill Definition. Read the version from
  `metadata.version` when present; omit it otherwise. Version is not a required
  authoring field, and reference contents are excluded from the hash.
- Skill definitions are peers. The runtime does not set precedence, reorder
  instructions, detect semantic conflicts, degrade a Skill, or provide an
  automatic fallback. Tenant publishing review and evaluations own that quality
  control.
- `load_reference` is available only within a Specialist invocation with Skills.
  It accepts an Activated Skill identity and reads every regular file in that
  Skill's `references` directory in deterministic filename order.
- Every `load_reference` call goes to the selected storage Adapter and returns
  the current contents. It bypasses process and invocation reference caches.
  An absent or empty directory succeeds with an empty result. A listing failure
  or any file-read failure produces an explicit bounded Tool failure and returns
  no partial contents. Reference contents are not attached to the cached Skill
  Definition and are not included in Specialist or Skill pins.
- Activation and reference loading do not register business Tools and do not
  alter the frozen effective Tool set. Their model-visible responses contain
  instructions or reference content, not new authority.

### Tool authority and instruction precedence

- The platform Tool Registry is the only source of executable Tool definitions.
  Existing Evidence and Calculation Tool registrations must be exposed through
  that common registry seam rather than copied into a Specialist-owned registry.
- Effective business Tools are the registered Tools in the intersection of
  Tenant Tool policy and the current Research Scope. Remove the
  Specialist-registration Tool allowlist from effective Tool calculation.
- `required-tools` is validation and eligibility metadata; `allowed-tools` is
  validation and usage guidance only. Neither field can register, grant, rebind,
  filter, or broaden a Tool, and a Specialist instruction cannot do so either.
- Effective Tools are resolved and frozen before constructing the Specialist
  Actor. Skill activation and live reference changes cannot mutate that set.
- Platform Specialist instructions and fixed security guards are composed first.
  The Tenant-authored Specialist instructions are appended after them. Activated
  Skill instructions enter later through the model-visible activation result.
- Instruction ordering does not replace deterministic enforcement. Tenant
  identity, Research Scope, Tool bindings, Evidence validation, Calculation
  validation, coordination limits, and graph routing remain code-owned.
- Do not add a semantic prompt-conflict detector. Content that asks to override
  platform rules has no authority because executable interfaces and graph
  transitions remain constrained by code.

### Migration and observability

- Keep code-defined and Markdown-defined Specialist Adapters side by side only
  for behavior comparison and migration. Both must pass the same contract suite;
  production configuration selects the Markdown catalog target.
- Avoid two parallel Skill runtime models in the Agent Graph. Adapt the
  process-lifetime Skill Catalog into invocation-local discovery and activation
  state instead of continuing the current standalone `SkillRegistration`
  semantics with cached references.
- Startup logs include per-Tenant loaded and skipped counts for Specialist and
  Skill definitions. Definition-level errors include source identity and a
  stable reason suitable for Tenant administrator remediation.
- Accepted execution telemetry and durable Task data expose the Specialist
  Definition Pin and Activated Skill Pins, but not full instructions, reference
  contents, Tool-call history, or model messages.
- No database migration is required solely to preserve historical Markdown
  bodies. If the existing serialized Specialist Result or Task Outcome contract
  needs a new pin field, use the repository's existing backward-compatible
  checkpoint/state evolution rules and keep the field bounded.

## Testing Decisions

- Good tests assert behavior visible through module interfaces: loaded catalog
  membership, Coordinator-visible descriptors, accepted/rejected dispatch,
  bound Tool names, model-visible Skill activation results, pins, Task Outcomes,
  logs, and SSE completion. They do not assert private dictionaries, parser helper
  calls, storage implementation details, or the exact full prompt string.
- The primary seam is the existing Specialist Actor and Agent Graph contract.
  Run one shared contract suite against the code-defined and Markdown-defined
  Adapters. Given equivalent trusted definitions and deterministic actors, both
  must accept the same Specialist Task Input, preserve context, expose the same
  Tool set, produce the same Specialist Attempt shape, pass the same acceptance
  rules, and yield the same Specialist Result shape. Adapter-specific pin values
  may differ, but both must provide valid stable pins.
- The highest integration seam is the existing `/v2/query/stream` HTTP/SSE Agent
  path with the real Agent Graph, request Tenant headers, deterministic PydanticAI
  models, and PostgreSQL checkpointing. A local Tenant definition tree should
  prove startup loading, unchanged Intent Result handling, all-catalog Coordinator
  selection, Specialist dispatch, multiple interleaved Skill activations,
  scope-limited Tool execution, accepted result publication, and persisted pins.
- Extend the existing Agent-first Specialist integration suite rather than
  creating a parallel end-to-end harness. Reuse its deterministic Coordinator,
  Specialist, Tool, SSE, concurrency, retry, and checkpoint prior art.
- Extend the existing Specialist PydanticAI runtime tests to capture model
  messages with sentinels. Prove compact summaries appear before activation,
  multiple distinct Skills can activate in first-use order, a repeated activation
  is idempotent, business Tool calls may occur between activations, and platform
  instructions remain present ahead of Tenant instructions. Assert authority by
  available Tool bindings, not by trusting prompt wording.
- Replace the current one-Skill activation assertions in the Specialist Skill
  tests with multi-activation behavior. Cover zero Skills, one Skill, several
  Skills, the twenty-summary cap, unknown names, missing `required-tools`,
  optional `allowed-tools`, and Tools excluded by Research Scope. Do not add
  `max_activated_skills` tests because no such setting exists.
- The Local and GCS implementations justify one real Loader seam. Use a shared
  Adapter contract fixture to prove that both return equivalent Agent and Skill
  source documents for the same relative Tenant tree, isolate Tenant prefixes,
  read complete Agent and Skill definitions at startup, leave references unread,
  and return current reference contents on each call. Use an in-memory fake for
  GCS; deterministic tests must not require network access.
- Test the catalog through its public interface with mixed valid and invalid
  definitions. Verify invalid entries are absent, valid siblings remain, unknown
  model profiles and Tool names in either `required-tools` or `allowed-tools` are
  rejected, a Specialist with a missing Skill is rejected, instruction length is
  bounded, and logs identify the affected Tenant and source without echoing
  content.
- Test Research Scope resolution and Coordinator projection together: two
  Business Intents with different Tool/source policies must receive the same full
  Tenant Specialist Descriptor set while retaining their different non-Agent
  constraints. This is the regression test that prevents Intent Policy from
  becoming an implicit Specialist or Skill allowlist.
- Resume a checkpoint against a changed current catalog while preserving the
  original Tool, source, query, and freshness constraints. Verify that changed
  descriptions and definitions refresh, newly added Specialists are visible to
  the Coordinator, and removed Specialists disappear from discovery.
- Resume an accepted Batch containing two Tasks after removing one Task's
  Specialist Definition. The removed Specialist's legitimate Task must produce a
  failed Task Outcome without a model call, while the valid sibling runs and the
  barrier collects both outcomes. Separate cases must prove that a forged
  Specialist ID or damaged Batch manifest still fails invariant validation for
  the Batch rather than becoming an ordinary Task failure.
- Test Tool authority at the existing Specialist Registry/binding seam. A Tool
  must be callable only when globally registered, Tenant-permitted, and
  Research-Scope-permitted. Changing Specialist or Skill metadata must never add
  a Tool. Existing Evidence, expected-unavailability, Calculation, Data Gap, and
  telemetry tests remain authoritative for Tool-result semantics.
- Test live references by activating a Skill, reading its references, changing
  the Local Adapter's reference fixture without rebuilding the catalog, and
  reading again. The second result must contain the new content, while the Agent
  Definition Pin and Skill Pin remain unchanged. The shared Local/GCS Adapter
  contract must also prove that an empty directory succeeds while listing or
  reading failures return no partial documents and remain distinguishable from
  an empty reference set.
- Test Skill Pins with and without `metadata.version`. Both forms must validate,
  preserve first-activation order, and carry the content hash of the cached Skill
  Definition; changing definition content must change the hash even when version
  is absent.
- Test restart semantics with two independently built catalogs rather than a hot
  reload mechanism. The first catalog continues using its cached Agent and Skill
  definitions; a second startup sees edits. A retried Task in the second runtime
  records the second pin and does not request historical content.
- Run focused deterministic unit and integration suites during implementation,
  then the full suite. PostgreSQL-backed tests use the repository's standard
  PostgreSQL test runner and fixture-safety rules.

## Out of Scope

- Markdown definitions for the Coordinator Agent, Query Understanding Agent, or
  Synthesis Agent.
- Exact VS Code custom-agent compatibility or a promise that arbitrary VS Code
  Agent files run unchanged.
- Changes to the model-produced Intent Result or Intent Catalog item schema,
  including `allowed_specialist_ids`.
- Intent-specific Specialist or Skill allowlists.
- A Specialist-specific Tool allowlist or any Tool grant from Agent/Skill
  Markdown.
- A `max_activated_skills` setting, exactly-one-Skill behavior, or a
  `required-skills` dependency mechanism between Skills.
- Runtime detection, precedence, merging, or automatic recovery for conflicting
  Skill instructions.
- Hot reload for `.agent.md` or `SKILL.md` content.
- Caching, pinning, versioning, or historical retention of reference-file
  contents.
- Persisting full historical Specialist or Skill definitions for exact replay.
- Persisting or intersecting a historical Specialist ID snapshot for resumed
  Runs.
- Atomic all-or-nothing loading of an entire Tenant catalog.
- Executing Skill scripts, binaries, assets, or arbitrary code from Local or GCS
  definition storage.
- A Tenant administrator UI, publishing workflow, approval workflow, or eval
  authoring interface.
- Removal of the code-defined Specialist Adapter in the same change; removal is a
  later migration step after parity is demonstrated.
- Reworking the legacy FlowEngine request path or its Agent Handler beyond reuse
  of shared Agent Skills parsing and storage behavior needed by this feature.

## Further Notes

- This spec implements the accepted decision to borrow the useful authoring
  conventions of VS Code Agent Markdown and the open Agent Skills format without
  adopting either runtime wholesale.
- `description` is the routing contract for both Specialist and Skill discovery.
  Its quality is operationally important and should be covered by Tenant
  publishing review and evaluations, even though the runtime performs only
  structural validation.
- Definition pins provide observability, not exact replay. Agent and Skill
  definitions are stable only within one process, retries use the current process
  catalog, and references intentionally remain live.
- The design has three testable seams but only two runtime-facing interfaces: the
  catalog/Loader side and the existing Specialist Actor/Agent Graph side. Parsing,
  hashing, validation, and prompt projection should stay behind those interfaces
  so callers do not accumulate storage or Markdown knowledge.
