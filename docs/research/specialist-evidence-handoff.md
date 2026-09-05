# Specialist evidence handoff in open-source research agents

Date: 2026-09-03

## Question

When a research Specialist finishes a delegated task, should it return:

1. claim-level items such as `FindingItem(statement, evidence_ids)`;
2. a summary plus one global source list;
3. prose with inline citations;
4. raw source excerpts; or
5. uncited prose and let the final writer insert citations later?

The relevant product is a read-only, evidence-backed information platform. The
handoff must remain small enough for repeated Coordinator rounds while retaining
enough provenance for final synthesis and publication checks.

## Conclusion

There is no single cross-framework evidence-handoff standard. The most common
open-source research convention is **bounded prose with inline citations plus a
separate source registry/list**. Generic multi-agent frameworks usually return
plain text and leave stronger evidence contracts to the application.

The strongest implementations preserve two distinct things:

- source records captured from tools, including URL/title and often retrieved
  snippets; and
- model-authored claims or prose that reference only those source records.

They can deterministically verify reference integrity—whether a citation is
well-formed and resolves to a source that was actually retrieved. They generally
cannot deterministically verify semantic entailment—whether that source really
supports the attached claim. That remains an LLM/evaluation problem unless the
application adds domain-specific deterministic checks.

For this platform, the recommended POC contract is a small claim-level envelope:

```python
class FindingItem(BaseModel):
    statement: str
    evidence_ids: tuple[EvidenceId, ...]  # at least one


class SpecialistFinding(BaseModel):
    summary: str
    findings: tuple[FindingItem, ...]
    gaps: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
```

This is stricter than the most common community implementation, but it applies
the community's strongest provenance pattern at the Agent boundary. Evidence
records and raw excerpts remain in the platform evidence store/request-local
registry; the Specialist and Coordinator exchange stable IDs, not repeated raw
tool output.

An expected unavailable read/fetch/Calculation call should be returned to the
active Specialist as a `ToolReturn` containing a bounded typed unavailable
value, not raised to terminate the
PydanticAI run. This lets the Agent try an allowed fallback in a multi-hop path
or retain successful siblings in internal fan-out. The terminal
`SpecialistFinding.gaps` records only sanitized missing business coverage; it
does not carry raw Tool errors or retry diagnostics.

## What current primary sources implement

### Open Deep Research: cited prose handed upward

LangChain's current Open Deep Research executes researcher subgraphs and hands
each result to the supervisor as the text value of `compressed_research`. It
also retains concatenated raw tool/AI messages under `raw_notes`, but the
supervisor-facing `ToolMessage` contains the compressed result rather than a
typed list of claims
([researcher and supervisor source](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py)).

The compression prompt requires inline citations and a source list, then says a
later LLM will merge these findings. The final-report prompt receives all
compressed findings as text and again instructs the final writer to emit source
references
([compression and final-report prompts](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/prompts.py)).

This is therefore:

```text
researcher tools
  -> raw messages
  -> LLM-compressed cited prose + source list
  -> supervisor
  -> LLM final writer
```

It is not a `FindingItem` schema. Citation placement and preservation across
compression/final writing are prompt-enforced. The inspected execution path
does not perform an entailment check between a sentence and its cited URL.

### Deep Agents: text by default, optional structured summary plus sources

Deep Agents explicitly treats the subagent boundary as context quarantine: the
parent sees the subagent's final result, not its intermediate tool calls. By
default that result is free-form text. An optional `response_format` can instead
return JSON, and the first-party example uses `ResearchFindings(summary,
confidence, sources)`—a summary with a global URL list, not claim-level evidence
links
([Deep Agents subagent documentation](https://docs.langchain.com/oss/python/deepagents/subagents)).

Its deep-research tutorial asks each subagent to research one aspect and return
findings, and directs the final report to include a numbered Sources section
([Deep Agents deep-research tutorial](https://docs.langchain.com/oss/python/deepagents/deep-research)).

This confirms that structured output is an optional application boundary, not a
built-in citation truth mechanism. Schema validation can guarantee the presence
and type of `sources`; it cannot guarantee that a source supports `summary`.

### GPT Researcher: a claim-like learning paired with one source URL

GPT Researcher's recursive deep-research example asks an LLM to return JSON with
`learnings: [{insight, sourceUrl}]`. It accumulates a mapping from learning text
to URL, then serializes each learning as `learning [Source: URL]` for the final
report writer
([deep-research implementation](https://github.com/assafelovic/gpt-researcher/blob/main/backend/report_type/deep_research/example.py)).

This is the closest inspected implementation to
`FindingItem(statement, evidence_ids)`, although the identity is a URL rather
than an internal evidence ID and the example associates one URL with a learning.
The model chooses the `sourceUrl`; JSON parsing checks structure, but source
membership and semantic support are not established merely by that schema.

The broader project describes its pipeline as summarizing and source-tracking
resources, then filtering and aggregating those summaries into the final report
([GPT Researcher repository](https://github.com/assafelovic/gpt-researcher)).

### STORM: source/snippet registry plus inline numeric citations

STORM keeps collected evidence separately from generated prose. A
`DialogueTurn` stores the agent utterance, search queries, and structured search
results. `StormInformationTable` deduplicates these results by URL, preserves
their snippets, and retrieves relevant snippets for article sections
([STORM information data classes](https://github.com/stanford-oval/storm/blob/main/knowledge_storm/storm_wiki/modules/storm_dataclass.py)).

During article construction, generated section text uses numeric inline
citations. Application code:

- parses citation numbers from the generated section;
- removes out-of-range citation numbers;
- retains only cited source records;
- deduplicates sources by URL; and
- rewrites local citation numbers into a unified article-wide index.

Those operations are visible in `StormArticle.update_section` and reference
management
([STORM citation post-processing](https://github.com/stanford-oval/storm/blob/main/knowledge_storm/storm_wiki/modules/storm_dataclass.py#L404-L462)).
The UI later converts citation numbers to links and shows the stored source
snippets as highlights
([STORM demo citation rendering](https://github.com/stanford-oval/storm/blob/main/frontend/demo_light/demo_util.py)).

STORM demonstrates the strongest deterministic citation hygiene among the
inspected systems: citation existence, bounds, deduplication, and stable
renumbering. It still does not prove that the cited snippet entails the adjacent
sentence. The repository has an open report of inconsistent/invalid citation
output, illustrating the remaining gap between syntactic reference repair and
semantic citation correctness
([STORM issue #168](https://github.com/stanford-oval/storm/issues/168)).

### AutoGen/Magentic-One: generic messages, no research evidence contract

Magentic-One maintains an LLM-authored facts ledger and plan, consumes the team
message thread, and generates its final answer as a `TextMessage`. Its structured
progress ledger governs completion, stalling, instructions, and next-speaker
selection—not evidence-to-claim provenance
([Magentic-One orchestrator](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py)).

AutoGen provides a generic `StructuredMessage` whose content can be any Pydantic
model, so an application can define a claim/evidence handoff, but the framework
does not prescribe one
([AutoGen message types](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/messages.py)).

This is useful negative evidence: a generic multi-agent protocol is not itself
an evidence model.

## Comparison

| Pattern | Community prevalence | Coordinator context cost | Deterministic checks | Main weakness |
|---|---:|---:|---|---|
| Claim-level `statement + evidence IDs` | Uncommon as a default; GPT Researcher's learning/URL pair is close | Low to medium | Shape, non-empty IDs, ID existence, authorization, deduplication | Semantic support is still model-authored |
| Summary + global source list | Common in generic structured research results | Low | Shape and source existence | Cannot tell which source supports which assertion |
| Prose with inline citations + source registry | Most common in research/report systems | Medium | Syntax, bounds, resolvable IDs, stable rendering | Parsing prose is brittle; citation entailment remains uncertain |
| Raw source excerpts in handoff | Usually retained separately, not returned wholesale to parent | High | Hash/source provenance and exact excerpt identity | Context bloat; excerpt selection can still be wrong |
| Citation insertion only at final stage | Used as a final formatting/synthesis pass, usually after earlier source tracking | Medium | Link resolution after generation | Highest risk of misattachment or dropped provenance if upstream findings were uncited |

## What can actually be validated deterministically

The platform can reject a Specialist result before storing it when:

- an `evidence_id` does not exist in the current Run's evidence registry;
- the current Specialist did not observe or is not authorized to reference that
  evidence;
- an evidence list is empty for a publishable factual finding;
- an ID is duplicated or malformed;
- source metadata required by the evidence type is absent;
- an inline marker references an unknown/out-of-range source;
- cited evidence is dropped during canonical collection; or
- a result exceeds bounded item/text/source limits.

It cannot generally determine with pure code whether:

- the evidence semantically entails the statement;
- the statement omitted a material qualifier;
- two sources are genuinely independent;
- the source is authoritative enough for the domain; or
- a synthesis introduced a new factual claim not explicitly present upstream.

Those need a source-quality policy, an LLM citation/entailment reviewer, a human
reviewer, or domain-specific code. An LLM gate can improve quality but should be
described as probabilistic validation, not deterministic validation.

## Recommendation for this platform

Adopt claim-level `FindingItem` for **key factual findings**, while keeping a
short free-text `summary` for Coordinator comprehension. This gives the platform
a useful machine-readable provenance graph without forcing every sentence or
every internal thought into a schema.

The execution adapter should perform only deterministic integrity checks:

```text
Specialist output
  -> Pydantic shape validation
  -> every evidence_id resolves in this Run
  -> every evidence_id was available to this Specialist
  -> stable deduplication/canonical ordering
  -> accepted SpecialistFinding
```

The Synthesis Agent should receive the bounded findings and a bounded
representation of their referenced evidence. It must cite only accepted
evidence IDs. A publication gate can then reject unknown/missing references and
uncited factual output, but semantic support must remain explicitly
probabilistic unless a particular Tool or domain schema makes it mechanically
checkable.

Avoid two extremes in the POC:

- Do not return only `summary + global evidence_ids`; it loses the relationship
  needed for evidence-aware replanning and final synthesis.
- Do not embed all raw excerpts in every `FindingItem`; keep raw evidence in its
  registry and resolve only bounded excerpts when a downstream Agent needs them.

This contract does not require every Specialist to emit domain-specific typed
business payloads. An optional typed payload remains a separate concern for
machine-consumed multi-hop or deterministic calculation.
