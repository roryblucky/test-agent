# Minimal synthesis input: independent review

Date: 2026-09-03

## Question

Should the POC add a separate `known_gaps` field to the Synthesis Agent input,
and if so, which component can produce it correctly?

## Conclusion

No. The current POC has no deterministic owner capable of deriving a complete
or materially correct global `known_gaps` list. Adding the field now duplicates
the existing Specialist Result `gaps` and creates an unstated coverage model.

Use the existing research content directly:

```python
class SynthesisInput(BaseModel):
    query: str
    findings: tuple[SpecialistResult, ...]
    evidence: tuple[EvidenceExcerpt, ...]
```

The Synthesis Agent may describe limitations that are explicitly present in
the accepted findings. The Graph can also include those accepted bounded gap
strings in a deterministic insufficient-Evidence answer when Synthesis is
skipped. Neither behavior claims a complete global coverage model.

## Why no component can currently derive `known_gaps`

- A Specialist Agent can report a gap relative to its own Assignment. It cannot
  know whether another Specialist covered that gap or whether it is material to
  the original query.
- The Coordinator Agent sees the cross-Assignment picture and may judge whether
  more research is needed. That is an LLM judgment, not deterministic
  derivation. Making it emit another gap list duplicates its finish decision.
- The Agent Graph can collect, order, and deduplicate explicitly emitted gap
  strings. It cannot infer semantic coverage or materiality without a
  predefined coverage/acceptance contract, which the POC intentionally does not
  have.
- The Synthesis Agent can discuss limitations from the supplied material, but
  it cannot author a field that is then presented as deterministic input to
  itself.

If a future domain introduces explicit required coverage keys, deterministic
code can calculate missing keys. That future type would be a domain-specific
coverage result, not a generic free-text `known_gaps` field.

## What primary implementations actually pass

OpenAI's Agents SDK `research_bot` passes the original query and successful
search summaries to its Writer. Failed searches are counted for progress but
are omitted from Writer input; there is no global gap object
([manager source](https://github.com/openai/openai-agents-python/blob/main/examples/research_bot/manager.py#L59-L103)).

Open Deep Research passes `research_brief`, conversation messages, and joined
findings to final report generation. The same path is used after a supervisor
completion signal, no tool call, or iteration exhaustion, and there is no
separate completeness/gap model
([termination source](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L216-L253),
[final report source](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/deep_researcher.py#L581-L626)).

STORM's section writer receives topic, outline/section, and retrieved
information snippets. It does not receive orchestration completion state or a
global gap list
([article generation source](https://github.com/stanford-oval/storm/blob/main/knowledge_storm/storm_wiki/modules/article_generation.py#L29-L47),
[writer signature](https://github.com/stanford-oval/storm/blob/main/knowledge_storm/storm_wiki/modules/article_generation.py#L149-L161)).

These implementations do not prove that omissions are always safe. They do show
that a separate generic `known_gaps` contract is not required to implement the
core research-to-writing handoff.

## Minimal POC behavior

1. Pass the query, accepted Specialist Results, and eligible Evidence excerpts
   to Synthesis.
2. Instruct Synthesis to make only evidence-supported claims and to preserve
   limitations explicitly stated by Specialists.
3. Keep Task failures, retries, budget state, and Coordinator decisions outside
   the Synthesis prompt.
4. If there is no eligible Evidence, return the existing deterministic
   insufficient-evidence outcome, including accepted bounded Specialist gaps,
   instead of asking the model to invent a report.
5. Apply the already-required citation and Evidence publication gates.

The POC does not need `report_scope`, `known_gaps`, a coverage engine, or a
deterministic projection that tries to turn operational failures into business
meaning.

## Why the earlier proposal was over-designed

The proposal began with a valid risk: a Writer might make incomplete research
sound complete. It then introduced new product concepts to solve that risk
before the POC had demonstrated it:

- a new `SynthesisBrief` type despite an existing Specialist Result boundary;
- `report_scope` without a formal definition of complete coverage;
- global `known_gaps` without an authoritative producer;
- a deterministic projection whose semantic decision cannot actually be
  deterministic;
- special partial-report behavior before the basic E2E path exists.

This mixed three concerns—runtime termination, research completeness, and
report writing—into a new contract. The simpler rule is to reuse accepted
research content, keep runtime state in the Agent Graph, and add a coverage
contract only when a concrete business scenario supplies measurable coverage
requirements.
