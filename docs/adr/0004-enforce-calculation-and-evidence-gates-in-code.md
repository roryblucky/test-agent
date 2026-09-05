# Enforce calculation and Evidence gates in deterministic code

Numerical results and source eligibility are enforced by deterministic platform code rather than LLM judgment: calculations over trusted Tool-resolved data produce reproducible Calculation Artifacts, and only Evidence satisfying the provenance contract may support factual or numerical claims. The POC uses an in-process executor limited to pre-registered functions, with a stable Tool contract that permits later replacement by an isolated sandbox. An LLM may select trusted input references but may not supply an authoritative raw numerical series.

POC calculation functions and financial providers may use simple deterministic fixtures or logging implementations because the first objective is the orchestration framework. Mock implementations must still obey the production-shaped typed contracts, provenance fields, method versions, units, time semantics, and failure gates; the LLM may select an allowed registered method but may not invent or execute formulas.

The final Research Report remains coherent model-authored Markdown with inline
Evidence markers and Calculation Artifact placeholders. For every registered
calculation used in the report, Synthesis selects the placeholder while code
renders the Artifact's canonical formatted value and concise disclosure;
Synthesis does not retype the value. Deterministic publication gates
validate marker syntax, identity, Tenant/Run eligibility, freshness, and
Calculation Artifact integrity before binding public citations. They do not
claim to identify every factual sentence or prove that cited Evidence
semantically entails adjacent prose; citation coverage and groundedness remain
advisory, as does detection of arbitrary unmarked numerical prose. A claim-level canonical output was rejected for this POC because it
would either duplicate the Markdown as a second factual source or constrain the
report to a deterministic claim renderer.

For the POC, a calculation Tool returns a concise model-visible value through
PydanticAI `ToolReturn.return_value` and carries its typed Calculation Artifact
in application-only `ToolReturn.metadata`. The LangGraph Specialist adapter
stages that metadata and the terminal Task Outcome in one branch contribution
only after the Specialist Result validates. The batch barrier validates every
attempt and support reference and promotes the entire batch, including complete
bounded Calculation Artifact records, in one state update. Failed or abandoned
outer attempts and rejected batches contribute no canonical Artifacts. This
state-update boundary is not an external distributed transaction, and no generic
Artifact store or Result payload framework is introduced.

Complete Calculation Artifacts remain internal. A validated calculation marker
is rendered as a concise disclosure of the human-readable method, unit, period
or as-of time, and material assumptions; method versions, hashes, complete
inputs, and execution details remain audit data. Calculation Artifacts are not
encoded as source citations, and the public response is not extended with an
artifact array before an external machine consumer exists. Agent-mode Synthesis
publishes no answer text until these blocking gates pass; it streams progress
during generation and then emits the approved Markdown through the existing SSE
event shapes. Linear mode retains its existing live-token behavior because its
post-answer assessments are advisory rather than blocking gates.

Synthesis receives deterministic bounded views of both eligible Evidence and
Calculation Artifacts. The Calculation catalog exposes aliases and interpretation
metadata but not the canonical value; the model selects a placeholder and code
renders the value. One frozen prepared value owns the alias maps and remains
unchanged across repair. The POC does not add a digest because it has no
active-Run cross-process recovery or external digest consumer.
