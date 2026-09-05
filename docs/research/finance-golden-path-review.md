# Finance golden-path review

Date: 2026-09-03

## Question

Does this proposed finance scenario genuinely demonstrate the four core patterns
of the bounded Coordinator–Specialist architecture?

> Analyze fund F over the last year and explain its main risks by comparing it
> with benchmark B and examining its top holdings and related news.

The proposed execution shape was:

1. Round 1 fans out price-series retrieval, holdings retrieval, and report
   retrieval.
2. Round 2 calculates metrics from the price series and searches for news based
   on the holdings.
3. The Coordinator finishes and the Synthesis Agent writes the report.

## Conclusion

The **business query is a good integrated golden scenario**, but the proposed
execution shape is not fully consistent with the current design:

1. A deterministic calculation should not be a separate top-level Task. The
   responsible market-analysis Specialist should retrieve the series and call
   registered calculation Tools inside its own autonomous run.
2. The scenario demonstrates an outcome-dependent second Coordination Round,
   but one successful path alone does not prove bounded replanning. A focused
   alternate-outcome test is also needed.
3. The offered choice between one integrated scenario and several small
   scenarios is a false dichotomy. Use one integrated E2E golden test **plus**
   focused invariant tests.

This is an inference from the primary-source primitives below. None of the
frameworks prescribes this exact finance example.

## Evidence from primary sources

### Dynamic fan-out and fan-in

LangGraph's official orchestrator-worker example uses `Send` to create dynamic
workers. Each worker gets its own state, while worker outputs are accumulated in
a reducer-backed shared field before the downstream synthesizer runs. This is
the direct framework primitive for the proposed first-round fan-out and batch
barrier
([LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents),
[source of the documented example](https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/workflows-agents.mdx)).

The framework supplies aggregation, but stable deterministic ordering remains
an application invariant. A list-append reducer only accumulates results; this
platform should continue to merge by stable Task ID and sort canonically before
the Coordinator consumes a batch.

### Code orchestration around autonomous Agents

LangGraph distinguishes workflows with code-owned paths from Agents that choose
their own processes and Tool usage. Its documented orchestrator-worker pattern
lets the orchestrator break work into subtasks, delegate them, and synthesize
their outputs. This matches a code-owned batch loop whose Specialist nodes each
run an autonomous PydanticAI Agent
([LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)).

PydanticAI defines an Agent as a model combined with instructions, Tools,
structured output, dependencies, and model settings. Its multi-agent guide
separates Agent delegation from programmatic hand-off and graph-based control
flow, and notes that delegated Agents can use different models and dependencies
([PydanticAI Agent](https://github.com/pydantic/pydantic-ai/blob/main/docs/agent.md),
[PydanticAI multi-agent applications](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md)).

OpenAI's Agents SDK independently separates LLM-owned choices from code-owned
orchestration. Its official guidance lists chaining outputs into later inputs,
bounded evaluator loops, and parallel execution of independent Agents as common
code-orchestration patterns
([OpenAI Agents SDK orchestration](https://openai.github.io/openai-agents-python/multi_agent/)).

Together, these sources support the project's boundary: LangGraph controls when
an assignment runs; a Specialist Agent controls how to satisfy its business
objective with its eligible Tools.

### Why calculation belongs inside the market Specialist

The proposed `price retrieval -> later calculation Task` split models a Tool
operation as though it were another delegated business actor. That would weaken
the already-confirmed contract that a top-level Task is a business Assignment
to a Specialist, not a record of each Tool call.

PydanticAI's Agent abstraction explicitly puts function Tools inside the Agent
run. Its delegation example also shows a delegated Agent making its own Tool
calls before returning control
([PydanticAI Agent](https://github.com/pydantic/pydantic-ai/blob/main/docs/agent.md),
[PydanticAI multi-agent applications](https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md)).
OpenAI's official orchestration guidance likewise describes an Agent as an LLM
equipped with Tools that autonomously plans how to acquire data and complete its
task
([OpenAI Agents SDK orchestration](https://openai.github.io/openai-agents-python/multi_agent/)).

Keeping price retrieval and registered calculations in one Specialist run also
avoids creating a new cross-Specialist contract for transporting raw price
series. The Specialist can instead return bounded findings, Evidence IDs, and
validated Calculation Artifacts.

## Minimal corrected integrated scenario

Use an explicit fixture date and synthetic identifiers so the golden test is
repeatable:

> As of 2026-06-30, analyze `FUND_F` over the preceding year. Compare it with
> `BENCHMARK_B` and explain the major risks using the fund's reported risks, top
> holdings, and material news about those holdings.

```text
Coordinator round 1: DispatchBatch
├── Market-analysis Specialist
│   objective: quantify performance and risk versus BENCHMARK_B
│   internal autonomous Tools:
│     price_series -> registered performance/risk calculations
├── Holdings Specialist
│   objective: identify top holdings and concentration risks
│   internal autonomous Tools:
│     fund_holdings
└── Fund-report Specialist
    objective: extract material risks disclosed by the fund
    internal autonomous Tools:
      fund_reports

deterministic batch barrier
    merge by Task ID; canonical sort

Coordinator round 2: DispatchBatch
└── News Specialist
    objective: investigate material news for the actual top holdings
    dependency: selected Holdings Specialist result from round 1
    internal autonomous Tools:
      company_news

deterministic batch barrier

Coordinator round 3: Finish
└── Synthesis Agent
    input: query + accepted Specialist Results + eligible Evidence
```

This shape demonstrates:

- **Fan-out:** the three independent Round 1 Assignments execute concurrently.
- **Deterministic fan-in:** all three outcomes cross one barrier and are merged
  and ordered by code before the next Coordinator decision.
- **Multi-hop dependency:** the news Assignment is created from the actual
  holdings finding and explicitly references that prior result.
- **Specialist delegation:** the Coordinator chooses a Specialist and business
  objective; each Specialist chooses its own eligible Tools internally.
- **Bounded iteration:** the Coordinator is reinvoked after each completed batch
  and the Agent Graph enforces the configured round bound.

The third initial Specialist is not essential to the graph pattern, but it is
useful to the business answer and demonstrates a non-dependent evidence branch
that still joins at synthesis. If minimal runtime code is more important than
coverage of the intended finance answer, the report branch can be removed from
the first implementation.

## What the golden test must assert

Do not use a live model to establish the control-flow contract. Script the
Coordinator and Specialist model responses, while executing the real graph,
validation, Tool adapters, reducers, and Synthesis boundary. OpenAI's current
testing guide follows this boundary: scripted models test application-owned Tool
execution, orchestration, retries, and workflow shape; real providers are
reserved for model/provider behavior
([OpenAI Agents SDK testing](https://openai.github.io/openai-agents-python/testing/)).

The integrated test should assert:

1. all Round 1 branches start before any is released by a controlled async test
   barrier;
2. the Coordinator's second call happens only after all Round 1 outcomes finish;
3. deliberately reversed completion order still produces canonical Task order;
4. the Round 2 Assignment references the Holdings result and names or otherwise
   carries the actual holdings selected from the fixture;
5. the market Specialist, rather than the Coordinator, calls price and
   registered calculation Tools;
6. the news Specialist, rather than the Coordinator, calls the news Tool;
7. `Finish` leads to one Synthesis invocation with only accepted Results and
   eligible Evidence;
8. the run ends within the configured round limit.

LangGraph's official testing guide also recommends compiling a fresh graph with
a new in-memory checkpointer for each test and separately invoking individual
nodes when testing node-level behavior
([LangGraph testing](https://docs.langchain.com/oss/python/langgraph/test)).

## Why the original A/B choices were incomplete

- **A: one composite E2E scenario** is necessary to prove that the seams work
  together, but it is insufficient for validation failures, limit enforcement,
  retry mapping, and alternate Coordinator decisions. A large failure only says
  the whole flow broke.
- **B: several scenarios, each proving one pattern** localizes failures, but it
  does not prove that fan-out, fan-in, cross-round dependencies, delegation, and
  synthesis compose in one request.

The missing and recommended option is:

> **C: one integrated finance golden path plus small focused invariant tests.**

At minimum, add one alternate-outcome test in which the holdings Task returns no
eligible finding or fails terminally. The next Coordinator decision must differ
from the successful path and still terminate within the round bound. This is
what demonstrates outcome-driven replanning rather than a merely scripted
three-stage workflow. It need not introduce a new production abstraction.

