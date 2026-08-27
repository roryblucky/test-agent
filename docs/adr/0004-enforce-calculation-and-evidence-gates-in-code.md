# Enforce calculation and Evidence gates in deterministic code

Numerical results and source eligibility are enforced by deterministic platform code rather than LLM judgment: calculations produce reproducible Calculation Artifacts, and only Evidence satisfying the provenance contract may support factual or numerical claims. The POC uses an in-process executor limited to pre-registered functions, with a stable Tool contract that permits later replacement by an isolated sandbox.

POC calculation functions and financial providers may use simple deterministic fixtures or logging implementations because the first objective is the orchestration framework. Mock implementations must still obey the production-shaped typed contracts, provenance fields, method versions, units, time semantics, and failure gates; the LLM may select an allowed registered method but may not invent or execute formulas.
