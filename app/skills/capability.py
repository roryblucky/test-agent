"""AgentSkills Capability — agentskills.io implemented as a Pydantic AI AbstractCapability.

pydantic-ai version: 1.93.0+

All three tiers of the agentskills.io spec map to AbstractCapability hooks:

  Tier 1 Discovery   →  ``get_instructions()`` (callable form for dynamic catalog)
  Tier 2 Activation  →  ``get_toolset()`` registers ``activate_skill`` tool
  Tier 3 References  →  ``get_toolset()`` registers ``load_skill_references`` tool
  Per-run state      →  lives in ``AgentDeps`` (created fresh per request by AgentHandler)

Usage::

    from app.skills.capability import SkillsCapability

    capability = SkillsCapability(registry=skill_registry, tenant_id="acme")
    agent = Agent(model=..., capabilities=[capability, mcp_cap, ...])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from app.agents.tools import activate_skill_tool, load_skill_references_tool

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.toolsets import AbstractToolset

    from app.agents.agent_deps import AgentDeps
    from app.skills.registry import TenantSkillRegistry

# ------------------------------------------------------------------
# Activation instructions injected alongside the discovery catalog
# ------------------------------------------------------------------
_ACTIVATION_INSTRUCTIONS = """\
When a task matches a skill's description, call activate_skill with the \
skill's name to load its full instructions. Follow those instructions before \
calling any domain tools. If the instructions mention reference materials, \
call load_skill_references to load them.\
"""


@dataclass
class SkillsCapability(AbstractCapability["AgentDeps"]):
    """Pydantic AI capability implementing the agentskills.io progressive disclosure model.

    Encapsulates all three tiers as a single, composable capability that can
    be passed to ``Agent(capabilities=[...])``.

    Tier 1 — Discovery (``get_instructions``)
        Returns a callable so the XML skill catalog is built fresh each request
        from the live registry state (handles hot-reload / late discovery).

    Tier 2 — Activation (``get_toolset`` → ``activate_skill`` tool)
        The LLM calls this tool at runtime to load a skill's full SKILL.md.
        Returns ``<skill_content>`` wrapped instructions + resource listing.

    Tier 3 — References (``get_toolset`` → ``load_skill_references`` tool)
        The LLM calls this tool to load reference documents on demand.

    Per-run state lives in ``AgentDeps``, created fresh per request by
    ``AgentHandler.handle()``. This capability is stateless and shared
    across all runs of the same agent.

    Attributes:
        registry: ``TenantSkillRegistry`` holding summaries + activation cache.
        tenant_id: Tenant whose skills this capability exposes.
    """

    registry: TenantSkillRegistry
    tenant_id: str

    # ------------------------------------------------------------------
    # AbstractCapability — static configuration (called at Agent build time)
    # ------------------------------------------------------------------

    def get_instructions(self):
        """Tier 1: Return a callable that yields the XML skill catalog per request.

        Using a callable (not a plain string) ensures the catalog reflects the
        current registry state at request time — important when ``discover()``
        is called after ``Agent`` construction or when skills are hot-reloaded.

        Spec: "If no skills are available, omit the catalog entirely."
        """
        registry = self.registry
        tenant_id = self.tenant_id

        async def _build_catalog(ctx: RunContext[AgentDeps]) -> str | None:
            catalog = registry.build_discovery_index(tenant_id)
            if not catalog:
                return None  # Spec: no empty catalog block
            return (
                f"<skills>\n"
                f"{catalog}\n\n"
                f"{_ACTIVATION_INSTRUCTIONS}\n"
                f"</skills>"
            )

        return _build_catalog

    def get_toolset(self) -> FunctionToolset["AgentDeps"] | None:
        """Tier 2 + 3: Register ``activate_skill`` and ``load_skill_references``.

        Spec: "If no skills are available, don't register the tool at all."
        We check the registry at build time; if summaries are empty (discover
        hasn't been called yet), return None — the capability's ``for_run``
        will be called before any tool use, so this is safe.

        The toolset carries no ``instructions`` here — the discovery catalog
        is handled by ``get_instructions`` above to keep separation clean.
        """
        summaries = self.registry.get_summaries(self.tenant_id)
        if not summaries:
            logger.debug(
                f"[{self.tenant_id}] SkillsCapability.get_toolset: no summaries yet, "
                "returning None. Ensure registry.discover() is called before Agent use."
            )
            return None

        return FunctionToolset([activate_skill_tool, load_skill_references_tool])

    # ------------------------------------------------------------------
    # AbstractCapability — per-run lifecycle
    # ------------------------------------------------------------------

    async def for_run(self, ctx: RunContext["AgentDeps"]) -> SkillsCapability:
        """No per-run state to manage — AgentDeps is already fresh per request.

        ``AgentHandler.handle()`` constructs a new ``AgentDeps`` for every
        ``agent.run_stream()`` call, setting ``skill_registry``, ``tenant_id``,
        and ``activated_skill_names=[]`` upfront. Nothing to do here.
        """
        return self
