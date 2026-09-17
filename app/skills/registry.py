"""Per-tenant Skill registry implementing agentskills.io progressive disclosure.

Three-tier progressive loading:

    Tier 1 — Discovery (startup)
        ``registry.discover(tenant_id)``
        Loads only name + description for all skills (~30-50 tokens/skill).
        Used to build the agent's capability index in system prompt.

    Tier 2 — Activation (on demand or pre-warm)
        ``registry.activate(tenant_id, skill_name)``
        Loads full SKILL.md instructions for a matched skill.
        Called when the agent determines it needs specific skills.

    Tier 3 — References (on demand during execution)
        ``registry.load_references(skill)``
        Downloads files from references/ directory for a specific skill.
        Called only when execution requires detailed reference context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.skills.schema import ReferenceDocument, SkillDefinition, SkillSummary

if TYPE_CHECKING:
    from app.skills.loader import SkillLoaderProtocol

logger = logging.getLogger(__name__)


class TenantSkillRegistry:
    """In-memory registry with three-tier progressive skill loading.

    Usage::

        registry = TenantSkillRegistry(loader)

        # Tier 1 — at startup (lightweight)
        await registry.discover("acme")
        summaries = registry.get_summaries("acme")

        # Tier 2 — when agent decides which skills to use
        skill = await registry.activate("acme", "vector-search")

        # Tier 3 — during execution if more context needed
        refs = await registry.load_references(skill)
    """

    def __init__(self, loader: SkillLoaderProtocol) -> None:
        self._loader = loader
        # Tier 1 cache: tenant_id → list of summaries
        self._summaries: dict[str, list[SkillSummary]] = {}
        # Tier 2 cache: (tenant_id, skill_name) → full SkillDefinition
        self._activated: dict[tuple[str, str], SkillDefinition] = {}
        # Resource listing cache: (tenant_id, skill_name) → list of filenames
        self._resource_files: dict[tuple[str, str], list[str]] = {}

    # ------------------------------------------------------------------
    # Tier 1 — Discovery
    # ------------------------------------------------------------------

    async def discover(self, tenant_id: str) -> None:
        """Load (or reload) Tier 1 summaries for a tenant.

        Only name + description are fetched. Safe to call at startup
        for all tenants, even with many skills.
        """
        summaries = await self._loader.discover_skills(tenant_id)
        self._summaries[tenant_id] = summaries
        logger.info(
            f"[{tenant_id}] Discovery complete: {len(summaries)} skill(s) → "
            f"{[s.name for s in summaries]}"
        )

    def get_summaries(self, tenant_id: str) -> list[SkillSummary]:
        """Return all Tier 1 summaries for a tenant (from cache)."""
        return self._summaries.get(tenant_id, [])

    def get_summary(self, tenant_id: str, skill_name: str) -> SkillSummary | None:
        """Return a single Tier 1 summary by name."""
        for s in self._summaries.get(tenant_id, []):
            if s.name == skill_name:
                return s
        return None

    def build_discovery_index(self, tenant_id: str) -> str:
        """Build the Tier 1 skill catalog in XML format for system prompt injection.

        Per agentskills.io spec: includes name, description, AND location
        (source_path) for each skill. Returns empty string if no skills.

        Format::

            <available_skills>
              <skill>
                <name>vector-search</name>
                <description>Search unstructured docs. Use when...</description>
                <location>gs://bucket/tenants/acme/skills/vector-search/SKILL.md</location>
              </skill>
            </available_skills>
        """
        summaries = self.get_summaries(tenant_id)
        if not summaries:
            return ""  # Spec: omit entirely if no skills available

        lines = ["<available_skills>"]
        for s in summaries:
            lines.append("  <skill>")
            lines.append(f"    <name>{s.name}</name>")
            lines.append(f"    <description>{s.description}</description>")
            lines.append(f"    <location>{s.source_path}</location>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    async def get_resource_files(
        self, tenant_id: str, skill_name: str
    ) -> list[str]:
        """List available reference filenames for a skill (Tier 2.5).

        Called during activation to populate <skill_resources> in the response.
        Results are cached. Files are NOT downloaded — just names listed.

        Returns:
            Sorted list of filenames in references/ (e.g. ['schema.md']).
        """
        cache_key = (tenant_id, skill_name)
        if cache_key in self._resource_files:
            return self._resource_files[cache_key]

        summary = self.get_summary(tenant_id, skill_name)
        if summary is None:
            return []

        try:
            filenames = await self._loader.list_resource_files(summary)
            self._resource_files[cache_key] = filenames
            return filenames
        except Exception:
            logger.exception(
                f"[{tenant_id}] Failed to list resource files for '{skill_name}'"
            )
            return []

    # ------------------------------------------------------------------
    # Tier 2 — Activation
    # ------------------------------------------------------------------

    async def activate(
        self, tenant_id: str, skill_name: str
    ) -> SkillDefinition | None:
        """Load full SKILL.md for a named skill (Tier 2 Activation).

        Checks the activation cache first. Downloads from GCS only on
        cache miss.

        Args:
            tenant_id: The tenant whose skill to activate.
            skill_name: Name of the skill to load.

        Returns:
            Fully loaded SkillDefinition, or None if not found / failed.
        """
        cache_key = (tenant_id, skill_name)
        if cache_key in self._activated:
            return self._activated[cache_key]

        # Warn if the skill was never discovered
        known = {s.name for s in self._summaries.get(tenant_id, [])}
        if skill_name not in known:
            logger.warning(
                f"[{tenant_id}] Skill '{skill_name}' not in discovery index. "
                "Run discover() first."
            )
            return None

        summary = self.get_summary(tenant_id, skill_name)
        if summary is None:
            logger.warning(
                f"[{tenant_id}] Cannot activate unknown skill: '{skill_name}'"
            )
            return None

        try:
            skill = await self._loader.activate_skill(summary)
            self._activated[cache_key] = skill
            logger.info(
                f"[{tenant_id}] Activated skill: '{skill_name}' "
                f"(tools: {skill.metadata.allowed_tools})"
            )
            return skill
        except Exception:
            logger.exception(
                f"[{tenant_id}] Failed to activate skill: '{skill_name}'"
            )
            return None

    def get_activated_skill(
        self, tenant_id: str, skill_name: str
    ) -> SkillDefinition | None:
        """Return a cached activated skill or None."""
        return self._activated.get((tenant_id, skill_name))

    # ------------------------------------------------------------------
    # Tier 3 — References
    # ------------------------------------------------------------------

    async def load_references(
        self, skill: SkillDefinition
    ) -> list[ReferenceDocument]:
        """Load reference documents for a skill (Tier 3 on demand).

        Downloads all files from the skill's ``references/`` directory.
        Attaches them to the skill object in-place for reuse.

        Args:
            skill: The activated SkillDefinition to load references for.

        Returns:
            List of ReferenceDocument objects.
        """
        if skill.references:
            # Already loaded (idempotent)
            return skill.references

        refs = await self._loader.load_references(skill)
        skill.references = refs  # Attach in-place for caching
        return refs

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_tool_names_for_skills(
        self, skills: list[SkillDefinition]
    ) -> set[str]:
        """Collect all unique allowed tool names from activated skills.

        Uses ``allowed-tools`` from the official agentskills.io spec.
        """
        tool_names: set[str] = set()
        for skill in skills:
            tool_names.update(skill.metadata.allowed_tools)
        return tool_names

    def get_required_tool_names_for_skills(
        self, skills: list[SkillDefinition]
    ) -> set[str]:
        """Collect all unique required tool names from activated skills."""
        tool_names: set[str] = set()
        for skill in skills:
            tool_names.update(skill.metadata.required_tools)
        return tool_names

    def invalidate(self, tenant_id: str) -> None:
        """Invalidate all cached data for a tenant (for hot-reload).

        Clears Tier 1 summaries, Tier 2 activation cache, and resource listing cache.
        """
        self._summaries.pop(tenant_id, None)
        for cache in (self._activated, self._resource_files):
            for k in [k for k in cache if k[0] == tenant_id]:
                del cache[k]
        logger.info(f"[{tenant_id}] Skill cache invalidated")


    @property
    def loaded_tenants(self) -> list[str]:
        """List all tenant IDs with loaded Tier 1 summaries."""
        return list(self._summaries)
