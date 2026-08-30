"""Tests for LayeredPromptBuilder 5-layer architecture."""

from __future__ import annotations

from app.config.models import DomainConfig, TenantOutputConfig
from app.prompts.builder import LayeredPromptBuilder


class TestBuildLayers:
    """Test that all 5 layers are assembled correctly."""

    def test_default_identity_and_guardrails(self):
        """Minimum prompt has identity + guardrails."""
        prompt = LayeredPromptBuilder.build()
        assert "<identity>" in prompt
        assert "enterprise knowledge assistant" in prompt
        assert "<guardrails>" in prompt
        assert "NEVER fabricate" in prompt

    def test_custom_identity(self):
        prompt = LayeredPromptBuilder.build(identity="You are ACME bot.")
        assert "ACME bot" in prompt
        assert "enterprise knowledge assistant" not in prompt

    def test_tenant_contract_layer(self):
        prompt = LayeredPromptBuilder.build(
            tenant_contract="Always respond in formal English."
        )
        assert "<tenant_contract>" in prompt
        assert "formal English" in prompt

    def test_domain_contract_layer(self):
        prompt = LayeredPromptBuilder.build(
            domain_contract="All investment advice must cite approved sources."
        )
        assert "<domain_contract>" in prompt
        assert "approved sources" in prompt

    def test_context_layer(self):
        prompt = LayeredPromptBuilder.build(context={"current_date": "2026-05-17"})
        assert "<context>" in prompt
        assert "2026-05-17" in prompt

    def test_extra_instructions_layer(self):
        prompt = LayeredPromptBuilder.build(
            extra_instructions="You are a planner agent."
        )
        assert "<instructions>" in prompt
        assert "planner agent" in prompt

    def test_all_layers_in_order(self):
        """All layers appear in correct order per design spec."""
        prompt = LayeredPromptBuilder.build(
            identity="Custom identity.",
            tenant_contract="Tenant rules.",
            domain_contract="Domain rules.",
            extra_instructions="Node contract.",
            reference_data="Retrieved documents here.",
            context={"current_date": "2026-01-01"},
        )
        # Check order: Identity > Guardrails > Tenant > Domain > Instructions > Reference > Context
        identity_pos = prompt.index("<identity>")
        guardrails_pos = prompt.index("<guardrails>")
        tenant_pos = prompt.index("<tenant_contract>")
        domain_pos = prompt.index("<domain_contract>")
        instructions_pos = prompt.index("<instructions>")
        reference_pos = prompt.index("<reference_data>")
        context_pos = prompt.index("<context>")

        assert identity_pos < guardrails_pos
        assert guardrails_pos < tenant_pos
        assert tenant_pos < domain_pos
        assert domain_pos < instructions_pos
        assert instructions_pos < reference_pos
        assert reference_pos < context_pos

    def test_reference_data_layer(self):
        """Reference data gets its own semantic tag, separate from instructions."""
        prompt = LayeredPromptBuilder.build(
            reference_data="[Document 1]\nContent here."
        )
        assert "<reference_data>" in prompt
        assert "Content here" in prompt
        # Should NOT be in instructions tag
        assert "<instructions>" not in prompt

    def test_optional_layers_omitted(self):
        """Missing optional layers produce no empty tags."""
        prompt = LayeredPromptBuilder.build()
        assert "<tenant_contract>" not in prompt
        assert "<domain_contract>" not in prompt
        assert "<context>" not in prompt
        assert "<instructions>" not in prompt
        assert "<reference_data>" not in prompt


class TestBuildFromConfig:
    """Test build_from_config with TenantConfig-like objects."""

    def test_with_output_config(self):
        """TenantOutputConfig is extracted into tenant_contract layer."""

        class FakeTenantConfig:
            output_config = TenantOutputConfig(
                default_format="markdown",
                disclaimer="Past performance is not indicative of future results.",
                forbidden_phrases=["guarantee", "risk-free"],
            )
            domain_config = None

        prompt = LayeredPromptBuilder.build_from_config(
            tenant_config=FakeTenantConfig(),
        )
        assert "<tenant_contract>" in prompt
        assert "Past performance" in prompt
        assert "guarantee" in prompt

    def test_with_domain_config(self):
        """DomainConfig is extracted into domain_contract layer."""

        class FakeTenantConfig:
            output_config = None
            domain_config = DomainConfig(
                name="wealth",
                allow_model_common_knowledge=False,
                locale="en-US",
            )

        prompt = LayeredPromptBuilder.build_from_config(
            tenant_config=FakeTenantConfig(),
        )
        assert "<domain_contract>" in prompt
        assert "Domain: wealth" in prompt
        assert "Do NOT use model common knowledge" in prompt
        assert "Locale: en-US" in prompt

    def test_with_both_configs(self):
        """Both configs produce both contract layers."""

        class FakeTenantConfig:
            output_config = TenantOutputConfig(disclaimer="Disclaimer text.")
            domain_config = DomainConfig(name="legal")

        prompt = LayeredPromptBuilder.build_from_config(
            tenant_config=FakeTenantConfig(),
        )
        assert "<tenant_contract>" in prompt
        assert "<domain_contract>" in prompt

    def test_without_configs(self):
        """No configs = no contract layers (backward compat)."""
        prompt = LayeredPromptBuilder.build_from_config()
        assert "<tenant_contract>" not in prompt
        assert "<domain_contract>" not in prompt

    def test_allow_common_knowledge_true(self):
        """When allow_model_common_knowledge=True, no restriction is emitted."""

        class FakeTenantConfig:
            output_config = None
            domain_config = DomainConfig(
                name="general",
                allow_model_common_knowledge=True,
            )

        prompt = LayeredPromptBuilder.build_from_config(
            tenant_config=FakeTenantConfig(),
        )
        assert "Do NOT use model common knowledge" not in prompt


class TestBackwardCompatibility:
    """Ensure existing callers still work with the expanded interface."""

    def test_old_3_arg_interface(self):
        """Old callers using only identity/context/extra_instructions still work."""
        prompt = LayeredPromptBuilder.build(
            identity="Legacy bot.",
            context={"current_date": "2026-01-01"},
            extra_instructions="Legacy instructions.",
        )
        assert "Legacy bot" in prompt
        assert "2026-01-01" in prompt
        assert "Legacy instructions" in prompt
        # No contract layers
        assert "<tenant_contract>" not in prompt
        assert "<domain_contract>" not in prompt
