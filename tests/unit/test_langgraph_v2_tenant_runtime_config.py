from app.config.models import (
    FlowConfig,
    LangGraphRuntimeMode,
    LLMConfig,
    TenantConfig,
)


def _tenant(**overrides: object) -> TenantConfig:
    return TenantConfig.model_validate(
        {
            "kmsAppName": "Runtime Mode Test",
            "applicationId": "runtime-mode-test",
            "adGroups": [],
            "llmConfig": LLMConfig(models={}),
            "flowConfig": FlowConfig(),
            **overrides,
        }
    )


def test_runtime_mode_defaults_to_linear() -> None:
    assert _tenant().runtime_mode is LangGraphRuntimeMode.LINEAR


def test_runtime_mode_accepts_and_serializes_the_tenant_alias() -> None:
    tenant = _tenant(runtimeMode="agent")

    assert tenant.runtime_mode is LangGraphRuntimeMode.AGENT
    assert tenant.model_dump(by_alias=True)["runtimeMode"] == "agent"
