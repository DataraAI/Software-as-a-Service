from __future__ import annotations

from robotask_manipulator.config import SemanticSettings, load_settings


def test_semantic_settings_prefers_local_model_path() -> None:
    settings = SemanticSettings(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        local_model_path="/models/qwen2.5-vl-7b",
    )
    assert settings.model_source == "/models/qwen2.5-vl-7b"


def test_load_settings_reads_service_and_azure_env(monkeypatch) -> None:
    monkeypatch.setenv("RTM_SERVICE_PORT", "8123")
    monkeypatch.setenv("RTM_SERVICE_AUTH_TOKEN", "secret-token")
    monkeypatch.setenv("RTM_AZURE_SOURCE_CONTAINER_ALLOWLIST", "media,uploads")
    monkeypatch.setenv("RTM_AZURE_ANNOTATION_CONTAINER", "datablob-annotations")

    settings = load_settings()

    assert settings.service.port == 8123
    assert settings.service.auth_token == "secret-token"
    assert settings.azure.source_container_allowlist == ("media", "uploads")
    assert settings.azure.annotation_container == "datablob-annotations"
