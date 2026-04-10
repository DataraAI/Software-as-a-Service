from __future__ import annotations

from robotask_manipulator.config import SemanticSettings


def test_semantic_settings_prefers_local_model_path() -> None:
    settings = SemanticSettings(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        local_model_path="/models/qwen2.5-vl-7b",
    )
    assert settings.model_source == "/models/qwen2.5-vl-7b"
