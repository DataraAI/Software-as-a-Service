from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_colab_video_demo import run_demo


def test_run_demo_writes_payloads_and_runs_inference(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "test_image.jpg"
    video_path = tmp_path / "test_video.mp4"
    image_path.write_bytes(b"image")
    video_path.write_bytes(b"video")

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    calls: list[dict] = []

    def fake_run_single_inference(*, input_path, output_dir, config_path, action_backend):
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        calls.append(
            {
                "payload": payload,
                "output_dir": Path(output_dir),
                "config_path": config_path,
                "action_backend": action_backend,
            }
        )
        episode_id = payload["episode_id"]
        return (
            Path(output_dir) / f"{episode_id}.json",
            Path(output_dir) / f"{episode_id}.raw.json",
            SimpleNamespace(
                episode_id=episode_id,
                frame_predictions=[object()],
                segments=[
                    SimpleNamespace(
                        step_index=0,
                        semantic=SimpleNamespace(description="hold cable near port"),
                        symbolic_action=SimpleNamespace(label="hold"),
                        frame_start_index=0,
                        frame_end_index=2,
                    )
                ],
            ),
        )

    monkeypatch.setattr("scripts.run_colab_video_demo.run_single_inference", fake_run_single_inference)

    results = run_demo(
        image_path=image_path,
        video_path=video_path,
        input_dir=input_dir,
        output_dir=output_dir,
        config_path="configs/colab_refined_video.yaml",
        instruction="Describe only the visible hand-object action conservatively.",
        task_name="ethernet_cable_insert",
        tags=["ethernet cable", "laptop port"],
        zip_outputs=False,
    )

    assert len(calls) == 2
    assert calls[0]["payload"]["metadata"]["tags"] == ["ethernet cable", "laptop port"]
    assert calls[1]["payload"]["metadata"]["tags"] == ["ethernet cable", "laptop port"]
    assert Path(results["payload_paths"]["image"]).exists()
    assert Path(results["payload_paths"]["video"]).exists()
    assert calls[0]["payload"]["asset_path"] == str(image_path.resolve())
    assert calls[1]["payload"]["asset_path"] == str(video_path.resolve())
