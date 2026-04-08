from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.batch_annotate import run_batch
from scripts.run_single_inference import run
from robotask_manipulator.utils.io import load_json_file


def test_single_inference_end_to_end(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = root / "data" / "sample_inputs" / "sample_workflow_episode_001.json"
    output_path, raw_path, output = run(
        input_path=input_path,
        output_dir=tmp_path,
        semantic_offline=True,
        action_backend="none",
    )

    assert output_path.exists()
    assert raw_path is not None and raw_path.exists()
    payload = load_json_file(output_path)
    assert payload["episode_id"] == output.episode_id
    assert payload["simulation_export"]["robot"] == "franka_panda"


def test_batch_manifest_generation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    args = Namespace(
        input_dir=str(root / "data" / "sample_inputs"),
        output_dir=str(tmp_path),
        config=None,
        semantic_model=None,
        semantic_backend=None,
        semantic_offline=True,
        action_backend="none",
        model_id=None,
        checkpoint=None,
        device=None,
        dtype=None,
        offline=False,
        benchmark=None,
    )
    status, manifest_path = run_batch(args)
    assert status == 0
    assert Path(manifest_path).exists()
    manifest = load_json_file(manifest_path)
    assert manifest["summary"]["episodes"] >= 1
