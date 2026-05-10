# Imports

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
import textwrap
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import cv2


def _is_remote_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https")


def _download_video(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def _infer_suffix(url: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    return suffix if suffix in {".mp4", ".mov", ".webm", ".mkv", ".m4v"} else ".mp4"


def _probe_video(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return fps, n, w, h


def _build_mask_video_from_indexed_pngs(
    mask_dir: Path,
    output_path: Path,
    fps: float,
    size: tuple[int, int],
    frame_count: int,
) -> None:
    width, height = size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open mask video writer for {output_path}")
    try:
        for i in range(frame_count):
            mask_path = mask_dir / f"mask_{i}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Expected mask file missing: {mask_path}")
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise RuntimeError(f"Could not read mask image: {mask_path}")
            if m.shape[:2] != (height, width):
                m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
            bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
    finally:
        writer.release()


class AnnotationEngine:
    def __init__(
        self,
        video_url: str,
        *,
        work_dir: Path | None = None,
        checkpoint_dir: Path | str | None = None,
        cosmos_repo: Path | str | None = None,
        rose_runner: Path | str | None = None,
        rose_prompt: str = "human",
        sam3_conda_env: str = "sam3",
        rose_conda_env: str = "rose",
        vipe_conda_env: str = "vipe",
        diffusion_conda_env: str | None = None,
    ):
        self.video_url = video_url
        self.work_dir = (
            work_dir.expanduser().resolve() if work_dir else Path(tempfile.mkdtemp(prefix="annotation_"))
        )
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else None
        self.cosmos_repo = Path(cosmos_repo).expanduser().resolve() if cosmos_repo else None
        self.rose_runner = (
            Path(rose_runner).expanduser().resolve() if rose_runner else None
        ) or (
            Path(os.environ["ANNOTATION_ROSE_RUNNER"]).expanduser().resolve()
            if os.environ.get("ANNOTATION_ROSE_RUNNER")
            else None
        )
        self.rose_prompt = rose_prompt
        self.sam3_conda_env = sam3_conda_env
        self.rose_conda_env = rose_conda_env
        self.vipe_conda_env = vipe_conda_env
        self.diffusion_conda_env = diffusion_conda_env

        self.local_video_path: Path | None = None
        self.masks_dir: Path = self.work_dir / "masks"
        self.mask_video_path: Path = self.work_dir / "mask_video.mp4"
        self.rose_output_dir: Path = self.work_dir / "rose_out"
        self.removed_human_video_path: Path = self.work_dir / "human_removed.mp4"
        self.vipe_results_dir: Path = self.work_dir / "vipe_results"
        self.diffusion_output_dir: Path = self.work_dir / "diffusion_output_generated"

    def run(self) -> None:
        self.preAnnotation()
        self.inAnnotation()
        self.postAnnotation()

    def _resolve_input_video(self) -> Path:
        if _is_remote_url(self.video_url):
            dest = self.work_dir / f"input{_infer_suffix(self.video_url)}"
            return _download_video(self.video_url, dest)
        p = Path(self.video_url).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Video path does not exist: {p}")
        return p

    def _run_conda(self, env: str, args: list[str], *, cwd: Path | None = None) -> None:
        cmd = ["conda", "run", "-n", env, "--no-capture-output", *args]
        subprocess.run(cmd, check=True, cwd=cwd)

    def _write_sam3_mask_worker(self, video_path: Path, masks_dir: Path) -> str:
        saas_root = Path(__file__).resolve().parent
        return textwrap.dedent(
            f"""
            import sys
            from pathlib import Path
            sys.path.insert(0, {str(saas_root)!r})
            import numpy as np
            from PIL import Image
            import DataraAI_segmentation as D

            video = Path({str(video_path)!r})
            masks_dir = Path({str(masks_dir)!r})
            masks_dir.mkdir(parents=True, exist_ok=True)
            outputs = D.mask_generation(video)
            for i, frame_idx in enumerate(sorted(outputs.keys())):
                out = outputs[frame_idx]
                human_binary_masks = out.get("out_binary_masks") or []
                if not len(human_binary_masks):
                    raise SystemExit("No human mask for frame index %r" % (frame_idx,))
                mask = human_binary_masks[0].astype(np.uint8) * 255
                Image.fromarray(mask).save(masks_dir / ("mask_%d.png" % i))
            """
        ).strip()

    def preAnnotation(self) -> None:
        self.local_video_path = self._resolve_input_video()
        assert self.local_video_path is not None

        fps, frame_count, width, height = _probe_video(self.local_video_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix="_sam3_masks.py", delete=False) as tmp:
            tmp.write(self._write_sam3_mask_worker(self.local_video_path, self.masks_dir))
            worker_path = Path(tmp.name)

        try:
            self._run_conda(self.sam3_conda_env, ["python", str(worker_path)])
        finally:
            worker_path.unlink(missing_ok=True)

        _build_mask_video_from_pngs(
            self.masks_dir,
            self.mask_video_path,
            fps,
            (width, height),
            frame_count,
        )

        if not self.rose_runner:
            raise RuntimeError(
                "ROSE inpainting requires --rose-runner or ANNOTATION_ROSE_RUNNER "
                "(path to a ROSE driver script in the rose conda environment)."
            )
        if not self.rose_runner.is_file():
            raise FileNotFoundError(f"rose_runner not found: {self.rose_runner}")

        self.rose_output_dir.mkdir(parents=True, exist_ok=True)
        self._run_conda(
            self.rose_conda_env,
            [
                "python",
                str(self.rose_runner),
                "--source_video",
                str(self.local_video_path),
                "--mask_video",
                str(self.mask_video_path),
                "--output_dir",
                str(self.rose_output_dir),
                "--prompt",
                self.rose_prompt,
                "--video_length",
                str(frame_count),
                "--sample_height",
                str(height),
                "--sample_width",
                str(width),
            ],
        )

        outputs = sorted(
            glob.glob(str(self.rose_output_dir / "*.mp4"))
            + glob.glob(str(self.rose_output_dir / "*.mov"))
            + glob.glob(str(self.rose_output_dir / "*.webm"))
        )
        if not outputs:
            nested = glob.glob(str(self.rose_output_dir / "**" / "*.mp4"), recursive=True)
            outputs = sorted(nested)
        if not outputs:
            raise RuntimeError(f"No video produced under ROSE output directory {self.rose_output_dir}")
        shutil.copy2(outputs[0], self.removed_human_video_path)

    def inAnnotation(self) -> None:
        if not self.removed_human_video_path.is_file():
            raise RuntimeError(
                f"Missing inpainted video at {self.removed_human_video_path}; run preAnnotation first."
            )
        self.vipe_results_dir.mkdir(parents=True, exist_ok=True)
        self._run_conda(
            self.vipe_conda_env,
            [
                "vipe",
                "infer",
                str(self.removed_human_video_path),
                "--pipeline",
                "lyra",
                "--output",
                str(self.vipe_results_dir),
            ],
        )

        vipe_rgb_video = self.vipe_results_dir / "rgb" / "video.mp4"
        if not vipe_rgb_video.is_file():
            raise RuntimeError(
                f"VIPE did not write expected file at {vipe_rgb_video}. "
                "Check vipe output layout for your lyra pipeline."
            )

        if not self.checkpoint_dir or not self.cosmos_repo:
            raise RuntimeError(
                "Diffusion step requires checkpoint_dir and cosmos_repo "
                "(Lyra / cosmos_predict1 checkout)."
            )
        if not self.cosmos_repo.is_dir():
            raise FileNotFoundError(f"cosmos_repo is not a directory: {self.cosmos_repo}")
        script_rel = Path("cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py")
        if not (self.cosmos_repo / script_rel).is_file():
            raise FileNotFoundError(
                f"Expected diffusion script at {self.cosmos_repo / script_rel}"
            )

        self.diffusion_output_dir.mkdir(parents=True, exist_ok=True)

        bash_inner = " ".join(
            [
                "CUDA_HOME=$CONDA_PREFIX",
                "PYTHONPATH=$(pwd)",
                "torchrun",
                "--nproc_per_node=1",
                str(script_rel),
                "--checkpoint_dir",
                str(self.checkpoint_dir),
                "--vipe_path",
                str(vipe_rgb_video),
                "--video_save_folder",
                str(self.diffusion_output_dir),
                "--disable_prompt_upsampler",
                "--num_gpus",
                "1",
                "--foreground_masking",
                "--multi_trajectory",
            ]
        )

        if self.diffusion_conda_env:
            subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    self.diffusion_conda_env,
                    "--no-capture-output",
                    "bash",
                    "-lc",
                    bash_inner,
                ],
                check=True,
                cwd=self.cosmos_repo,
            )
        else:
            subprocess.run(
                ["bash", "-lc", bash_inner],
                check=True,
                cwd=self.cosmos_repo,
            )

    def postAnnotation(self):
        # ...

        # Then return ego + annotations
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a video from a URL (or use a local path) and run the annotation pipeline."
    )
    parser.add_argument(
        "video_url",
        type=str,
        help="HTTPS URL to an input video, or a path to a local video file",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for intermediates and outputs (default: temporary directory)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory for cosmos_predict1 diffusion (Lyra)",
    )
    parser.add_argument(
        "--cosmos-repo",
        type=Path,
        default=None,
        help="Root of the Lyra / cosmos repo containing cosmos_predict1/diffusion/inference/",
    )
    parser.add_argument(
        "--rose-runner",
        type=Path,
        default=None,
        help="Python script that runs ROSE inpainting (also ANNOTATION_ROSE_RUNNER)",
    )
    parser.add_argument(
        "--rose-prompt",
        type=str,
        default="human",
        help="Text prompt forwarded to the ROSE runner",
    )
    parser.add_argument("--sam3-conda-env", type=str, default="sam3")
    parser.add_argument("--rose-conda-env", type=str, default="rose")
    parser.add_argument("--vipe-conda-env", type=str, default="vipe")
    parser.add_argument(
        "--diffusion-conda-env",
        type=str,
        default=None,
        help="Optional conda env name for torchrun / Lyra (CUDA_HOME=$CONDA_PREFIX in that env)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    engine = AnnotationEngine(
        cli.video_url,
        work_dir=cli.work_dir,
        checkpoint_dir=cli.checkpoint_dir,
        cosmos_repo=cli.cosmos_repo,
        rose_runner=cli.rose_runner,
        rose_prompt=cli.rose_prompt,
        sam3_conda_env=cli.sam3_conda_env,
        rose_conda_env=cli.rose_conda_env,
        vipe_conda_env=cli.vipe_conda_env,
        diffusion_conda_env=cli.diffusion_conda_env,
    )
    engine.run()
