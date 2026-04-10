"""Media ingestion and frame extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from robotask_manipulator.config import IngestionSettings
from robotask_manipulator.schemas import EpisodeInput, FrameObservation, MediaMetadata, MediaType
from robotask_manipulator.utils.validation import InvalidInputError, ensure_asset_exists

LOGGER = logging.getLogger(__name__)


class MediaIngestor:
    """Convert images, videos, or explicit frame sequences into one canonical episode input."""

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".ppm"}
    VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}

    def __init__(self, settings: IngestionSettings) -> None:
        self.settings = settings

    def from_payload(self, payload: dict[str, Any], base_dir: str | Path) -> EpisodeInput:
        """Build an episode input from a payload."""
        base_dir = Path(base_dir)

        if "frames" in payload:
            frames = [self._frame_from_mapping(frame, base_dir, index) for index, frame in enumerate(payload["frames"])]
            media_metadata = MediaMetadata(
                media_type=MediaType.FRAME_SEQUENCE,
                source_ref=str(base_dir),
                frame_count=len(frames),
                metadata=payload.get("metadata", {}),
            )
        else:
            media_path = self._resolve_path(base_dir, payload["media_path"] if "media_path" in payload else payload["asset_path"])
            media_type = self._detect_media_type(media_path)
            if media_type == MediaType.IMAGE:
                frames, media_metadata = self._ingest_image(
                    media_path,
                    payload.get("metadata", {}),
                    payload.get("state"),
                )
            elif media_type == MediaType.VIDEO:
                frames, media_metadata = self._ingest_video(media_path, payload.get("metadata", {}))
            else:
                raise InvalidInputError(f"Unsupported media type for input: {media_path}")

        return EpisodeInput(
            episode_id=payload["episode_id"],
            task_name=payload.get("task_name", payload["episode_id"]),
            instruction=payload["instruction"],
            media_metadata=media_metadata,
            frames=frames,
            metadata=payload.get("metadata", {}),
            benchmark=payload.get("benchmark"),
        )

    def _frame_from_mapping(self, mapping: dict[str, Any], base_dir: Path, index: int) -> FrameObservation:
        asset_ref = str(self._resolve_path(base_dir, mapping["asset_ref"]))
        return FrameObservation(
            frame_id=mapping.get("frame_id", f"frame-{index:03d}"),
            asset_ref=asset_ref,
            frame_index=mapping.get("frame_index", index),
            timestamp_s=mapping.get("timestamp_s"),
            state=mapping.get("state"),
            metadata=mapping.get("metadata", {}),
        )

    def _ingest_image(
        self,
        image_path: Path,
        metadata: dict[str, Any],
        state: list[float] | None = None,
    ) -> tuple[list[FrameObservation], MediaMetadata]:
        asset = ensure_asset_exists(image_path)
        with Image.open(asset) as image:
            width, height = image.size
        frame = FrameObservation(
            frame_id="frame-000",
            asset_ref=str(asset),
            frame_index=0,
            timestamp_s=0.0,
            state=state,
            metadata=metadata,
        )
        media_metadata = MediaMetadata(
            media_type=MediaType.IMAGE,
            source_ref=str(asset),
            width=width,
            height=height,
            frame_count=1,
            metadata=metadata,
        )
        return [frame], media_metadata

    def _ingest_video(self, video_path: Path, metadata: dict[str, Any]) -> tuple[list[FrameObservation], MediaMetadata]:
        asset = ensure_asset_exists(video_path)
        frames, fps = self._extract_video_frames(asset)
        media_metadata = MediaMetadata(
            media_type=MediaType.VIDEO,
            source_ref=str(asset),
            fps=fps,
            frame_count=len(frames),
            duration_s=(len(frames) / fps) if fps else None,
            metadata=metadata,
        )
        return frames, media_metadata

    def _extract_video_frames(self, video_path: Path) -> tuple[list[FrameObservation], float | None]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise InvalidInputError(
                "Video ingestion requires OpenCV. Install opencv-python-headless or use extracted frames instead."
            ) from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise InvalidInputError(f"Unable to open video input: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
        frames: list[FrameObservation] = []
        frame_index = 0
        saved_index = 0
        stride = max(1, self.settings.video_frame_stride)
        output_dir = video_path.parent / f"{video_path.stem}_frames"
        output_dir.mkdir(parents=True, exist_ok=True)

        max_frames = self.settings.max_frames if self.settings.max_frames > 0 else None

        while capture.isOpened() and (max_frames is None or saved_index < max_frames):
            success, frame = capture.read()
            if not success:
                break
            if frame_index % stride == 0:
                output_path = output_dir / f"frame_{saved_index:04d}.jpg"
                cv2.imwrite(str(output_path), frame)
                timestamp_s = (frame_index / fps) if fps else None
                frames.append(
                    FrameObservation(
                        frame_id=f"frame-{saved_index:03d}",
                        asset_ref=str(output_path.resolve()),
                        frame_index=saved_index,
                        timestamp_s=timestamp_s,
                    )
                )
                saved_index += 1
            frame_index += 1

        capture.release()
        LOGGER.info("Extracted %s frames from %s", len(frames), video_path.name)
        if not frames:
            raise InvalidInputError(f"No frames were extracted from video input: {video_path}")
        return frames, fps

    def _detect_media_type(self, path: Path) -> MediaType:
        suffix = path.suffix.lower()
        if suffix in self.IMAGE_SUFFIXES:
            return MediaType.IMAGE
        if suffix in self.VIDEO_SUFFIXES:
            return MediaType.VIDEO
        raise InvalidInputError(f"Unsupported media type for path: {path}")

    def _resolve_path(self, base_dir: Path, candidate: str | Path) -> Path:
        path = Path(candidate)
        return path if path.is_absolute() else (base_dir / path).resolve()
