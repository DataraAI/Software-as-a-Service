import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from packages.sam3.sam3.model_builder import build_sam3_video_predictor

def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        # response["outputs"] keys: 'out_obj_ids', 'out_probs', 'out_boxes_xywh', 'out_binary_masks', 'frame_stats'
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def mask_generation(video_path: str, segment: str = "humans"):
    video_predictor = build_sam3_video_predictor()

    # Start a session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=response["session_id"],
            frame_index=0, # Arbitrary frame index
            text=segment,
        )
    )

    outputs_per_frame = propagate_in_video(video_predictor, session_id)

    _ = video_predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    video_predictor.shutdown()

    return outputs_per_frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, help="video_path")
    parser.add_argument("--segment", type=str, help="segment")
    
    args = parser.parse_args()
    video_path = args.video_path
    segment = args.segment
    if not video_path.endswith(".mp4"):
        raise ValueError("Video path must end with .mp4")
        exit(1)

    outputs_per_frame = mask_generation(video_path, segment)

    out_dir = Path.home() / "masks" / video_path.split("/")[-1].split(".")[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    n_digits = len(str(len(outputs_per_frame)))

    for frame_index, output in outputs_per_frame.items():
        masks = output["out_binary_masks"]
        obj_ids = output["out_obj_ids"]
        for i in range(len(masks)):
            mask = masks[i]
            oid = obj_ids[i]
            arr = (mask.astype(np.uint8)) * 255
            oid_dir = out_dir / str(int(oid))
            oid_dir.mkdir(parents=True, exist_ok=True)
            path = oid_dir / f"frame_{frame_index:0{n_digits}d}.png"
            Image.fromarray(arr, mode="L").save(path)


if __name__ == "__main__":
    main()
