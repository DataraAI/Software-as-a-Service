from packages.sam3.model_builder import build_sam3_video_predictor

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

    output = response["outputs"]
    masks, _, _ = output["masks"], output["boxes"], output["scores"]

    _ = video_predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    video_predictor.shutdown()
    return masks
