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

    # output keys: 'out_obj_ids', 'out_probs', 'out_boxes_xywh', 'out_binary_masks', 'frame_stats'
    output = response["outputs"]
    outputs_per_frame = propagate_in_video(video_predictor, session_id)

    _ = video_predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    video_predictor.shutdown()

    return outputs_per_frame
