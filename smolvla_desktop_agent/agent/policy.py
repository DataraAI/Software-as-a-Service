from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@dataclass
class PolicyConfig:
    model_id: str
    device: str = "cuda"
    state_dim: int = 8
    image_key: str = "observation.images.main"
    state_key: str = "observation.state"
    task_key: str = "task"


class SmolVLADesktopPolicy:
    def __init__(self, cfg: PolicyConfig) -> None:
        self.cfg = cfg

        if cfg.device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(cfg.device)

        self.policy = SmolVLAPolicy.from_pretrained(cfg.model_id).to(self.device).eval()

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            cfg.model_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def build_frame(
        self,
        image: Image.Image,
        instruction: str,
        state: Optional[Sequence[float]] = None,
    ) -> dict:
        """
        Build a frame dict for SmolVLA.
        IMPORTANT:
        The keys must match the input feature names your fine-tuned checkpoint expects.
        Defaults here are examples for a simple desktop setup.
        """
        if state is None:
            state = np.zeros(self.cfg.state_dim, dtype=np.float32)
        else:
            state = np.asarray(state, dtype=np.float32)

        frame = {
            self.cfg.image_key: image,
            self.cfg.state_key: state,
            self.cfg.task_key: instruction,
        }
        return frame

    @torch.inference_mode()
    def predict_action_vector(
        self,
        image: Image.Image,
        instruction: str,
        state: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        frame = self.build_frame(image=image, instruction=instruction, state=state)

        pred_action = self.policy.select_action(frame)
        pred_action = self.postprocess(pred_action)

        if isinstance(pred_action, torch.Tensor):
            pred_action = pred_action.detach().cpu().numpy()
        else:
            pred_action = np.asarray(pred_action)

        pred_action = np.asarray(pred_action, dtype=np.float32).flatten()
        return pred_action