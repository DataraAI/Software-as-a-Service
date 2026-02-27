# datara_ai/annotation/frame_annotator.py
from typing import Optional, List, Dict, Any
import torch

from transformers.image_utils import load_image
from qwen_vl_utils import process_vision_info

from .hf_model import load_qwen
from .prompts import frame_label_prompt
from .parsing import extract_first_json


class FrameAnnotator:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 256,
        allowed_parts: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        allowed_actions: Optional[List[str]] = None,
    ):
        self.model, self.processor = load_qwen(model_id)
        self.max_new_tokens = max_new_tokens
        self.allowed_parts = allowed_parts
        self.allowed_tools = allowed_tools
        self.allowed_actions = allowed_actions

    def annotate(self, image_path: str) -> Dict[str, Any]:
        image = load_image(image_path)
        prompt = frame_label_prompt(
            allowed_parts=self.allowed_parts,
            allowed_tools=self.allowed_tools,
            allowed_actions=self.allowed_actions,
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Safe device move: if CUDA exists, move inputs to cuda.
        # (Model uses device_map="auto", so it can span devices; cuda inputs is usually correct.)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return extract_first_json(output_text)