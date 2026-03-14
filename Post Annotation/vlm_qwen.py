# vlm_qwen.py
import json
import torch
from transformers import AutoProcessor
from transformers.image_utils import load_image
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

class QwenVLM:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", max_new_tokens=256):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        # Load once
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def describe_frame(self, image_path: str) -> dict:
        """
        Returns a structured description of a manufacturing frame.
        Output is JSON (dict) so you can store it easily.
        """
        image = load_image(image_path)

        prompt = """
You are inspecting a single frame from a manufacturing process.

Return ONLY valid JSON with this schema:
{
  "summary": "1-2 sentences describing what is happening",
  "parts": ["..."],
  "tools": ["..."],
  "actions": ["..."],
  "quality_notes": ["..."]
}

Rules:
- Be concrete (name visible parts/tools/actions).
- If uncertain, make best guesses but keep them plausible.
- No markdown, no extra text outside JSON.
""".strip()

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

        # Safe device handling
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

        # Extract JSON
        start = output_text.find("{")
        end = output_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model did not return JSON.\nOutput:\n{output_text}")

        return json.loads(output_text[start:end+1])