import os
import re
import json
import tempfile
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.image_utils import load_image


# ----------------------------
# Prompt (keep simple + strict)
# ----------------------------
SYSTEM_PROMPT = (
    "You are a production vision tagging engine. "
    "Return ONLY valid JSON. No markdown. No explanations."
)

USER_PROMPT = """
Generate semantic tags for the image.

Output MUST be ONLY valid JSON with this exact schema:
{"VLMTags":["tag1","tag2","tag3"]}

Rules:
- 3 to 8 tags total
- tags are short lowercase phrases (1-3 words)
- no duplicates
- no extra keys
- no explanations
"""


# ----------------------------
# Model singleton (load once)
# ----------------------------
_MODEL: Optional[Qwen2_5_VLForConditionalGeneration] = None
_PROCESSOR: Optional[AutoProcessor] = None

def _get_model(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """
    Loads model+processor once per process and keeps it on GPU.
    """
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            # If installed, can speed up:
            # attn_implementation="flash_attention_2",
        ).eval()
        _PROCESSOR = AutoProcessor.from_pretrained(model_id)
    return _MODEL, _PROCESSOR


# ---------------------------------------
# Input handling: local path or HTTP(S) URL
# ---------------------------------------
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

def _is_url(s: str) -> bool:
    return bool(s) and bool(_URL_RE.match(s))

def _ensure_local_image(image_ref: str, timeout_s: int = 20) -> Tuple[str, Optional[str]]:
    """
    Returns (local_path, tmp_path_or_none).
    If image_ref is a URL (incl. SAS), downloads to temp file.
    """
    if not image_ref:
        raise ValueError("image_ref is empty")

    if _is_url(image_ref):
        parsed = urlparse(image_ref)
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
            ext = ".img"

        fd, tmp_path = tempfile.mkstemp(prefix="vlm_", suffix=ext)
        os.close(fd)

        with requests.get(image_ref, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        return tmp_path, tmp_path

    # Treat as local file path
    if not os.path.exists(image_ref):
        raise FileNotFoundError(f"Local image not found: {image_ref}")

    return image_ref, None


# ----------------------------
# JSON extraction + validation
# ----------------------------
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

class PostAnnotationError(RuntimeError):
    pass

def _normalize_tags(tags: List[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    for t in tags:
        if not isinstance(t, str):
            continue
        s = t.strip().lower()
        s = re.sub(r"\s+", " ", s)
        if not s:
            continue
        if s in seen:
            continue
        # keep tags short-ish
        if len(s) > 40:
            continue
        seen.add(s)
        out.append(s)

    return out[:8]

def _extract_vlmtags_json(text: str) -> Dict[str, List[str]]:
    """
    Enforces output schema: {"VLMTags":[...]}
    If model returns extra text, extracts the first {...} block.
    """
    if not isinstance(text, str) or not text.strip():
        raise PostAnnotationError("Empty model output")

    s = text.strip()

    # If it isn't pure JSON, attempt to extract a JSON object substring.
    if not (s.startswith("{") and s.endswith("}")):
        m = _JSON_OBJ_RE.search(s)
        if not m:
            raise PostAnnotationError("No JSON object found in model output")
        s = m.group(0)

    try:
        obj = json.loads(s)
    except Exception as e:
        raise PostAnnotationError(f"Model output is not valid JSON: {e}") from e

    if not isinstance(obj, dict) or set(obj.keys()) != {"VLMTags"}:
        raise PostAnnotationError('Output must be exactly: {"VLMTags":[...]}')

    tags = obj.get("VLMTags")
    if not isinstance(tags, list):
        raise PostAnnotationError('"VLMTags" must be a list')

    norm = _normalize_tags(tags)
    if len(norm) < 3:
        raise PostAnnotationError("Too few tags (<3) after normalization")

    return {"VLMTags": norm}


# ----------------------------
# Inference (single code path)
# ----------------------------
@torch.inference_mode()
def _infer_raw(image_path: str) -> str:
    model, processor = _get_model()
    image = load_image(image_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Stability-focused generation params
    temperature = 0.1
    do_sample = temperature > 0.0

    out_ids = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=do_sample,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.05,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    return processor.batch_decode(out_ids, skip_special_tokens=True)[0]


# ----------------------------
# Public function your engine calls
# ----------------------------
def post_annotation(image_ref: str) -> Dict[str, List[str]]:
    """
    Entry point for production.
    Input: local path OR http(s)/SAS URL
    Output: {"VLMTags": ["tag1", ...]}
    """
    tmp_path = None
    try:
        local_path, tmp_path = _ensure_local_image(image_ref)
        raw = _infer_raw(local_path)
        return _extract_vlmtags_json(raw)
    except Exception as e:
        raise PostAnnotationError(f"post_annotation failed: {e}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
