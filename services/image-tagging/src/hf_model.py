# datara_ai/annotation/hf_model.py
import torch
from typing import TYPE_CHECKING

# Import for type checkers / linters only. At runtime we import inside load_qwen
if TYPE_CHECKING:
    pass

_MODEL = None
_PROCESSOR = None

def load_qwen(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """
    Loads and caches model + processor (singleton).
    Call this from anywhere; it will only load once.
    """
    global _MODEL, _PROCESSOR
    # Import transformers at runtime to avoid import errors in environments
    # where the package isn't installed (helps linters/static analysis).
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as e:
        raise ImportError(
            "The 'transformers' package is required to load the Qwen model. "
            "Install it (and any required extensions) and try again."
        ) from e

    if _MODEL is None or _PROCESSOR is None:
        _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        _PROCESSOR = AutoProcessor.from_pretrained(model_id)
    return _MODEL, _PROCESSOR

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
