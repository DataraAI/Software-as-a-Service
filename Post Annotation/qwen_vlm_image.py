from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import argparse
import json
import os
import torch
import requests
import nltk
for resource in ("punkt", "punkt_tab", "stopwords"):
    try:
        if resource == "stopwords":
            nltk.data.find("corpora/stopwords")
        else:
            nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

"""
Example usage:

  # From URL (requires network access to the host):
  python Post\\ Annotation/qwen_vlm_image.py \\
    --prompt "Describe the image." \\
    --egoURL "https://daasblob.blob.core.windows.net/roboteyeview/carAutomation/BMW/frontGrille/egos/frontGrille_016_Rotate_right_90_degrees.png"
"""



parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, help="input prompt", default="Describe the image.")
parser.add_argument("--egoURL", type=str, help="input URL of ego image")
# parser.add_argument("--container_name", type=str, help="Azure Blob's container name")

args = parser.parse_args()
prompt = args.prompt
egoURL = args.egoURL
# container_name = args.container_name

if "?" in egoURL:
    egoURL = egoURL[:egoURL.index("?")]


# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": egoURL,
            },
            {"type": "text", "text": prompt},
        ],
    }
]

# Preparation for inference
# egoURL can be an https URL or a local path; qwen_vl_utils accepts both.
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
try:
    image_inputs, video_inputs = process_vision_info(messages)
except (OSError, requests.exceptions.RequestException) as e:
    if isinstance(egoURL, str) and (egoURL.startswith("http://") or egoURL.startswith("https://")):
        raise SystemExit(
            "Failed to fetch image from URL (host unreachable or DNS error). "
            "Use a local path instead, e.g. --egoURL /path/to/image.png"
        ) from e
    raise
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# Extract keywords from VLM output using NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
combined_text = " ".join(output_text) if isinstance(output_text, list) else output_text
tokens = word_tokenize(combined_text.lower())
keywords = [
    t for t in tokens
    if t.isalnum() and len(t) > 1 and t not in stop_words
]
# Preserve order, remove duplicates (first occurrence kept)
seen = set()
vlm_tags = []
for k in keywords:
    if k not in seen:
        seen.add(k)
        vlm_tags.append(k)

schema_output = {"VLM_tags": vlm_tags}
with open(os.path.expanduser("~") + "/vlm_tags.json", "w") as f:
    json.dump(schema_output, f, indent=2)
print(os.path.expanduser("~") + "/vlm_tags.json")
