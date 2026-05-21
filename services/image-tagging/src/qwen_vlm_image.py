from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import argparse
import json
import os
import torch
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def setup_nltk():
    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            if resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            else:
                nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

def main(argv=None):
    setup_nltk()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="input prompt", default="Describe the image.")
    parser.add_argument("--egoURL", type=str, help="input URL of ego image")
    parser.add_argument("--output_json", type=str, help="output path for JSON")

    args = parser.parse_args(argv)
    prompt = args.prompt
    egoURL = args.egoURL

    if not egoURL:
        print("Error: egoURL is required.")
        return None

    if "?" in egoURL:
        egoURL = egoURL[:egoURL.index("?")]

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

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
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    stop_words = set(stopwords.words("english"))
    combined_text = " ".join(output_text) if isinstance(output_text, list) else output_text
    tokens = word_tokenize(combined_text.lower())
    keywords = [
        t for t in tokens
        if t.isalnum() and len(t) > 1 and t not in stop_words
    ]
    seen = set()
    vlm_tags = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            vlm_tags.append(k)

    schema_output = {"VLM_tags": vlm_tags}
    
    output_path = args.output_json or (os.path.expanduser("~") + "/vlm_tags.json")
    with open(output_path, "w") as f:
        json.dump(schema_output, f, indent=2)
    
    print(output_path)
    return output_path

if __name__ == "__main__":
    main()
