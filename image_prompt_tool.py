"""
Stub xformers.ops before any other imports so diffusers never loads the real one.
The real xformers uses triton APIs that are incompatible with PyTorch 2.7+ (e.g.
JITCallable._set_src / 'Cannot set attribute src directly'). We provide a fallback
using PyTorch's scaled_dot_product_attention so the pipeline still runs.
"""
import importlib.util
import sys
import types

_xformers = types.ModuleType("xformers")
_xformers_ops = types.ModuleType("xformers.ops")
# Stubs need __spec__ so importlib.util.find_spec("xformers") does not raise.
_xformers.__spec__ = importlib.util.spec_from_loader("xformers", None, is_package=True)
_xformers_ops.__spec__ = importlib.util.spec_from_loader("xformers.ops", None, is_package=False)


def _memory_efficient_attention_fallback(query, key, value, attn_bias=None, op=None, scale=None, **kwargs):
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_bias, dropout_p=kwargs.get("p", 0.0), scale=scale
    )


_xformers_ops.memory_efficient_attention = _memory_efficient_attention_fallback
_xformers.ops = _xformers_ops
sys.modules["xformers"] = _xformers
sys.modules["xformers.ops"] = _xformers_ops

import os
import urllib.request
from PIL import Image
from io import BytesIO

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, help="input prompt")
parser.add_argument("--imageURL", type=str, help="input imageURL")
parser.add_argument("--container_name", type=str, help="Azure Blob's container name")

args = parser.parse_args()
prompt = args.prompt
imageURL = args.imageURL

if "?" in imageURL:
    imageURL = imageURL[:imageURL.index("?")]

container_name = args.container_name

# Example imageURL
# https://datara04749.blob.core.windows.net/roboteyeview/automotive/bmw/frontGrille/orig/frontGrille_000.png

blobPath = imageURL[imageURL.index(container_name) : imageURL.index("/orig") + 5]
blobPathComponents = blobPath.split("/")
# [roboteyeview, automotive, bmw, frontGrille, orig]
blobPathComponents[-1] = "egos"
# [roboteyeview, automotive, bmw, frontGrille, egos]


import torch
from diffusers.utils import load_image

try:
    from diffusers import QwenImageEditPlusPipeline
except (ImportError, AttributeError) as e:
    raise RuntimeError(
        "The qwen-multiple-angles-2509 base checkpoint requires diffusers >= 0.36.0 "
        "(QwenImageEditPlusPipeline). Upgrade on this machine with: "
        "pip install -U 'diffusers>=0.36.0'"
    ) from e

base_model_path = "/home/ubuntu/models/qwen-multiple-angles-2509/qwen-base"
lora_path = "/home/ubuntu/models/qwen-multiple-angles-2509/qwen-image-edit-lora"


pipe = QwenImageEditPlusPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda")
pipe.load_lora_weights(lora_path)

input_image = load_image(imageURL)

output = pipe(image=input_image, prompt=prompt)
image = output.images[0]




# # e.g. prompt: "Rotate right 45 degrees, and remove the human(s)."
promptCommaInd = len(prompt)
if "," in prompt:
    promptCommaInd = prompt.index(",")

prompt = prompt[:promptCommaInd]
promptSplit = prompt.split(" ")
# promptSplit: []
# "_".join(promptSplit) becomes "Rotate_right_45_degrees"

from PIL import Image
# Change new_image.png to the following format:
# imageURL basename: frontGrille_000.png
# new basename: frontGrille_000_ego_Rotate_right_45_degrees.png

base_name = os.path.basename(imageURL)          # frontGrille_000.png
name_no_ext = os.path.splitext(base_name)[0]    # frontGrille_000
prompt_joined = "_".join(promptSplit)           # Rotate_right_45_degrees

# frontGrille_000_ego_Rotate_right_45_degrees.png
# or
# frontGrille_000_ego_Rotate_right_45_degrees.png
new_filename = f"{name_no_ext}_{prompt_joined}.png"

# "/".join(blobPathComponents)
# ==
# /home/ubuntu/ego_images/roboteyeview/automotive/bmw/frontGrille/egos
imageFilepath = "/home/ubuntu/ego_images"
imageFilepath = os.path.join(imageFilepath, *blobPathComponents)
os.makedirs(imageFilepath, exist_ok=True)

imageFilepath = os.path.join(imageFilepath, new_filename)
image.save(imageFilepath)

print(imageFilepath)
# print(imageFilepath[imageFilepath.index("ego_images/"):])