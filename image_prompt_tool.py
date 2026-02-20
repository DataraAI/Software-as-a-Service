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
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

base_model_path = "./models/qwen-multiple-angles-2509/qwen-base"
lora_path = "./models/qwen-multiple-angles-2509/qwen-image-edit-lora"


pipe = DiffusionPipeline.from_pretrained(
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
import os



base_name = os.path.basename(imageURL)          # frontGrille_000.png
name_no_ext = os.path.splitext(base_name)[0]    # frontGrille_000
prompt_joined = "_".join(promptSplit)           # Rotate_right_45_degrees

# frontGrille_000_ego_Rotate_right_45_degrees.png
# or
# frontGrille_000_ego_Rotate_right_45_degrees.png
new_filename = f"{name_no_ext}_{prompt_joined}.png"

# "/".join(blobPathComponents)
# ==
# '~/ego_images/roboteyeview/automotive/bmw/frontGrille/egos'
imageFilepath = os.path.abspath("ego_images")
imageFilepath += "/" + "/".join(blobPathComponents)
os.makedirs(imageFilepath, exist_ok=True)

imageFilepath += "/" + new_filename
image.save(imageFilepath)

print(imageFilepath)
# print(imageFilepath[imageFilepath.index("ego_images/"):])
