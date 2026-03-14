"""
Stub xformers.ops before any diffusers import to avoid triton compatibility errors
(JITCallable._set_src / 'Cannot set attribute src directly' with PyTorch 2.7+).
"""
import importlib.util
import sys
import types

_xformers = types.ModuleType("xformers")
_xformers_ops = types.ModuleType("xformers.ops")
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

import argparse
import os
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image, ImageDraw

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    DDIMScheduler,
)
from diffusers.utils import load_image


def make_inpaint_condition(image: Image.Image, mask_image: Image.Image) -> torch.Tensor:
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    mask_np = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
    assert image_np.shape[0:2] == mask_np.shape[0:2], "image and mask must have same HxW"

    image_np[mask_np > 0.5] = -1.0
    image_np = np.expand_dims(image_np, 0).transpose(0, 3, 1, 2)
    return torch.from_numpy(image_np)

#Fix to resizing: letterboxing - added padding so image stays 512x512 while maintaining aspect ratio
def letterbox_to_square(image: Image.Image, target_size: int = 512, fill=0, resample=Image.Resampling.LANCZOS):
    w, h = image.size
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = image.resize((new_w, new_h), resample)

    if image.mode == "RGB":
        canvas = Image.new("RGB", (target_size, target_size), fill)
    else:
        canvas = Image.new(image.mode, (target_size, target_size), fill)

    x0 = (target_size - new_w) // 2
    y0 = (target_size - new_h) // 2
    canvas.paste(resized, (x0, y0))

    content_box = (x0, y0, x0 + new_w, y0 + new_h)
    return canvas, content_box


def unletterbox_and_resize(image: Image.Image, content_box, output_size):

    cropped = image.crop(content_box)
    return cropped.resize(output_size, Image.Resampling.LANCZOS)


def preset_mask(name: str, size=(512, 512), content_box=None) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if content_box is None:
        x_base0, y_base0, x_base1, y_base1 = 0, 0, w, h
    else:
        x_base0, y_base0, x_base1, y_base1 = content_box

    cw = x_base1 - x_base0
    ch = y_base1 - y_base0

    def rect(x0f, y0f, x1f, y1f):
        draw.rectangle(
            [
                int(x_base0 + x0f * cw),
                int(y_base0 + y0f * ch),
                int(x_base0 + x1f * cw),
                int(y_base0 + y1f * ch),
            ],
            fill=255
        )

    def ellipse(x0f, y0f, x1f, y1f):
        draw.ellipse(
            [
                int(x_base0 + x0f * cw),
                int(y_base0 + y0f * ch),
                int(x_base0 + x1f * cw),
                int(y_base0 + y1f * ch),
            ],
            fill=255
        )

    # Ground / floor-like regions
    if name == "ground_center":
        rect(0.25, 0.72, 0.75, 0.98)

    elif name == "ground_left":
        rect(0.02, 0.72, 0.42, 0.98)

    elif name == "ground_right":
        rect(0.58, 0.72, 0.98, 0.98)

    elif name == "ground_band":
        rect(0.00, 0.72, 1.00, 0.98)

    elif name == "ground_center_wide":
        rect(0.15, 0.70, 0.85, 0.98)

    # Middle regions
    elif name == "middle_center":
        rect(0.28, 0.35, 0.72, 0.72)

    elif name == "middle_left":
        rect(0.02, 0.35, 0.42, 0.72)

    elif name == "middle_right":
        rect(0.58, 0.35, 0.98, 0.72)

    elif name == "middle_band":
        rect(0.00, 0.35, 1.00, 0.72)

    # Top regions
    elif name == "top_center":
        rect(0.28, 0.02, 0.72, 0.30)

    elif name == "top_left":
        rect(0.02, 0.02, 0.42, 0.30)

    elif name == "top_right":
        rect(0.58, 0.02, 0.98, 0.30)

    elif name == "top_band":
        rect(0.00, 0.00, 1.00, 0.30)

    # Side regions
    elif name == "left_band":
        rect(0.00, 0.00, 0.30, 1.00)

    elif name == "right_band":
        rect(0.70, 0.00, 1.00, 1.00)

    # Central box
    elif name == "center_box":
        rect(0.20, 0.20, 0.80, 0.80)

    # Spill / puddle shapes
    elif name == "puddle_center":
        ellipse(0.30, 0.78, 0.70, 0.95)

    elif name == "puddle_left":
        ellipse(0.05, 0.78, 0.40, 0.95)

    elif name == "puddle_right":
        ellipse(0.60, 0.78, 0.95, 0.95)

    # Vertical/taller shape for fire/smoke near a machine/car
    elif name == "vertical_center":
        rect(0.38, 0.30, 0.62, 0.92)

    elif name == "vertical_left":
        rect(0.10, 0.30, 0.34, 0.92)

    elif name == "vertical_right":
        rect(0.66, 0.30, 0.90, 0.92)

    else:
        raise ValueError(
            f"Unknown mask preset: {name}. "
            f"Try ground_center, ground_left, ground_right, middle_center, "
            f"top_center, left_band, right_band, puddle_center, vertical_center."
        )

    return mask


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--imageURL", type=str, required=True)
    parser.add_argument("--container_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--negative_prompt", type=str, default="lowres, blurry, bad anatomy, artifacts")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--maskURL", type=str, default=None)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--mask_preset", type=str, default=None)
    parser.add_argument("--out_root", type=str, default=os.path.expanduser("~/corner_images_controlnet"))

    args, _ = parser.parse_known_args(argv)
    prompt = args.prompt
    imageURL = args.imageURL

    if "?" in imageURL:
        imageURL = imageURL[: imageURL.index("?")]

    container_name = args.container_name

    # Load original image and remember original size
    original_image = load_image(imageURL).convert("RGB")
    original_size = original_image.size

    # Letterbox to 512x512 instead of stretching
    init_image, content_box = letterbox_to_square(
        original_image,
        target_size=512,
        fill=(0, 0, 0),
        resample=Image.Resampling.LANCZOS,
    )

    # Build/load mask in a way that stays aligned with the letterboxed image
    if args.mask_path:
        raw_mask = Image.open(args.mask_path).convert("L")
        mask_image, _ = letterbox_to_square(
            raw_mask,
            target_size=512,
            fill=0,
            resample=Image.Resampling.LANCZOS,
        )
    elif args.maskURL:
        raw_mask = load_image(args.maskURL).convert("L")
        mask_image, _ = letterbox_to_square(
            raw_mask,
            target_size=512,
            fill=0,
            resample=Image.Resampling.LANCZOS,
        )
    elif args.mask_preset:
        mask_image = preset_mask(args.mask_preset, size=(512, 512), content_box=content_box)
    else:
        print("Warning: No mask, maskURL, or mask_preset provided. Using an empty mask.")
        mask_image = Image.new("L", (512, 512), 0)

    control_image = make_inpaint_condition(init_image, mask_image)

    # Load models
    base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    controlnet_id = "lllyasviel/control_v11p_sd15_inpaint"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    else:
        pipe = pipe.to("cpu")

    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    out = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=gen,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
    )

    image = out.images[0]

    # Remove padding and resize back to original dimensions
    image = unletterbox_and_resize(image, content_box, original_size)

    # file name logic
    prompt_for_name = prompt
    promptCommaInd = prompt_for_name.index(",") if "," in prompt_for_name else len(prompt_for_name)
    prompt_for_name = prompt_for_name[:promptCommaInd]
    promptSplit = prompt_for_name.split(" ")

    base_name = os.path.basename(imageURL)
    name_no_ext = os.path.splitext(base_name)[0]
    prompt_joined = "_".join(promptSplit)

    new_filename = f"{name_no_ext}_cn_cc_{prompt_joined}_seed{args.seed}.png"

    parsed_url = urlparse(imageURL)
    path_segments = [segment for segment in parsed_url.path.split("/") if segment]

    try:
        container_idx = path_segments.index(container_name)
    except ValueError:
        print(f"Warning: Container name '{container_name}' not found in the URL path. Saving to 'corner_images_controlnet/default_output/'.")
        blob_output_dir_components = ["default_output"]
    else:
        blob_output_dir_components = path_segments[container_idx:-1]

        if blob_output_dir_components:
            blob_output_dir_components[-1] = "corner_cases"
        else:
            blob_output_dir_components = [container_name, "corner_cases"]

    out_root = os.path.expanduser(args.out_root)
    imageFilepath = os.path.join(out_root, *blob_output_dir_components)
    os.makedirs(imageFilepath, exist_ok=True)
    imageFilepath = os.path.join(imageFilepath, new_filename)

    image.save(imageFilepath)
    print(imageFilepath)


if __name__ == "__main__":
    main()
