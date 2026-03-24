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

import torch
from PIL import Image

from diffusers.utils import load_image

from addit_flux_pipeline import AdditFluxPipeline
from addit_flux_transformer import AdditFluxTransformer2DModel
from addit_scheduler import AdditFlowMatchEulerDiscreteScheduler
from addit_methods import add_object_real


def letterbox_to_square(image: Image.Image, target_size: int = 1024, fill=0, resample=Image.Resampling.LANCZOS):
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--imageURL", type=str, required=True)
    parser.add_argument("--container_name", type=str, required=True)

    # Add-it-specific inputs
    parser.add_argument("--prompt_source", type=str, default="A photo in an industrial warehouse")
    parser.add_argument("--subject_token", type=str, default="fire")
    parser.add_argument("--seed_obj", type=int, default=1)

    # Keep these exposed because Add-it actually uses them
    parser.add_argument("--extended_scale", type=float, default=1.1)
    parser.add_argument("--structure_transfer_step", type=int, default=4)
    parser.add_argument("--localization_model", type=str, default="attention")
    parser.add_argument("--show_attention", action="store_true")
    parser.add_argument("--disable_inversion", action="store_true")
    parser.add_argument("--use_offset", action="store_true")

    parser.add_argument("--out_root", type=str, default="corner_images_addit")

    args, _ = parser.parse_known_args(argv)

    prompt = args.prompt
    raw_imageURL = args.imageURL
    imageURL = raw_imageURL.split("?", 1)[0]
    container_name = args.container_name

    parsed_url = urlparse(imageURL)
    path_segments = [segment for segment in parsed_url.path.split("/") if segment]

    try:
        container_idx = path_segments.index(container_name)
    except ValueError:
        print(f"Warning: Container name '{container_name}' not found in the URL path. Saving to default_output.")
        blob_output_dir_components = ["default_output"]
    else:
        blob_output_dir_components = path_segments[container_idx:-1]

        if blob_output_dir_components:
            blob_output_dir_components[-1] = "corner_cases"
        else:
            blob_output_dir_components = [container_name, "corner_cases"]

    # Load original image and remember original size
    original_image = load_image(raw_imageURL).convert("RGB")
    original_size = original_image.size

    # Same letterbox pattern standard diffusion script, but 1024 for Add-it
    source_image, content_box = letterbox_to_square(
        original_image,
        target_size=1024,
        fill=(0, 0, 0),
        resample=Image.Resampling.LANCZOS,
    )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "black-forest-labs/FLUX.1-dev"

    my_transformer = AdditFluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    pipe = AdditFluxPipeline.from_pretrained(
        model_id,
        transformer=my_transformer,
        torch_dtype=dtype,
    )
    pipe.scheduler = AdditFlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()

    prompt_source = args.prompt_source
    prompt_target = f"{prompt_source}, {prompt}"

    if args.subject_token not in prompt_target:
        raise ValueError(
            f"subject_token='{args.subject_token}' must appear in prompt_target. "
            f"Include it in --prompt or change --subject_token. prompt_target={prompt_target}"
        )

    _, edited_image = add_object_real(
        pipe,
        source_image=source_image,
        prompt_source=prompt_source,
        prompt_object=prompt_target,
        subject_token=args.subject_token,
        seed_src=6311,
        seed_obj=args.seed_obj,
        extended_scale=args.extended_scale,
        structure_transfer_step=args.structure_transfer_step,
        blend_steps=[18],
        localization_model=args.localization_model,
        use_offset=args.use_offset,
        show_attention=args.show_attention,
        use_inversion=not args.disable_inversion,
        display_output=False,
    )


    edited_image = unletterbox_and_resize(edited_image, content_box, original_size)

    # File name logic
    prompt_for_name = prompt
    promptCommaInd = prompt_for_name.index(",") if "," in prompt_for_name else len(prompt_for_name)
    prompt_for_name = prompt_for_name[:promptCommaInd]
    promptSplit = prompt_for_name.split(" ")

    base_name = os.path.basename(imageURL)
    name_no_ext = os.path.splitext(base_name)[0]
    prompt_joined = "_".join(promptSplit)

    new_filename = f"{name_no_ext}_addit_cc_{prompt_joined}_seed{args.seed_obj}.png"

    imageFilepath = os.path.abspath(args.out_root)
    imageFilepath = os.path.join(imageFilepath, *blob_output_dir_components)
    os.makedirs(imageFilepath, exist_ok=True)
    imageFilepath = os.path.join(imageFilepath, new_filename)

    edited_image.save(imageFilepath)
    print(imageFilepath)


if __name__ == "__main__":
    main()