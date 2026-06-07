"""ADDIT + SAM2 corner-case generation entrypoint for Datara SaaS.

The DaaS backend calls this script over SSH and expects the final stdout line
to be the absolute path of the generated image.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import sys
import types
from pathlib import Path
from urllib.parse import unquote, urlparse


def _install_xformers_fallback() -> None:
    """Provide the tiny xformers API surface ADDIT expects on ARM64 installs."""
    xformers = types.ModuleType("xformers")
    xformers_ops = types.ModuleType("xformers.ops")
    xformers.__spec__ = importlib.util.spec_from_loader("xformers", None, is_package=True)
    xformers_ops.__spec__ = importlib.util.spec_from_loader("xformers.ops", None, is_package=False)

    def memory_efficient_attention(query, key, value, attn_bias=None, op=None, scale=None, **kwargs):
        import torch.nn.functional as functional

        _ = op
        return functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_bias,
            dropout_p=kwargs.get("p", 0.0),
            scale=scale,
        )

    xformers_ops.memory_efficient_attention = memory_efficient_attention
    xformers.ops = xformers_ops
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = xformers_ops


_install_xformers_fallback()


def _bootstrap_addit_path(argv: list[str] | None) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--addit_root", default=os.getenv("ADDIT_ROOT", "/home/ubuntu/packages/addit"))
    known_args, _ = parser.parse_known_args(argv)
    addit_root = os.path.abspath(os.path.expanduser(known_args.addit_root))
    if addit_root not in sys.path:
        sys.path.insert(0, addit_root)
    return addit_root


ADDIT_ROOT = _bootstrap_addit_path(sys.argv[1:])

import torch
from diffusers.utils import load_image
from PIL import Image

from addit_flux_pipeline import AdditFluxPipeline
from addit_flux_transformer import AdditFluxTransformer2DModel
from addit_methods import add_object_real
from addit_scheduler import AdditFlowMatchEulerDiscreteScheduler


STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "create",
    "generate",
    "insert",
    "make",
    "of",
    "on",
    "onto",
    "place",
    "put",
    "some",
    "the",
    "to",
    "with",
    "add",
    "adding",
    "added",
    "case",
    "corner",
    "image",
    "photo",
    "scene",
    "view",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "black",
    "white",
    "gray",
    "grey",
    "brown",
    "small",
    "large",
    "big",
    "tiny",
}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _slugify(value: str, *, max_length: int = 64, fallback: str = "corner_case") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (slug[:max_length].strip("_") or fallback)


def infer_subject_token(prompt: str) -> str:
    colon_match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_-]{0,48})\s*:", prompt)
    if colon_match:
        return colon_match.group(1).lower()

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", prompt.lower())
    for token in tokens:
        if token not in STOPWORDS:
            return token
    if tokens:
        return tokens[0]
    raise ValueError("Could not infer an ADDIT subject token from the prompt.")


def letterbox_to_square(
    image: Image.Image,
    target_size: int = 1024,
    fill: tuple[int, int, int] = (0, 0, 0),
    resample: int = Image.Resampling.LANCZOS,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), resample)

    canvas = Image.new("RGB", (target_size, target_size), fill)
    x0 = (target_size - resized_width) // 2
    y0 = (target_size - resized_height) // 2
    canvas.paste(resized, (x0, y0))
    return canvas, (x0, y0, x0 + resized_width, y0 + resized_height)


def unletterbox_and_resize(image: Image.Image, content_box: tuple[int, int, int, int], output_size: tuple[int, int]) -> Image.Image:
    cropped = image.crop(content_box)
    return cropped.resize(output_size, Image.Resampling.LANCZOS)


def build_output_path(*, out_root: str, container_name: str, image_url: str, prompt: str, subject_token: str, seed_obj: int) -> Path:
    root = Path(os.path.expanduser(out_root))
    if not root.is_absolute():
        root = Path.cwd() / root

    parsed = urlparse(image_url.split("?", 1)[0])
    source_name = unquote(os.path.basename(parsed.path)) or "source.png"
    source_stem = _slugify(os.path.splitext(source_name)[0], max_length=50, fallback="source")
    prompt_slug = _slugify(prompt, max_length=50)
    subject_slug = _slugify(subject_token, max_length=24, fallback="object")
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    filename = f"{source_stem}_addit_{subject_slug}_{prompt_slug}_{prompt_hash}_seed{seed_obj}.png"
    return (root / container_name / filename).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one ADDIT corner-case image.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--imageURL", required=True)
    parser.add_argument("--container_name", required=True)
    parser.add_argument("--prompt_source", default=os.getenv("ADDIT_PROMPT_SOURCE", "A photo in an industrial warehouse"))
    parser.add_argument("--subject_token", default="")
    parser.add_argument("--seed_src", type=int, default=6311)
    parser.add_argument("--seed_obj", type=int, default=1)
    parser.add_argument("--extended_scale", type=float, default=1.1)
    parser.add_argument("--structure_transfer_step", type=int, default=4)
    parser.add_argument("--blend_step", type=int, default=18)
    parser.add_argument("--localization_model", default=os.getenv("ADDIT_LOCALIZATION_MODEL", "attention_points_sam"))
    parser.add_argument("--show_attention", action="store_true")
    parser.add_argument("--disable_inversion", action="store_true")
    parser.add_argument("--use_offset", action="store_true")
    parser.add_argument("--out_root", default=os.getenv("ADDIT_CORNER_OUT_ROOT", "corner_images_controlnet"))
    parser.add_argument("--addit_root", default=ADDIT_ROOT)
    parser.add_argument("--model_id", default=os.getenv("ADDIT_MODEL_ID", "black-forest-labs/FLUX.1-dev"))
    parser.add_argument("--image_size", type=int, default=1024)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt = args.prompt.strip()
    if not prompt:
        raise ValueError("Missing prompt.")

    subject_token = (args.subject_token or infer_subject_token(prompt)).strip().lower()
    prompt_source = args.prompt_source.strip()
    prompt_target = f"{prompt_source}, {prompt}"
    if subject_token not in prompt_target.lower():
        raise ValueError(
            f"subject_token='{subject_token}' must appear in the effective prompt. "
            "Pass --subject_token explicitly or include it in --prompt."
        )

    _log(f"ADDIT root: {ADDIT_ROOT}")
    _log(f"Loading source image: {args.imageURL.split('?', 1)[0]}")
    original_image = load_image(args.imageURL).convert("RGB")
    original_size = original_image.size
    source_image, content_box = letterbox_to_square(original_image, target_size=args.image_size)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _log(f"Loading ADDIT model {args.model_id} with dtype={dtype}")
    transformer = AdditFluxTransformer2DModel.from_pretrained(
        args.model_id,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipe = AdditFluxPipeline.from_pretrained(
        args.model_id,
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.scheduler = AdditFlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")

    _log(
        "Running ADDIT with "
        f"subject_token={subject_token}, localization_model={args.localization_model}, seed_obj={args.seed_obj}"
    )
    with torch.inference_mode():
        _, edited_image = add_object_real(
            pipe,
            source_image=source_image,
            prompt_source=prompt_source,
            prompt_object=prompt_target,
            subject_token=subject_token,
            seed_src=args.seed_src,
            seed_obj=args.seed_obj,
            extended_scale=args.extended_scale,
            structure_transfer_step=args.structure_transfer_step,
            blend_steps=[args.blend_step],
            localization_model=args.localization_model,
            use_offset=args.use_offset,
            show_attention=args.show_attention,
            use_inversion=not args.disable_inversion,
            display_output=False,
        )

    edited_image = unletterbox_and_resize(edited_image, content_box, original_size)
    output_path = build_output_path(
        out_root=args.out_root,
        container_name=args.container_name,
        image_url=args.imageURL,
        prompt=prompt,
        subject_token=subject_token,
        seed_obj=args.seed_obj,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edited_image.save(output_path)

    print(str(output_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
