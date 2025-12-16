# main.py
import os
import re
import csv
import time
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import login
from contextlib import nullcontext
from PIL import Image


# ----------------------------
# Utilities
# ----------------------------
def slugify(s: str, max_len: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_image(img: Image.Image, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    img.save(path)


def make_grid(images: List[Image.Image], cols: int) -> Image.Image:
    if not images:
        raise ValueError("No images to make a grid.")
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img.convert("RGB"), (c * w, r * h))
    return grid


@dataclass
class DemoSpec:
    age: int
    sex: str
    race: str
    finding: str  # e.g., "Normal chest radiograph." / "Pneumonia." etc.


def build_prompt(spec: DemoSpec, style: str = "roentgen_v2") -> str:
    """
    RoentGen-v2 paper-like prompt format:
      "<AGE> year old <RACE> <SEX>. <IMPRESSION>"
    We'll keep it simple but consistent.
    """
    sex = spec.sex.strip().lower()
    race = spec.race.strip().title()
    age = int(spec.age)
    finding = spec.finding.strip()
    if not finding.endswith("."):
        finding += "."

    if style == "roentgen_v2":
        # Example: "50 year old White female. Normal chest radiograph."
        return f"{age} year old {race} {sex}. {finding}"
    else:
        # fallback
        return f"{age} year old {sex}. {finding}"


def parse_list_arg(s: str) -> List[str]:
    # accepts "a,b,c" or JSON list like '["a","b"]'
    s = s.strip()
    if s.startswith("["):
        return json.loads(s)
    return [x.strip() for x in s.split(",") if x.strip()]


# ----------------------------
# Main generation routine
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate RoentGen-v2 synthetic CXRs with demographic prompts.")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--model_id", type=str, default="stanfordmimi/RoentGen-v2", help="HF model id.")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    # Demographics & prompts
    parser.add_argument("--ages", type=str, default="50", help='Ages list, e.g. "20,40,60" or JSON list.')
    parser.add_argument("--sexes", type=str, default="female,male", help='Sex list, e.g. "female,male".')
    parser.add_argument("--races", type=str, default="White,Black,Asian,Hispanic", help='Race list.')
    parser.add_argument("--finding", type=str, default="Normal chest radiograph.", help="Finding / impression sentence.")
    parser.add_argument("--style", type=str, default="roentgen_v2", help="Prompt style.")

    # Sampling options
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="How many images per prompt.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Diffusion steps.")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for pipeline calls (1 is safest).")

    # HPC friendliness
    parser.add_argument("--limit", type=int, default=-1, help="Optional cap on number of prompts.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N prompts (useful for job arrays).")
    parser.add_argument("--save_grid", action="store_true", help="Also save a grid image per run.")
    parser.add_argument("--grid_cols", type=int, default=4, help="Grid columns if --save_grid.")
    parser.add_argument("--write_metadata", action="store_true", help="Write CSV metadata for generated images.")
    parser.add_argument("--metadata_name", type=str, default="metadata.csv", help="Metadata CSV filename.")
    args = parser.parse_args()

    # ---- HF token / login
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN is None:
        raise RuntimeError(
            "HF_TOKEN is not set. Submit the job with:\n"
            "  sbatch --export=HF_TOKEN=hf_xxx run_roentgen.sh"
        )
    login(HF_TOKEN)

    # ---- device / dtype
    if args.device == "auto":
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device = torch.device(args.device)

    if args.dtype == "auto":
        if device.type == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"[INFO] Using device={device}, dtype={dtype}", flush=True)

    # ---- load pipeline
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        token=HF_TOKEN,
    )
    pipe.to(device)

    # Optional speed/memory tweaks
    if device.type == "cuda":
        # If supported by your environment, uncomment:
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.set_progress_bar_config(disable=True)

    # ---- build prompt list (cartesian product)
    ages = [int(x) for x in parse_list_arg(args.ages)]
    sexes = parse_list_arg(args.sexes)
    races = parse_list_arg(args.races)

    prompt_specs: List[DemoSpec] = []
    for age in ages:
        for sex in sexes:
            for race in races:
                prompt_specs.append(DemoSpec(age=age, sex=sex, race=race, finding=args.finding))

    # apply offset/limit for job arrays
    prompt_specs = prompt_specs[args.offset:]
    if args.limit is not None and args.limit > 0:
        prompt_specs = prompt_specs[: args.limit]

    prompts = [build_prompt(spec, style=args.style) for spec in prompt_specs]

    print(f"[INFO] Total prompts to generate: {len(prompts)}", flush=True)
    ensure_dir(args.outdir)

    # ---- metadata CSV
    meta_path = os.path.join(args.outdir, args.metadata_name)
    meta_file = None
    meta_writer = None
    if args.write_metadata:
        meta_file = open(meta_path, "w", newline="", encoding="utf-8")
        meta_writer = csv.DictWriter(
            meta_file,
            fieldnames=[
                "timestamp",
                "model_id",
                "prompt",
                "age",
                "sex",
                "race",
                "finding",
                "seed",
                "steps",
                "guidance",
                "outfile",
            ],
        )
        meta_writer.writeheader()

    # ---- autocast context
    if device.type == "cuda":
        ctx = torch.autocast("cuda", dtype=dtype) if dtype in (torch.float16, torch.bfloat16) else nullcontext()
    else:
        ctx = nullcontext()

    # ---- generation loop (batched)
    all_images_for_grid: List[Image.Image] = []
    t0 = time.time()

    # Expand prompts to account for multiple images per prompt
    expanded: List[Dict[str, Any]] = []
    for idx, (spec, prompt) in enumerate(zip(prompt_specs, prompts)):
        for k in range(args.num_images_per_prompt):
            seed = args.seed + (idx * args.num_images_per_prompt + k)
            expanded.append({"spec": spec, "prompt": prompt, "seed": seed})

    print(f"[INFO] Total images to generate: {len(expanded)}", flush=True)

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    with ctx:
        for batch in chunks(expanded, args.batch_size):
            batch_prompts = [x["prompt"] for x in batch]
            gens = [
                torch.Generator(device=device).manual_seed(int(x["seed"]))
                if device.type == "cuda"
                else torch.Generator().manual_seed(int(x["seed"]))
                for x in batch
            ]

            # Note: Some diffusers pipelines support per-sample generators; if not, we fall back to single-item batches.
            try:
                out = pipe(
                    batch_prompts,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=gens if len(gens) > 1 else gens[0],
                )
                images = out.images
            except TypeError:
                # safest fallback: run one-by-one
                images = []
                for x in batch:
                    out = pipe(
                        x["prompt"],
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=gens[0] if isinstance(gens, list) else gens,
                    )
                    images.append(out.images[0])

            for x, img in zip(batch, images):
                spec: DemoSpec = x["spec"]
                prompt: str = x["prompt"]
                seed: int = int(x["seed"])

                fname = (
                    f"age{spec.age}_sex{slugify(spec.sex)}_race{slugify(spec.race)}_"
                    f"finding{slugify(spec.finding)}_seed{seed}.png"
                )
                outpath = os.path.join(args.outdir, fname)
                save_image(img, outpath)

                if args.save_grid:
                    all_images_for_grid.append(img)

                if meta_writer is not None:
                    meta_writer.writerow(
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "model_id": args.model_id,
                            "prompt": prompt,
                            "age": spec.age,
                            "sex": spec.sex,
                            "race": spec.race,
                            "finding": spec.finding,
                            "seed": seed,
                            "steps": args.num_inference_steps,
                            "guidance": args.guidance_scale,
                            "outfile": outpath,
                        }
                    )

            print(f"[INFO] Generated {len(batch)} images...", flush=True)

    # ---- grid
    if args.save_grid and all_images_for_grid:
        grid = make_grid(all_images_for_grid, cols=args.grid_cols)
        grid_path = os.path.join(args.outdir, "grid.png")
        save_image(grid, grid_path)
        print(f"[INFO] Saved grid: {grid_path}", flush=True)

    if meta_file is not None:
        meta_file.close()
        print(f"[INFO] Saved metadata CSV: {meta_path}", flush=True)

    dt = time.time() - t0
    print(f"[DONE] Completed in {dt:.1f}s. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
