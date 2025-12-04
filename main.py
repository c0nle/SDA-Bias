import os
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import login
from contextlib import nullcontext

# === Hugging Face Token aus Environment lesen ===
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("Please set the HF_TOKEN environment variable with your Hugging Face access token.")

# Einmalig bei Jobstart einloggen
login(HF_TOKEN)

# === Device auswählen (GPU, sonst CPU) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

print(f"Using device: {device}, dtype: {dtype}")

# === RoentGen-v2 Modell laden ===
pipe = DiffusionPipeline.from_pretrained(
    "stanfordmimi/RoentGen-v2",
    torch_dtype=dtype,
    token=HF_TOKEN,
)
pipe.to(device)

# === Bild generieren ===
prompt = "50 year old female. Normal chest radiograph."

if device.type == "cuda":
    autocast_ctx = torch.autocast("cuda", dtype=dtype)
else:
    autocast_ctx = nullcontext()

with autocast_ctx:
    result = pipe(prompt)

image = result.images[0]

# === Output speichern ===
outdir = "outputs"
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, "xray_50yo_female_normal.png")
image.save(outfile)

print(f"Done! Saved image to: {outfile}")
