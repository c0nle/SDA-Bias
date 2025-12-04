# main.py
import os
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import login
from contextlib import nullcontext

# ==== Hugging Face Token aus Environment ====
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError(
        "HF_TOKEN is not set. Submit the job with:\n"
        "  sbatch --export=HF_TOKEN=hf_xxx run_roentgen.sh"
    )

# einmalig pro Job HF-Login
login(HF_TOKEN)

# ==== Device auswählen (GPU auf dem Cluster) ====
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

print(f"Using device: {device}, dtype: {dtype}", flush=True)

# ==== RoentGen-v2 Modell laden ====
pipe = DiffusionPipeline.from_pretrained(
    "stanfordmimi/RoentGen-v2",
    torch_dtype=dtype,
    token=HF_TOKEN,
)
pipe.to(device)

# ==== Bild generieren ====
prompt = "50 year old female. Normal chest radiograph."

if device.type == "cuda":
    ctx = torch.autocast("cuda", dtype=dtype)
else:
    ctx = nullcontext()

with ctx:
    result = pipe(prompt)

image = result.images[0]

# ==== Output speichern ====
outdir = "outputs"
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, "xray_50yo_female_normal.png")
image.save(outfile)

print(f"Done! Saved image to: {outfile}", flush=True)
