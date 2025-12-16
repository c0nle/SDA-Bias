#!/usr/bin/env bash
#SBATCH --job-name=roentgen_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-9

set -euo pipefail

echo "=== Array task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date) ==="

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Submit with: sbatch --export=HF_TOKEN=hf_xxx run_roentgen_array.sh"
  exit 1
fi

mkdir -p logs outputs

# --- Environment (anpassen) ---
# source ~/.bashrc
# conda activate roentgen
# oder: source /path/to/venv/bin/activate

python -V
nvidia-smi || true

# --- Chunking Parameter ---
# Jeder Array-Task macht CHUNK_SIZE Bilder (genauer: Prompts) ab offset
CHUNK_SIZE=50
OFFSET=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))

OUTDIR="outputs/arr_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}"

python main.py \
  --outdir "${OUTDIR}" \
  --ages "20,50,80" \
  --sexes "female,male" \
  --races "White,Black,Asian,Hispanic" \
  --finding "Normal chest radiograph." \
  --num_images_per_prompt 1 \
  --num_inference_steps 30 \
  --guidance_scale 3.5 \
  --batch_size 1 \
  --offset "${OFFSET}" \
  --limit "${CHUNK_SIZE}" \
  --write_metadata

echo "=== Array task ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
