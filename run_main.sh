#!/usr/local_rwth/bin/zsh
#SBATCH --account=rwth1954
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00          # 30 Minuten z.B.
#SBATCH --job-name=roentgen
#SBATCH --output=logs/%x-%j.out  # Log-Dateien in logs/

### Module laden
module load GCCcore/.13.3.0
module load CUDA/12.8.0
module load Python/3.12.3

### In Projektordner wechseln
cd /rwthfs/rz/cluster/home/rwth1954/SDA-Bias

# virtuelle Umgebung aktivieren (muss vorher angelegt worden sein)
source .venv/bin/activate

echo "Job startet auf Host $(hostname)"
date

# Prüfen, ob HF_TOKEN gesetzt ist
if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN ist nicht gesetzt."
  echo "Bitte Job so abschicken:"
  echo "  sbatch --export=HF_TOKEN=hf_DEIN_TOKEN_HIER run_roentgen.sh"
  exit 1
fi

# Python-Skript ausführen
python main.py

date
echo "Job beendet."
