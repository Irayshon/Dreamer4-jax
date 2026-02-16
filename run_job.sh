#!/bin/bash
#SBATCH --partition=dgx-b200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out

set -euo pipefail

cd ~/projects/tiny_dreamer_4
source .venv/bin/activate

# Default profile can be switched to production for full runs.
PROFILE_PATH="configs/profiles/quick_test.yaml"

python -m dreamer.pipeline run --config "${PROFILE_PATH}"

