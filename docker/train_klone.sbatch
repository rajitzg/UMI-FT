#!/bin/bash
#SBATCH --job-name=umift-train
#SBATCH --account=CHANGE_ME          # your Klone condo account
#SBATCH --partition=CHANGE_ME        # GPU partition for that account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/gscratch/scrubbed/rzghosh
#SBATCH --output=logs/umift-train-%j.out
#SBATCH --error=logs/umift-train-%j.err

set -euo pipefail

# --- paths on Klone (edit if needed) ---
SIF="${SIF:-/gscratch/scrubbed/rzghosh/images/umift-train.sif}"
# Hydra task config (see PyriteML/diffusion_policy/config/task/) and any extra overrides, e.g.
#   TASK=umift_ss_v2_crop EXTRA_ARGS="logging.name=ss_v2_crop" sbatch docker/train_klone.sbatch
TASK="${TASK:-umift_zucchini}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DATA_HOST="${DATA_HOST:-/gscratch/scrubbed/rzghosh/umidata}"
CKPT_HOST="${CKPT_HOST:-/gscratch/weirdlab/rzghosh/training_outputs}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/rzghosh/wandb.env}"  # export WANDB_API_KEY=...

mkdir -p logs "${CKPT_HOST}"

module load apptainer

# Optional: confirm GPU is visible before training
apptainer exec --nv "${SIF}" python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

APPTAINER_ARGS=(
  --nv
  # Docker WORKDIR is ignored by apptainer exec; without --pwd it uses the
  # host cwd (often /gscratch/...), so relative train.py looks on the host.
  --pwd /opt/umift/PyriteML
  # zarr DirectoryStore may open with write (locks/metadata); don't use :ro
  --bind "${DATA_HOST}:/data"
  --bind "${CKPT_HOST}:/checkpoints"
  --env UMIFT_ROOT=/opt/umift
  --env PYRITE_DATASET_FOLDERS=/data
  --env PYRITE_CHECKPOINT_FOLDERS=/checkpoints
  --env HYDRA_FULL_ERROR=1
)

# Optional: overlay newer PyriteML code over the (stale) copy baked into the SIF, e.g.
#   PYRITE_OVERLAY=/gscratch/scrubbed/rzghosh/UMI-FT/PyriteML sbatch ...
if [[ -n "${PYRITE_OVERLAY:-}" ]]; then
  APPTAINER_ARGS+=(--bind "${PYRITE_OVERLAY}:/opt/umift/PyriteML")
fi

if [[ -f "${WANDB_ENV}" ]]; then
  APPTAINER_ARGS+=(--env-file "${WANDB_ENV}")
else
  echo "WARNING: ${WANDB_ENV} not found; training will use logging.mode as configured (consider offline)."
fi

apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  accelerate launch --mixed_precision=bf16 --num_processes=1 \
    /opt/umift/PyriteML/train.py --config-name=train_umift_workspace \
    task="${TASK}" \
    task.dataset_path=/data/acp_replay_buffer.zarr ${EXTRA_ARGS}
