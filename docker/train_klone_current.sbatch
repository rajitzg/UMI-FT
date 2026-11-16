#!/bin/bash
#SBATCH --job-name=s_v3.5_hist
#SBATCH --account=weirdlab          # your Klone condo account
#SBATCH --partition=gpu-l40s       # GPU partition for that account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="l40|l40s|a40|a100"
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --chdir=/gscratch/scrubbed/rzghosh
#SBATCH --output=/gscratch/weirdlab/rzghosh/layup/logs/s_v3.5_hist/%j.out
#SBATCH --error=/gscratch/weirdlab/rzghosh/layup/logs/s_v3.5_hist/%j.err

set -euo pipefail

# --- paths on Klone (edit if needed) ---
SIF="${SIF:-/gscratch/weirdlab/rzghosh/layup/images/umift-train_v6.sif}"
DATA_HOST="${DATA_HOST:-/gscratch/scrubbed/rzghosh/umidata/stringer_v3.5/og}"
CKPT_HOST="${CKPT_HOST:-/gscratch/weirdlab/rzghosh/layup/training_outputs/stringer_v3.5_hist}"
WANDB_ENV="${WANDB_ENV:-/gscratch/weirdlab/rzghosh/layup/wandb.env}"  # export WANDB_API_KEY=...

mkdir -p "${CKPT_HOST}"

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

if [[ -f "${WANDB_ENV}" ]]; then
  APPTAINER_ARGS+=(--env-file "${WANDB_ENV}")
else
  echo "WARNING: ${WANDB_ENV} not found; training will use logging.mode as configured (consider offline)."
fi

apptainer exec "${APPTAINER_ARGS[@]}" "${SIF}" \
  accelerate launch --mixed_precision=bf16 --num_processes=1 \
    /opt/umift/PyriteML/train.py --config-name=train_umift_workspace \
    task=umift_ss_v2_no_scan \
    dataloader.batch_size=64 \
    dataloader.prefetch_factor=8 \
    dataloader.num_workers=32 \
    val_dataloader.batch_size=64 \
    val_dataloader.shuffle=True \
    training.num_epochs=200 \
    training.checkpoint_every=5 \
    task.sparse_obs_rgb_down_sample_steps=5 \
    task.sparse_obs_rgb_horizon=10 \
    task.sparse_obs_ultrawide_down_sample_steps=5 \
    task.sparse_obs_ultrawide_horizon=10 \
    task.sparse_obs_depth_down_sample_steps=5 \
    task.sparse_obs_depth_horizon=10 \
    task.sparse_obs_scan_heatmap_down_sample_steps=1 \
    task.sparse_obs_scan_heatmap_horizon=1 \
    task.sparse_obs_low_dim_down_sample_steps=16 \
    task.sparse_obs_low_dim_horizon=50 \
    task.sparse_obs_gripper_down_sample_steps=1 \
    task.sparse_obs_gripper_horizon=1 \
    task.sparse_obs_wrench_down_sample_steps=2 \
    task.sparse_obs_wrench_horizon=64 \
    task.sparse_action_down_sample_steps=1 \
    task.sparse_action_horizon=24 \
    task.dataset_path=/data/acp_replay_buffer.zarr \
    logging.name=stringer-v3.5-hist \
    hydra.run.dir=/checkpoints/stringer-v3.5-hist

