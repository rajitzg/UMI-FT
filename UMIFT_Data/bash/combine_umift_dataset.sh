#!/usr/bin/env bash
#
# Whole-dataset stages: combine every unified zarr under a folder into one
# UMIFT-schema zarr, then run virtual-target postprocessing.
#
# Safe to re-run any time more trajectories have been added -- it always
# rebuilds the combined zarr from everything currently under <folder>/zarr/
# (old + newly-arrived trajectories together), and virtual-target
# postprocessing skips episodes it already labeled. Each run IS a full
# rebuild of the combined zarr though, not an incremental append, so cost
# scales with total accumulated episode count, not just what's new.
#
# Run this by hand once a session (or several) is done. It is NOT triggered
# automatically per trajectory by frame_ingest_server.py -- see
# process_local_recordings.sh for the per-trajectory stages that are.
#
# Usage:
#   bash UMI-FT/UMIFT_Data/bash/combine_umift_dataset.sh <folder_name>
#   bash UMI-FT/UMIFT_Data/bash/combine_umift_dataset.sh <folder_name> --pose-source tag
#   bash UMI-FT/UMIFT_Data/bash/combine_umift_dataset.sh <folder_name> --output acp_replay_buffer_tag.zarr
#
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") <folder_name> [options]

  folder_name  Subfolder under \${LOCAL_FT_ROOT} (default: /mnt/data/umidata/ft)
               with unified zarrs under <folder>/zarr/.

Options:
  --pose-source {auto,t265,tag}
                   Pose for ts_pose_fb_0 (default: auto).
                   auto = tag_traj sidecar when present, else T265.
                   tag  = require sidecars for every episode.
                   t265 = ignore sidecars, always use T265 odom.
  --output NAME    Output zarr name or path (default: acp_replay_buffer.zarr).
                   Relative names land under <folder>/; absolute paths used as-is.
  --start-pad-seconds N
                   Forwarded to the converter's --start-pad-seconds (default: 5.0).
                   Seconds of repeated first-frame data prepended to every
                   episode so training can sample the true reset state. 0 disables.
  --dagger-pattern SUBSTR
                   Tag episodes whose unified zarr filename contains SUBSTR as
                   source=dagger (rest = original). Training then weights them
                   via task.dataset.dagger_frac.
  -h, --help       Show this help.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

FOLDER_NAME="$1"
shift
POSE_SOURCE="auto"
OUTPUT_NAME="acp_replay_buffer.zarr"
START_PAD_SECONDS=""
DAGGER_PATTERN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pose-source)
            if [[ $# -lt 2 ]]; then
                echo "Error: --pose-source requires a value (auto|t265|tag)" >&2
                exit 1
            fi
            POSE_SOURCE="$2"
            shift
            ;;
        --pose-source=*)
            POSE_SOURCE="${1#*=}"
            ;;
        --output)
            if [[ $# -lt 2 ]]; then
                echo "Error: --output requires a zarr name or path" >&2
                exit 1
            fi
            OUTPUT_NAME="$2"
            shift
            ;;
        --output=*)
            OUTPUT_NAME="${1#*=}"
            ;;
        --start-pad-seconds)
            if [[ $# -lt 2 ]]; then
                echo "Error: --start-pad-seconds requires a value" >&2
                exit 1
            fi
            START_PAD_SECONDS="$2"
            shift
            ;;
        --start-pad-seconds=*)
            START_PAD_SECONDS="${1#*=}"
            ;;
        --dagger-pattern)
            if [[ $# -lt 2 ]]; then
                echo "Error: --dagger-pattern requires a value" >&2
                exit 1
            fi
            DAGGER_PATTERN="$2"
            shift
            ;;
        --dagger-pattern=*)
            DAGGER_PATTERN="${1#*=}"
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

case "${POSE_SOURCE}" in
    auto|t265|tag) ;;
    *)
        echo "Error: --pose-source must be auto, t265, or tag (got: ${POSE_SOURCE})" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_pipeline_common.sh"

LOCAL_DIR="${LOCAL_FT_ROOT}/${FOLDER_NAME}"
UNIFIED_ZARR_DIR="${LOCAL_DIR}/zarr"
if [[ "${OUTPUT_NAME}" = /* ]]; then
    UMIFT_ZARR="${OUTPUT_NAME}"
else
    UMIFT_ZARR="${LOCAL_DIR}/${OUTPUT_NAME}"
fi

if [[ ! -d "${UNIFIED_ZARR_DIR}" ]] || [[ -z "$(ls -A "${UNIFIED_ZARR_DIR}" 2>/dev/null || true)" ]]; then
    echo "Error: no unified zarr output found under ${UNIFIED_ZARR_DIR}" >&2
    echo "Run process_local_recordings.sh (or process_ft_wild_recordings.sh) first." >&2
    exit 1
fi

activate_conda
export PYTHONPATH="${WS_ROOT}/UMI-FT/PyriteUtility:${PYTHONPATH:-}"

CONVERT_EXTRA_ARGS=()
if [[ -n "${START_PAD_SECONDS}" ]]; then
    CONVERT_EXTRA_ARGS+=(--start-pad-seconds "${START_PAD_SECONDS}")
fi

if [[ -n "${DAGGER_PATTERN}" ]]; then
    CONVERT_EXTRA_ARGS+=(--dagger-pattern "${DAGGER_PATTERN}")
fi

log "Converting unified zarr -> UMIFT schema (conda env: ${CONDA_ENV}, pose-source=${POSE_SOURCE}) ..."
python "${CONVERT_UMIFT_SCRIPT}" \
    --input "${UNIFIED_ZARR_DIR}" \
    --output "${UMIFT_ZARR}" \
    --pose-source "${POSE_SOURCE}" \
    "${CONVERT_EXTRA_ARGS[@]}"

log "Running virtual-target postprocessing on ${UMIFT_ZARR} ..."
export UMIFT_DATASET_PATH="${UMIFT_ZARR}"
export PYRITE_DATASET_FOLDERS="${LOCAL_DIR}"
python "${POSTPROCESS_SCRIPT}"

log "Done."
log "  Unified zarr: ${UNIFIED_ZARR_DIR}/"
log "  UMIFT zarr:   ${UMIFT_ZARR}"
log "  Pose source:  ${POSE_SOURCE}"
