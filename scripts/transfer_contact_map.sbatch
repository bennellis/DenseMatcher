#!/bin/bash
#SBATCH --job-name=transfer_contact_map
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=4g
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j-transfer_contact_map.out

# Usage:
#   sbatch transfer_contact_map.sbatch \
#     --contact-npz  <path/to/sequence.npz> \
#     --source-asset <path/to/converted_new/SOURCE_UID> \
#     --target-asset <path/to/converted_new/TARGET_UID> \
#     [--output       <path/to/output.npz>]          # inferred if omitted
#     [--correspondence <path/to/cache.npz>]          # inferred if omitted
#     [--map-method   hungarian|inverse|...]          # default: hungarian
#
# Inferred defaults (when not specified):
#   --output        <npz_dir>/<npz_stem>_transferred_<target_uid>.npz
#   --correspondence <npz_dir>/src2tgt_<source_uid>_to_<target_uid>.npz
#
# The script checks whether the correspondence cache already exists.
# If it does, it loads it (no DenseMatcher GPU pass needed).
# If not, it runs DenseMatcher and saves the result for future reuse.

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CONTACT_NPZ=""
SOURCE_ASSET=""
TARGET_ASSET=""
OUTPUT_NPZ=""
CORRESPONDENCE=""
MAP_METHOD="hungarian"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --contact-npz)    CONTACT_NPZ="$2";    shift 2 ;;
        --source-asset)   SOURCE_ASSET="$2";   shift 2 ;;
        --target-asset)   TARGET_ASSET="$2";   shift 2 ;;
        --output)         OUTPUT_NPZ="$2";     shift 2 ;;
        --correspondence) CORRESPONDENCE="$2"; shift 2 ;;
        --map-method)     MAP_METHOD="$2";     shift 2 ;;
        *)                EXTRA_ARGS+=("$1");  shift   ;;
    esac
done

if [[ -z "$CONTACT_NPZ" || -z "$SOURCE_ASSET" || -z "$TARGET_ASSET" ]]; then
    echo "ERROR: --contact-npz, --source-asset, and --target-asset are required."
    echo ""
    echo "Usage:"
    echo "  sbatch $(basename "$0") \\"
    echo "    --contact-npz  <path/to/sequence.npz> \\"
    echo "    --source-asset <path/to/converted_new/SOURCE_UID> \\"
    echo "    --target-asset <path/to/converted_new/TARGET_UID> \\"
    echo "    [--output <path/to/output.npz>] \\"
    echo "    [--correspondence <path/to/cache.npz>] \\"
    echo "    [--map-method hungarian]"
    exit 1
fi

# ---------------------------------------------------------------------------
# Infer paths
# ---------------------------------------------------------------------------
SOURCE_UID=$(basename "${SOURCE_ASSET%/}")
TARGET_UID=$(basename "${TARGET_ASSET%/}")
NPZ_DIR=$(dirname "$CONTACT_NPZ")
NPZ_STEM=$(basename "$CONTACT_NPZ" .npz)

if [[ -z "$OUTPUT_NPZ" ]]; then
    OUTPUT_NPZ="${NPZ_DIR}/${NPZ_STEM}_transferred_${TARGET_UID}.npz"
fi

if [[ -z "$CORRESPONDENCE" ]]; then
    CORRESPONDENCE="${NPZ_DIR}/src2tgt_${SOURCE_UID}_to_${TARGET_UID}.npz"
fi

SCRIPT="/cluster/home/beellis/dexterous_manipulation/DenseMatcher/scripts/transfer_full_contact_map.py"
AGGRE_WEIGHTS="/cluster/home/beellis/dexterous_manipulation/DenseMatcher/checkpoints/SDDINO_weights"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# module load stack/2025-06 gcc/8.5.0 cuda/11.8.0
module load eth_proxy

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate densematcher_3.10

cd /cluster/home/beellis/dexterous_manipulation/DenseMatcher

# ---------------------------------------------------------------------------
# Resolve correspondence args
# ---------------------------------------------------------------------------
if [[ -f "$CORRESPONDENCE" ]]; then
    echo "[cache] Found cached correspondence — skipping DenseMatcher pass."
    CORR_ARGS=(--load-correspondence "$CORRESPONDENCE")
else
    echo "[cache] No cached correspondence found — will compute and save."
    CORR_ARGS=(--save-correspondence "$CORRESPONDENCE")
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "------------------------------------------------------------"
echo "contact-npz:    $CONTACT_NPZ"
echo "source-asset:   $SOURCE_ASSET  (uid: $SOURCE_UID)"
echo "target-asset:   $TARGET_ASSET  (uid: $TARGET_UID)"
echo "output:         $OUTPUT_NPZ"
echo "correspondence: $CORRESPONDENCE"
echo "map-method:     $MAP_METHOD"
echo "------------------------------------------------------------"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python "$SCRIPT" \
    --contact-npz    "$CONTACT_NPZ" \
    --source-asset   "$SOURCE_ASSET" \
    --target-asset   "$TARGET_ASSET" \
    --output         "$OUTPUT_NPZ" \
    --aggre-weights  "$AGGRE_WEIGHTS" \
    --map-method     "$MAP_METHOD" \
    "${CORR_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
