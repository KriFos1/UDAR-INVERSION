#!/bin/bash
#SBATCH --job-name=udarinv-benchmark3-2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.log
# Runs or continues the chain of a config inside one allocation and resubmits itself until it is complete.
# Submit from this folder, with the partition and account of your cluster:
#   sbatch --partition=... --account=... run.sbatch [config.toml]
# Environment (use --export=ALL or set them in your profile):
#   UDARINV_FWD2D     the 2D solver build (see solver/README.md)
#   UDARINV_PYTHON    python of the environment where udar-inversion is installed (default: python)
#   UDARINV_ONEAPI    oneapi-vars.sh, if the oneAPI runtime is not already loaded
set -u
cd "$SLURM_SUBMIT_DIR"
CONFIG=${1:-config.toml}
[ -z "${UDARINV_ONEAPI:-}" ] || source "$UDARINV_ONEAPI" >/dev/null 2>&1
# Stop between points 30 minutes before the time limit; exit code 75 means "not finished yet".
"${UDARINV_PYTHON:-python}" -u -m udarinv.cli run "$CONFIG" --seconds 19800
status=$?
if [ "$status" -eq 75 ]; then
  sbatch --dependency="afterok:$SLURM_JOB_ID" run.sbatch "$CONFIG"
  exit 0
fi
exit "$status"
