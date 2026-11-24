#!/bin/bash
# Fetch benchmark folders of https://gitlab.norceresearch.no/krfo/udar-benchmarks into data/udar-benchmarks.
# Usage: experiments/fetch_data.sh [Benchmark-3 Benchmark-2 ...]     (default: Benchmark-2 and Benchmark-3)
# Only the named folders are checked out; Benchmark-2 needs about 2.5 GB of disk.
# UDARINV_DATA=/some/folder puts the data elsewhere (then point [data] path of the configs there).
set -euo pipefail
cd "$(dirname "$0")/.."
target=${UDARINV_DATA:-data/udar-benchmarks}
folders=("$@")
[ ${#folders[@]} -gt 0 ] || folders=(Benchmark-2 Benchmark-3)
if [ ! -d "$target/.git" ]; then
  mkdir -p "$(dirname "$target")"
  git clone --filter=blob:none --no-checkout https://gitlab.norceresearch.no/krfo/udar-benchmarks.git "$target"
  git -C "$target" sparse-checkout init --cone
fi
git -C "$target" sparse-checkout set "${folders[@]}"
git -C "$target" checkout main
git -C "$target" pull --ff-only
echo "Benchmarks in $target: $(ls "$target" | tr '\n' ' ')"
