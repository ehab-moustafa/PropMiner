#!/usr/bin/env bash
# Move entire plan folders into war-room, with symlinks in place.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAR_ROOM="$ROOT/DISCUSSION_WITH_AI/war-room"

mkdir -p "$WAR_ROOM"

# All plan folders (excluding empty/minimal ones and the main tmad-dominance)
PLAN_FOLDERS=(
  "00-comprehensive-integration"
  "00-core-analysis"
  "01-geforce-kernel-v2"
  "02-ptr-array-grouped-gemm"
  "03-stream-split-pregemm"
  "04-triple-gpu-half-buffer"
  "05-fuse-noise-noisinga-gemm"
  "06-sigma-install-b-hash-batching"
  "07-cccl-share-compaction"
  "08-consumer-tma-legacy"
  "09-sm120-upcast-hack-comprehensive-analysis"
)

moved=0
skipped=0
for folder in "${PLAN_FOLDERS[@]}"; do
  src="$ROOT/DISCUSSION_WITH_AI/plans/$folder"
  if [[ -d "$src" ]]; then
    count=$(find "$src" -type f | wc -l | tr -d ' ')
    if [[ "$count" -eq 0 ]]; then
      echo "  skip   $folder (empty)"
      ((skipped++))
      continue
    fi
    mv "$src" "$WAR_ROOM/$folder"
    ln -sf "$WAR_ROOM/$folder" "$ROOT/DISCUSSION_WITH_AI/plans/$folder"
    echo "  moved  plans/$folder -> war-room/$folder (symlinked)"
    ((moved++))
  else
    echo "  skip   $folder (not found)"
    ((skipped++))
  fi
done

echo ""
echo "Done: $moved moved, $skipped skipped"
echo "War room: $WAR_ROOM"
