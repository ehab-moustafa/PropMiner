#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAR_ROOM="$ROOT/DISCUSSION_WITH_AI/war-room"

mkdir -p "$WAR_ROOM"

FILES=(
  "DISCUSSION_WITH_AI/plans/02-ptr-array-grouped-gemm/AUDIT.md"
  "DISCUSSION_WITH_AI/plans/02-ptr-array-grouped-gemm/PLAN.md"
  "DISCUSSION_WITH_AI/plans/01-geforce-kernel-v2/AUDIT.md"
  "DISCUSSION_WITH_AI/plans/01-geforce-kernel-v2/MERGED_PLAN.md"
  "PROD_TO_CONFIRM.MD"
)

moved=0
skipped=0
for f in "${FILES[@]}"; do
  src="$ROOT/$f"
  if [[ -f "$src" ]]; then
    base="$(basename "$f")"
    mv "$src" "$WAR_ROOM/$base"
    echo "  moved  $f -> war-room/$base"
    ((moved++))
  else
    echo "  skip   $f (not found)"
    ((skipped++))
  fi
done

for f in "${FILES[@]}"; do
  base="$(basename "$f")"
  ln -sf "$WAR_ROOM/$base" "$ROOT/$f" 2>/dev/null || true
done

echo ""
echo "Done: $moved moved, $skipped skipped"
echo "War room: $WAR_ROOM"
