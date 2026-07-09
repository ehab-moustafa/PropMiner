#!/bin/bash
# ============================================================
# 00-tmad-dominance — Organize all subagent analysis docs
# ============================================================
# Moves every analysis doc from across the project into this
# single folder with a clean subdirectory structure.
#
# Run from the PropMiner root:
#   bash DISCUSSION_WITH_AI/plans/00-tmad-dominance/organize.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Destination
DEST="$SCRIPT_DIR"

# Helper: move a file if it exists, skip otherwise
move_file() {
    local src="$1"
    local dst_dir="$2"
    local basename=$(basename "$src")

    if [ -f "$src" ]; then
        mkdir -p "$dst_dir"
        mv "$src" "$dst_dir/$basename"
        echo "  ✓ $src → $dst_dir/$basename"
    else
        echo "  ⊘ $src (not found, skipping)"
    fi
}

echo "============================================"
echo "  00-tmad-dominance — Organizing Analysis"
echo "============================================"
echo ""

# 00-core-analysis
echo "[1/7] Core Algorithm Analysis"
move_file \
    "$PROJECT_ROOT/DISCUSSION_WITH_AI/plans/00-core-analysis/ANALYSIS.md" \
    "$DEST/00-core-analysis"
move_file \
    "$PROJECT_ROOT/DISCUSSION_WITH_AI/plans/00-core-analysis/PLAN.md" \
    "$DEST/00-core-analysis" 2>/dev/null || true

# 01-competitor-analysis
echo "[2/7] Competitor Analysis"
move_file \
    "$PROJECT_ROOT/research/03-competitor-analysis/COMPETITOR_ANALYSIS.md" \
    "$DEST/01-competitor-analysis"

# 02-5090-architecture
echo "[3/7] 5090 Architecture"
move_file \
    "$PROJECT_ROOT/docs/RTX5090_GB202_ARCHITECTURE_DEEP_DIVE.md" \
    "$DEST/02-5090-architecture"

# 03-consensus-analysis
echo "[4/7] Pearl Consensus & cuPOW"
move_file \
    "$PROJECT_ROOT/docs/CUPOW_CONSENSUS_ANALYSIS.md" \
    "$DEST/03-consensus-analysis"

# 04-cuda-kernel
echo "[5/7] CUDA Kernel Optimization"
move_file \
    "$PROJECT_ROOT/docs/CUDA_KERNEL_OPTIMIZATIONS.md" \
    "$DEST/04-cuda-kernel"

# 05-systems-bottlenecks
echo "[6/7] Memory & PCIe Bottlenecks"
move_file \
    "$PROJECT_ROOT/docs/SYSTEMS_BOTTLENECKS_ANALYSIS.md" \
    "$DEST/05-systems-bottlenecks"

# 06-integration-plan
echo "[7/7] Comprehensive Integration Plan"
move_file \
    "$PROJECT_ROOT/DISCUSSION_WITH_AI/plans/00-comprehensive-integration/PLAN.md" \
    "$DEST/06-integration-plan"

# Master document
echo "[master] TMAD Mastery Master Analysis"
move_file \
    "$PROJECT_ROOT/docs/TMAD_MASTERY_COMPREHENSIVE_ANALYSIS.md" \
    "$DEST"

echo ""
echo "============================================"
echo "  Done. Structure:"
echo "============================================"
find "$DEST" -type f -name "*.md" | sort | sed 's|'"$DEST"'/||'
