#!/bin/bash
set -e

cd "$(dirname "$0")/.."

if [ ! -f "./build/propminer" ]; then
    echo "PropMiner not found. Build first:"
    echo "  ./scripts/build.sh          # sm_120a / blackwell (RTX 5090)"
    echo "  ./scripts/build_and_benchmark.sh"
    exit 1
fi

exec ./build/propminer "$@"