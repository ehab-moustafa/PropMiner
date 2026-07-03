#!/bin/bash
set -e

cd "$(dirname "$0")/.."

if [ ! -f "./build/propminer" ]; then
    echo "PropMiner not found. Build first:"
    echo "  ./scripts/build.sh"
    exit 1
fi

exec ./build/propminer "$@"