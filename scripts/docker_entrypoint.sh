#!/usr/bin/env bash
# Route container start by PROPMINER_MODE:
#   test      — quick self-test then exit (default)
#   full      — self-test + 60s benchmark (+ optional sweep if enabled)
#   mine      — connect to pool and mine until stopped
set -euo pipefail

MODE="${PROPMINER_MODE:-test}"

case "${MODE}" in
    mine|mining)
        exec ./scripts/run_mining.sh "$@"
        ;;
    full|benchmark)
        export PROPMINER_QUICK_EXIT=0
        export PROPMINER_SKIP_BENCH="${PROPMINER_SKIP_BENCH:-0}"
        export PROPMINER_SKIP_SWEEP="${PROPMINER_SKIP_SWEEP:-1}"
        export PROPMINER_SKIP_NCU="${PROPMINER_SKIP_NCU:-1}"
        exec ./scripts/remote_test_kit.sh
        ;;
    test|*)
        exec ./scripts/remote_test_kit.sh
        ;;
esac
