#!/usr/bin/env bash
# Route container start by PROPMINER_MODE:
#   full      — self-test + 120s benchmark (default for zero-config Salad validation)
#   test      — quick self-test only
#   mine      — connect to pool and mine until stopped (requires PROPMINER_WALLET)
set -euo pipefail

MODE_RAW="${PROPMINER_MODE:-full}"
MODE="$(printf '%s' "${MODE_RAW}" | tr '[:upper:]' '[:lower:]')"

echo "[entrypoint] PROPMINER_MODE=${MODE_RAW} -> ${MODE}" >&2

case "${MODE}" in
    mine|mining)
        exec ./scripts/run_mining.sh "$@"
        ;;
    full|benchmark)
        export PROPMINER_QUICK_EXIT=0
        export PROPMINER_SKIP_BENCH=0
        export PROPMINER_SKIP_SWEEP=1
        export PROPMINER_SKIP_NCU=1
        exec ./scripts/remote_test_kit.sh
        ;;
    test)
        export PROPMINER_QUICK_EXIT=1
        export PROPMINER_SKIP_BENCH=1
        export PROPMINER_SKIP_SWEEP=1
        export PROPMINER_SKIP_NCU=1
        exec ./scripts/remote_test_kit.sh
        ;;
    *)
        echo "[entrypoint] ERROR: unknown PROPMINER_MODE='${MODE_RAW}'." >&2
        echo "[entrypoint] Valid values: full, test, mine (or mining)." >&2
        exit 1
        ;;
esac
