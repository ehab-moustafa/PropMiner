#!/usr/bin/env bash
# SRBMiner-style Salad bootstrap: ubuntu:24.04 + wget release tarball.
# Salad image: ubuntu:24.04
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

TAG="${PROPMINER_RELEASE_TAG:-continuous}"
REPO="${PROPMINER_RELEASE_REPO:-ehab-moustafa/PropMiner}"
ASSET="${PROPMINER_SALAD_ASSET:-PropMiner-Salad-Linux.tar.gz}"
URL="${PROPMINER_RELEASE_URL:-https://github.com/${REPO}/releases/download/${TAG}/${ASSET}}"

fail_keepalive() {
    echo "[salad] FATAL: $*" >&2
    echo "[salad] Container staying alive for logs (sleep infinity)..." >&2
    exec sleep infinity
}

apt-get update
apt-get install -y wget ca-certificates libssl3

cd /tmp
rm -rf PropMiner-Salad "${ASSET}"

echo "[salad] Downloading ${URL}"
if ! wget "${URL}" -O "${ASSET}"; then
    fail_keepalive "wget failed — is ${ASSET} published on release ${TAG}? Check https://github.com/${REPO}/releases/tag/${TAG}"
fi

if ! tar xf "${ASSET}"; then
    fail_keepalive "tar extract failed for ${ASSET}"
fi

if [[ ! -d PropMiner-Salad ]]; then
    fail_keepalive "PropMiner-Salad/ missing after extract"
fi

cd PropMiner-Salad
chmod +x run.sh propminer
echo "[salad] Starting run.sh (version=$(cat VERSION 2>/dev/null || echo unknown))"
./run.sh || fail_keepalive "run.sh exited"
