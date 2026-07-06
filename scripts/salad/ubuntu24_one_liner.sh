#!/usr/bin/env bash
# SRBMiner-style Salad bootstrap: ubuntu:24.04 + wget release tarball.
# Salad image: ubuntu:24.04
# Salad command: bash -c "$(curl -fsSL https://raw.githubusercontent.com/ehab-moustafa/PropMiner/master/scripts/salad/ubuntu24_one_liner.sh)"
#
# Or paste ubuntu24_one_liner.sh contents directly into Salad "Command".
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

TAG="${PROPMINER_RELEASE_TAG:-continuous}"
REPO="${PROPMINER_RELEASE_REPO:-ehab-moustafa/PropMiner}"
ASSET="${PROPMINER_SALAD_ASSET:-PropMiner-Salad-Linux.tar.gz}"
URL="${PROPMINER_RELEASE_URL:-https://github.com/${REPO}/releases/download/${TAG}/${ASSET}}"

apt-get update
apt-get install -y wget ca-certificates libssl3

cd /tmp
rm -rf PropMiner-Salad "${ASSET}"
wget -q "${URL}" -O "${ASSET}"
tar xf "${ASSET}"
cd PropMiner-Salad
chmod +x run.sh propminer
exec ./run.sh
