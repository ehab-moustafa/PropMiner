#!/bin/bash
# Run this on the RTX 5090 Ubuntu 24.04 machine.
set -e
apt-get update
apt-get install -y git build-essential cmake curl

git clone https://github.com/ehab-moustafa/PropMiner.git
cd PropMiner
chmod +x scripts/remote_test_kit.sh
./scripts/remote_test_kit.sh

echo "Done. Download the results/ folder and paste summary.txt back into Cursor."
