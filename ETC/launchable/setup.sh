#!/bin/bash
set -euo pipefail
bootstrap_file="$(mktemp -t ai4sci-bootstrap.XXXXXX)"
curl --fail --silent --show-error --location --retry 3 \
  https://raw.githubusercontent.com/yang926/AI4Sci-PhysicsNeMo-Bootcamp/main/ETC/launchable/bootstrap.py \
  --output "$bootstrap_file"
python3 "$bootstrap_file" --launchable
