#!/usr/bin/env bash

set -euo pipefail

rm -rf /tmp/dinov2_repo
git clone --filter=blob:none https://github.com/facebookresearch/dinov2.git /tmp/dinov2_repo >/dev/null 2>&1
cd /tmp/dinov2_repo

echo "COMMITS"
git log --oneline -- dinov2/layers/attention.py | head -n 20

echo "SEARCH"
git log -S 'init_attn_std: float | None' --oneline -- dinov2/layers/attention.py | head -n 20

echo "CURRENT_SNIPPET"
sed -n '1,120p' dinov2/layers/attention.py

echo "OLD_SNIPPET"
git show ebc1cba:dinov2/layers/attention.py | sed -n '1,120p'

echo "TAGS"
git tag | tail -n 20
