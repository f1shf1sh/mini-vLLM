#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p data
DATA_URL=https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
curl -fL --retry 2 --connect-timeout 15 "$DATA_URL" -o data/tinyshakespeare.txt
printf '%s\n' '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed  data/tinyshakespeare.txt' | sha256sum --check
