#!/usr/bin/env bash
# One-line install (Linux/Mac):
#   curl -fsSL https://raw.githubusercontent.com/USER/waifu-nl2tags/main/install.sh | bash
set -e
REPO="${REPO:-git+https://github.com/USER/waifu-nl2tags.git}"
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip wheel
# CUDA torch first (cu124); falls back to default wheel if the index is unreachable
pip install torch --index-url https://download.pytorch.org/whl/cu124 || pip install torch
pip install "waifu-nl2tags[train] @ ${REPO}"
echo; nl2tags doctor
echo; echo "Ready. Try:  nl2tags quickstart"
