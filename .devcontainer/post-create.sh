#!/usr/bin/env bash
# Runs once after the container is created.
set -euo pipefail

echo ">> Fixing ownership of the Claude Code volume"
sudo chown -R vscode:vscode /home/vscode/.claude

echo ">> Installing Python dependencies (uv sync)"
uv sync

echo ">> Downloading demo data"
uv run python scripts/download_data.py || echo "!! Data download failed; rerun: uv run python scripts/download_data.py"

echo ">> Installing the marimo-pair Claude Code plugin"
claude plugin marketplace add marimo-team/marimo-pair \
  || echo "!! marketplace add failed (maybe already added)"
claude plugin install marimo-pair@marimo-pair \
  || echo "!! plugin install failed; inside claude run: /plugin install marimo-pair@marimo-pair"

echo ">> Done. Start marimo with: bash scripts/start-marimo.sh"
