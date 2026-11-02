#!/usr/bin/env bash
# Start the marimo editor so that:
#  - Chrome on the host can open it at http://localhost:2718 (VS Code forwards the port)
#  - Claude Code's marimo-pair skill can auto-discover it (--no-token)
set -euo pipefail
cd "$(dirname "$0")/.."

NOTEBOOK="${1:-notebooks/}"

exec uv run marimo edit "$NOTEBOOK" \
  --host 0.0.0.0 \
  --port 2718 \
  --headless \
  --no-token
