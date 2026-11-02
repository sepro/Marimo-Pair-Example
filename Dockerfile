# Dev image: Python 3.14 + uv + marimo (via uv sync) + Claude Code
FROM python:3.14-slim-trixie

# uv / uvx from the official image (pinned release)
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /usr/local/bin/

# System packages. jq + curl + bash are required by the marimo-pair skill scripts.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git curl ca-certificates sudo jq procps less \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with passwordless sudo
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid 1000 -m -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

USER vscode

# venv lives outside the bind mount; use the image's Python 3.14 instead of downloading one.
# CLAUDE_CONFIG_DIR keeps all Claude Code state (login, plugins, skills) inside the persisted volume.
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/home/vscode/.venv \
    UV_PYTHON_DOWNLOADS=never \
    CLAUDE_CONFIG_DIR=/home/vscode/.claude \
    PATH="/home/vscode/.venv/bin:/home/vscode/.local/bin:$PATH"

# Claude Code (native installer)
RUN mkdir -p /home/vscode/.claude \
    && curl -fsSL https://claude.ai/install.sh | bash

WORKDIR /workspace

# marimo editor port (forwarded to the host by VS Code)
EXPOSE 2718
