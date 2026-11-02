# marimo-pair

Pair-program with Claude Code on [marimo](https://marimo.io) notebooks inside a VS Code dev container, while you watch the notebook live in Chrome on your own machine.

Stack: Python 3.14, uv, marimo, polars, altair, numpy, scipy, VS Code dev container, Claude Code + [marimo-pair](https://github.com/marimo-team/marimo-pair) skill.

## Requirements

- Docker Desktop (running)
- VS Code with the **Dev Containers** extension (`ms-vscode-remote.remote-containers`)
- A Claude account (Pro/Max/Team/Enterprise) or Console account for Claude Code
- Google Chrome (or any browser) on the host

## Setup

1. Open this folder in VS Code.
2. Click **Reopen in Container**, or run `Dev Containers: Reopen in Container` from the Command Palette (`Ctrl+Shift+P`).
3. Wait for the first build (a few minutes). `post-create.sh` then automatically:
   - runs `uv sync` (creates the venv in `/home/vscode/.venv`),
   - downloads the Palmer Penguins dataset to `data/penguins.csv`,
   - installs the `marimo-pair` Claude Code plugin.
4. Commit the generated `uv.lock`.

## Work with Claude on a notebook

You need two terminals in VS Code (both run inside the container).

**Terminal 1: start marimo**

```bash
bash scripts/start-marimo.sh                         # opens the notebooks/ folder
bash scripts/start-marimo.sh notebooks/penguins.py   # or one specific notebook
```

(Alternatively: `Terminal → Run Task… → marimo: start editor (port 2718)`.)

**Open it in Chrome on your host:** go to <http://localhost:2718>.
VS Code forwards port 2718 from the container to your machine (see the **Ports** tab). Keep VS Code connected to the container while you work; the forward closes when the window does.

**Terminal 2: start Claude Code**

```bash
claude
```

The first time, log in when prompted. The login is stored in a Docker volume, so it survives container rebuilds.

Then ask Claude to pair with you, for example:

```text
/marimo-pair Open notebooks/penguins.py and add a cell that compares bill depth between islands with a boxplot.
```

Claude discovers the running marimo server, runs code inside the live kernel and adds or edits cells. You see the changes appear in Chrome as it works. Keep reviewing and steering in the browser while Claude edits.

Tips:

- marimo runs with `--no-token` so the skill can find it automatically. Don't publish port 2718 beyond localhost.
- If Claude keeps asking to approve the skill's Bash scripts, choose "always allow" for them, or add them to `.claude/settings.json`, as described in the [marimo-pair FAQ](https://github.com/marimo-team/marimo-pair#faq).
- After a session, `/retro-marimo-pair` sends feedback to the marimo team.

## Other commands

```bash
uv add <package>                          # add a dependency
uv run pytest                             # run tests
uv run ruff check . && uv run ruff format .
uv run marimo new                         # create a new notebook
uv run python scripts/download_data.py --force   # re-download the demo data
```

## Troubleshooting

- **Skill not available in Claude:** run `/plugin` in Claude and check that `marimo-pair` is installed. If it isn't: `/plugin marketplace add marimo-team/marimo-pair`, then `/plugin install marimo-pair@marimo-pair`.
- **`localhost:2718` doesn't load in Chrome:** open the **Ports** tab in VS Code and check that 2718 is forwarded. If another program on the host uses 2718, stop it (the config needs that exact local port).
- **No data in the notebook:** `uv run python scripts/download_data.py`.

## Layout

```
.devcontainer/        devcontainer.json + post-create.sh
.claude/settings.json project-level Claude Code settings (enables marimo-pair)
data/                 downloaded demo data (penguins.csv)
notebooks/penguins.py example marimo notebook
scripts/              download_data.py, start-marimo.sh
tests/                pytest tests
Dockerfile            dev image (python:3.14-slim-trixie + uv + Claude Code)
pyproject.toml        project + dependencies (uv)
```
