#!/usr/bin/env bash
# Start the control plane. Override with PM_* env vars (see README).
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
exec .venv/bin/uvicorn control_plane.app:app --host "${PM_HOST:-0.0.0.0}" --port "${PM_PORT:-8100}"
