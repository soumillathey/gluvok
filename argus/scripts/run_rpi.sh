#!/usr/bin/env bash
# ==============================================================================
# run_rpi.sh - Start Argus ANPR FastAPI Server on Raspberry Pi
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "============================================================"
echo " Starting Argus ANPR Microservice on Raspberry Pi"
echo " Working directory: $DIR"
echo "============================================================"

# Check if uv is available, otherwise use python3
if command -v uv >/dev/null 2>&1; then
    exec uv run python main.py --server --host 0.0.0.0 --port 8000
elif [ -f "$DIR/.venv/bin/python" ]; then
    exec "$DIR/.venv/bin/python" main.py --server --host 0.0.0.0 --port 8000
else
    exec python3 main.py --server --host 0.0.0.0 --port 8000
fi
