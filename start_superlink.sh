#!/usr/bin/env bash
# Run this script on the SuperLink machine.
# It starts the SuperLink with two listening ports:
#   - 9092  Fleet API   (SuperNodes connect here)
#   - 9093  ServerApp I/O API  (flwr run connects here)
#
# Usage:
#   bash start_superlink.sh

set -euo pipefail

# Locate flower-superlink: first try PATH, then look next to the active Python
# (covers the common case where flwr is installed in a venv that is not activated).
if command -v flower-superlink &>/dev/null; then
    SUPERLINK_BIN="flower-superlink"
else
    _PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
    if [ -z "$_PY" ]; then
        echo "Error: no Python interpreter found in PATH." >&2
        exit 1
    fi
    _CANDIDATE="$(dirname "$_PY")/flower-superlink"
    if [ -x "$_CANDIDATE" ]; then
        SUPERLINK_BIN="$_CANDIDATE"
    else
        echo "Error: 'flower-superlink' not found." >&2
        echo "Activate the virtual environment where 'flwr' is installed, or run:" >&2
        echo "  pip install 'flwr[simulation]>=1.15,<2.0'" >&2
        exit 1
    fi
fi

echo "Using: $SUPERLINK_BIN"
echo ""
echo "Starting SuperLink..."
echo "  Fleet API      -> 0.0.0.0:9092  (SuperNodes connect here)"
echo "  ServerApp API  -> 0.0.0.0:9093  (notebook / flwr run connects here)"
echo ""
echo "Press Ctrl+C to stop."

"$SUPERLINK_BIN" \
  --insecure \
  --database ":flwr-in-memory:"
