#!/usr/bin/env bash
# Run this script on each SuperNode machine.
#
# Usage:
#   bash start_supernode.sh <SUPERLINK_IP> [CLIENTAPPIO_PORT]
#
# Arguments:
#   SUPERLINK_IP      IP address of the machine running the SuperLink (required)
#   CLIENTAPPIO_PORT  Local port for ClientApp I/O (default: 9094)
#                     Use a different port if you run two nodes on the same machine.
#
# Examples:
#   bash start_supernode.sh 192.168.1.10          # node on a dedicated machine
#   bash start_supernode.sh 192.168.1.10 9094     # first node on shared machine
#   bash start_supernode.sh 192.168.1.10 9096     # second node on same machine

set -euo pipefail

SUPERLINK_IP="${1:?Error: SUPERLINK_IP is required. Usage: bash start_supernode.sh <SUPERLINK_IP> [PORT]}"
CLIENTAPPIO_PORT="${2:-9094}"

# Locate flower-supernode: first try PATH, then look next to the active Python.
if command -v flower-supernode &>/dev/null; then
    SUPERNODE_BIN="flower-supernode"
else
    _PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
    if [ -z "$_PY" ]; then
        echo "Error: no Python interpreter found in PATH." >&2
        exit 1
    fi
    _CANDIDATE="$(dirname "$_PY")/flower-supernode"
    if [ -x "$_CANDIDATE" ]; then
        SUPERNODE_BIN="$_CANDIDATE"
    else
        echo "Error: 'flower-supernode' not found." >&2
        echo "Activate the virtual environment where 'flwr' is installed, or run:" >&2
        echo "  pip install 'flwr[simulation]>=1.15,<2.0'" >&2
        exit 1
    fi
fi

echo "Using: $SUPERNODE_BIN"
echo ""
echo "Starting SuperNode..."
echo "  SuperLink Fleet API -> ${SUPERLINK_IP}:9092"
echo "  ClientApp I/O       -> 0.0.0.0:${CLIENTAPPIO_PORT}"
echo ""
echo "Press Ctrl+C to stop."

"$SUPERNODE_BIN" \
  --insecure \
  --superlink "${SUPERLINK_IP}:9092" \
  --clientappio-api-address "0.0.0.0:${CLIENTAPPIO_PORT}" \
  --max-retries 0
