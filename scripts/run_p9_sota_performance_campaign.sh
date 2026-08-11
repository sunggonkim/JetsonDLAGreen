#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. The old implementation mislabeled a local
# policy as BOER and could not produce a defensible comparison. Keep one
# executable path for the active matrix instead of retaining that campaign.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT_DIR}/scripts/run_p9_active_frontier_campaign.sh" "$@"
