#!/usr/bin/env bash
set -euo pipefail

escape_json() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  printf '%s' "${value}"
}

l4t="$(head -n 1 /etc/nv_tegra_release 2>/dev/null || true)"
gpu_query="$(nvidia-smi --query-gpu=name,uuid,mig.mode.current,mig.mode.pending \
  --format=csv,noheader 2>&1 || true)"
mig_list="$(nvidia-smi -L 2>&1 || true)"

release_major=0
release_minor=0
if [[ "${l4t}" =~ R([0-9]+).*REVISION:[[:space:]]*([0-9]+)\.([0-9]+) ]]; then
  release_major="${BASH_REMATCH[1]}"
  release_minor="${BASH_REMATCH[2]}"
fi

r39_ready=false
if (( release_major > 39 || (release_major == 39 && release_minor >= 2) )); then
  r39_ready=true
fi

mig_enabled=false
if [[ "${gpu_query}" == *", Enabled,"* ]]; then
  mig_enabled=true
fi

mig_instances=0
if [[ "${mig_list}" == *"MIG "* ]]; then
  mig_instances="$(grep -c 'MIG ' <<<"${mig_list}")"
fi

cat <<EOF
{
  "l4t_release": "$(escape_json "${l4t}")",
  "r39_2_or_newer": ${r39_ready},
  "gpu_query": "$(escape_json "${gpu_query}")",
  "mig_enabled": ${mig_enabled},
  "mig_instances": ${mig_instances},
  "device_list": "$(escape_json "${mig_list}")"
}
EOF

if [[ "${r39_ready}" != true ]]; then
  echo "MIG experiments require JetPack 7.2 / L4T R39.2 or newer" >&2
  exit 3
fi
if [[ "${mig_enabled}" != true ]]; then
  echo "MIG is supported by this BSP but is not enabled" >&2
  exit 4
fi
if (( mig_instances < 2 )); then
  echo "two MIG instances have not been created" >&2
  exit 5
fi
