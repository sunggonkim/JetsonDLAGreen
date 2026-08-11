#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 BACKUP_DIRECTORY" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_ROOT="$(realpath -m "$1")"
RUN_DIR="${BACKUP_ROOT}/thor-r38-backup-$(date -u +%Y%m%dT%H%M%SZ)"
SOURCE_DEVICE="$(findmnt -n -o SOURCE -T "${ROOT_DIR}")"
readonly ROOT_DIR BACKUP_ROOT RUN_DIR SOURCE_DEVICE

mkdir -p "${BACKUP_ROOT}"
BACKUP_DEVICE="$(findmnt -n -o SOURCE -T "${BACKUP_ROOT}")"
readonly BACKUP_DEVICE

if [[ "${SOURCE_DEVICE}" == "${BACKUP_DEVICE}" && "${ALLOW_SAME_DEVICE:-0}" != "1" ]]; then
  echo "refusing backup to ${BACKUP_ROOT}: source and destination are both ${SOURCE_DEVICE}" >&2
  echo "mount an external disk, or set ALLOW_SAME_DEVICE=1 for a non-flash-safe snapshot" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/system"

cat /etc/nv_tegra_release >"${RUN_DIR}/system/nv_tegra_release.txt"
cp /etc/os-release "${RUN_DIR}/system/os-release"
uname -a >"${RUN_DIR}/system/uname.txt"
lsblk -o NAME,SIZE,FSTYPE,UUID,PARTUUID,MOUNTPOINTS,MODEL \
  >"${RUN_DIR}/system/lsblk.txt"
findmnt --real >"${RUN_DIR}/system/findmnt.txt"
df -h >"${RUN_DIR}/system/df.txt"
dpkg-query -W -f='${binary:Package}\t${Version}\n' \
  >"${RUN_DIR}/system/dpkg-packages.tsv"
apt-mark showmanual >"${RUN_DIR}/system/apt-manual.txt"
cp -a /etc/apt/sources.list /etc/apt/sources.list.d "${RUN_DIR}/system/" 2>/dev/null || true
nvidia-smi -q >"${RUN_DIR}/system/nvidia-smi.txt" 2>&1 || true
nvpmodel -q >"${RUN_DIR}/system/nvpmodel.txt" 2>&1 || true

git -C "${ROOT_DIR}" status --short >"${RUN_DIR}/git-status.txt"
git -C "${ROOT_DIR}" diff --binary >"${RUN_DIR}/working-tree.patch"
git -C "${ROOT_DIR}" ls-files --others --exclude-standard -z \
  | tar --null -T - -C "${ROOT_DIR}" -czf "${RUN_DIR}/untracked-files.tar.gz"
git -C "${ROOT_DIR}" bundle create "${RUN_DIR}/repository.bundle" --all

if [[ -d "${ROOT_DIR}/results" ]]; then
  tar -C "${ROOT_DIR}" -czf "${RUN_DIR}/results.tar.gz" results
fi

(cd "${RUN_DIR}" && sha256sum ./* system/* >SHA256SUMS)
echo "upgrade backup: ${RUN_DIR}"
echo "verify with: (cd '${RUN_DIR}' && sha256sum -c SHA256SUMS)"
