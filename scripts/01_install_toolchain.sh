#!/usr/bin/env bash
# 01_install_toolchain.sh
# Install the FROZEN Maxwell-sm_50 toolchain on Ubuntu 24.04 (x86_64):
#   CUDA Toolkit 12.9 (TOOLKIT ONLY, no driver) + cuDNN 9.10.x + build tools.
# Safety: never touches the NVIDIA driver; pins cuDNN <= 9.10; guards disk.
# Idempotent and re-runnable. See BUILD_SPEC.md.
set -euo pipefail

log() { printf '\n\033[1;36m[install] %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[install][FATAL] %s\033[0m\n' "$*" >&2; exit 1; }

# Resolve the project root from THIS script's location, so the repo works no
# matter where it was cloned (do not hardcode ~/projects/...).
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 0. Preconditions -------------------------------------------------------
[[ "$(uname -m)" == "x86_64" ]] || die "expected x86_64"
. /etc/os-release
[[ "${VERSION_ID:-}" == "24.04" ]] || die "expected Ubuntu 24.04, got ${VERSION_ID:-unknown}"
command -v nvidia-smi >/dev/null || die "nvidia-smi missing; need a working NVIDIA driver first"

DRV_BEFORE="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
log "Driver before: ${DRV_BEFORE} | GPU compute capability: ${CC}"
[[ "${CC}" == "5.0" ]] || log "WARNING: GPU compute cap is ${CC}, not 5.0 (continuing anyway)"

# Measure the filesystems that actually back the install (/usr/local) and the
# build tree ($HOME), at MB precision (df -BG floors to whole GiB).
avail_gb() { df -BM --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9' | awk '{printf "%d", $1/1024}'; }
ROOT_GB="$(avail_gb /usr/local)"; HOME_GB="$(avail_gb "${HOME}")"
log "Free disk: /usr/local=${ROOT_GB} GB | \$HOME=${HOME_GB} GB"
# Fresh install needs toolkit(~7GB)+cuDNN(~3GB)+build(4-8GB) of headroom.
# On a re-run the toolkit is already present, so only cuDNN+build remain.
NEED_GB=25; [[ -d /usr/local/cuda-12.9/bin ]] && NEED_GB=10
[[ "${ROOT_GB}" -ge "${NEED_GB}" ]] || die "need >= ${NEED_GB} GB free where /usr/local lives, have ${ROOT_GB} GB"

# --- 1. Protect the working driver -----------------------------------------
# Hold any installed NVIDIA driver packages so apt cannot replace/remove them.
HOLD_PKGS="$(dpkg -l 2>/dev/null | awk '/^ii/ && $2 ~ /^(nvidia-(driver|dkms|kernel|compute|utils|firmware|fabricmanager)|libnvidia|xserver-xorg-video-nvidia)/ {print $2}')"
if [[ -n "${HOLD_PKGS}" ]]; then
  log "Holding driver packages:"; echo "${HOLD_PKGS}" | sed 's/^/    /'
  # shellcheck disable=SC2086
  sudo apt-mark hold ${HOLD_PKGS} || die "failed to hold driver packages; refusing to continue"
fi

# --- 2. Base build tools ----------------------------------------------------
log "Installing base build tools"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git curl wget ca-certificates \
  python3-pip python3-venv python3-dev libopenblas-dev pkg-config

# --- 3. Add NVIDIA CUDA apt repo (ubuntu2404) -------------------------------
if [[ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]] && \
   ! ls /etc/apt/sources.list.d/ 2>/dev/null | grep -qi cuda; then
  log "Adding NVIDIA CUDA repo keyring"
  TMP_DEB="$(mktemp --suffix=.deb)"
  curl -fsSL -o "${TMP_DEB}" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i "${TMP_DEB}"
  rm -f "${TMP_DEB}"
else
  log "CUDA apt repo already present"
fi
sudo apt-get update -y

# --- 4. CUDA Toolkit 12.9 (TOOLKIT ONLY — never the driver metapackages) ----
log "Installing cuda-toolkit-12-9 (no driver)"
sudo apt-get install -y --no-install-recommends cuda-toolkit-12-9
[[ -d /usr/local/cuda-12.9 ]] || die "/usr/local/cuda-12.9 not found after install"
# Free the .deb cache NOW (before cuDNN + the CUDA build tree) so the peak disk
# usage never stacks toolkit debs + cuDNN debs + build objects simultaneously.
# (To save a further ~3-4 GB you may replace cuda-toolkit-12-9 above with the
#  compile-only subset: cuda-nvcc-12-9 cuda-cudart-dev-12-9 cuda-libraries-dev-12-9
#  cuda-nvrtc-dev-12-9 cuda-cccl-12-9 cuda-nvtx-12-9 cuda-profiler-api-12-9 —
#  drops Nsight. Kept full toolkit here for zero missing-component risk.)
sudo apt-get clean

# --- 5. cuDNN pinned to 9.10.x (NEVER 9.11+, which drops Maxwell) -----------
madison_910() { apt-cache madison "$1" 2>/dev/null | awk '{print $3}' | grep -E '^9\.10\.' | sort -V | tail -1; }
CUDNN_VER="$(madison_910 libcudnn9-cuda-12 || true)"
CUDNN_DEV_VER="$(madison_910 libcudnn9-dev-cuda-12 || true)"
CUDNN_HDR_VER="$(madison_910 libcudnn9-headers-cuda-12 || true)"
[[ -n "${CUDNN_VER}" && -n "${CUDNN_DEV_VER}" && -n "${CUDNN_HDR_VER}" ]] || die "no cuDNN 9.10.x in apt (only 9.11+? that drops Maxwell). Use the cuDNN 8 + ctranslate2==4.4.0 fallback."
log "Installing cuDNN pinned: rt=${CUDNN_VER} hdr=${CUDNN_HDR_VER} dev=${CUDNN_DEV_VER}"
# libcudnn9-dev hard-depends on the exact-version headers pkg; apt will NOT
# auto-add it under exact-version pinning, so list all three explicitly.
sudo apt-get install -y --no-install-recommends --allow-downgrades \
  "libcudnn9-cuda-12=${CUDNN_VER}" \
  "libcudnn9-headers-cuda-12=${CUDNN_HDR_VER}" \
  "libcudnn9-dev-cuda-12=${CUDNN_DEV_VER}"
sudo apt-mark hold libcudnn9-cuda-12 libcudnn9-headers-cuda-12 libcudnn9-dev-cuda-12 || die "failed to hold cuDNN packages"

# --- 6. Persist CUDA env (does not affect anything until sourced) -----------
ENVFILE="${PROJ}/cuda-env.sh"
mkdir -p "$(dirname "${ENVFILE}")"
cat > "${ENVFILE}" <<'ENV'
# source this before building / running
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
ENV
log "Wrote CUDA env to ${ENVFILE}"

# --- 7. Cleanup + verify ----------------------------------------------------
sudo apt-get clean
DRV_AFTER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
# shellcheck disable=SC1090
source "${ENVFILE}"
NVCC_VER="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"

log "=== RESULT ==="
echo "  driver before/after : ${DRV_BEFORE} -> ${DRV_AFTER}"
echo "  nvcc release        : ${NVCC_VER}"
echo "  cuDNN               : ${CUDNN_VER}"
echo "  free disk now       : /usr/local=$(avail_gb /usr/local) GB | \$HOME=$(avail_gb "${HOME}") GB"
[[ "${DRV_AFTER}" == "${DRV_BEFORE}" ]] || die "DRIVER CHANGED (${DRV_BEFORE} -> ${DRV_AFTER}) — investigate before proceeding"
[[ "${NVCC_VER}" == "12.9" ]] || die "nvcc is ${NVCC_VER}, expected 12.9"
log "Toolchain ready. Driver untouched. Next: scripts/02_build_ct2.sh"
