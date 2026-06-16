#!/usr/bin/env bash
# 02_build_ct2.sh
# Build CTranslate2 v4.8.0 + PR#1766 sm_50 patch, install the C++ lib, and
# build the Python wheel into a venv. See BUILD_SPEC.md for the frozen pins.
set -euo pipefail

log() { printf '\n\033[1;36m[build] %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[build][FATAL] %s\033[0m\n' "$*" >&2; exit 1; }

PROJ="${HOME}/projects/ct2-maxwell-final"
SRC="${PROJ}/CTranslate2"
PATCH="${PROJ}/patches/1766-sm50.patch"
VENV="${PROJ}/venv"
CT2_TAG="v4.8.0"
JOBS="$(nproc)"

# --- 0. Env -----------------------------------------------------------------
[[ -f "${PROJ}/cuda-env.sh" ]] || die "cuda-env.sh missing; run 01_install_toolchain.sh first"
# shellcheck disable=SC1090
source "${PROJ}/cuda-env.sh"
command -v nvcc >/dev/null || die "nvcc not on PATH after sourcing cuda-env.sh"
[[ -f "${PATCH}" ]] || die "patch not found: ${PATCH}"
log "nvcc $(nvcc --version | grep -oE 'release [0-9.]+'), jobs=${JOBS}"

# --- 1. Clone CTranslate2 (idempotent) --------------------------------------
if [[ ! -d "${SRC}/.git" ]]; then
  log "Cloning CTranslate2 ${CT2_TAG}"
  git clone --branch "${CT2_TAG}" --depth 1 --recursive \
    https://github.com/OpenNMT/CTranslate2.git "${SRC}"
else
  log "CTranslate2 already cloned; resetting to ${CT2_TAG}"
  git -C "${SRC}" fetch --depth 1 origin "refs/tags/${CT2_TAG}:refs/tags/${CT2_TAG}" || true
  git -C "${SRC}" checkout -f "${CT2_TAG}"
  git -C "${SRC}" submodule update --init --recursive
fi

# --- 2. Apply sm_50 patch (skip if already applied) -------------------------
cd "${SRC}"
# Deterministic on every run: restore the two patched files to the tag state,
# then apply fresh. Avoids the "mixed/partially applied tree" ambiguity.
git checkout -f -- CMakeLists.txt src/ops/awq/dequantize_gpu.cu
git apply --check "${PATCH}" || die "patch does not apply cleanly to ${CT2_TAG}"
git apply "${PATCH}"
log "Applied 1766-sm50.patch (from clean tree)"
# Confirm the patch markers are present
grep -q 'CUDA_VERSION_MAJOR EQUAL 12' CMakeLists.txt || die "CMake patch marker missing"
grep -q '__CUDA_ARCH__ < 530' src/ops/awq/dequantize_gpu.cu || die "kernel patch marker missing"

# --- 3. Configure + build the C++ library -----------------------------------
# Re-check disk right before the build tree (objects are 4-8 GB transiently).
BUILD_FREE_GB="$(df -BM --output=avail "${HOME}" 2>/dev/null | tail -1 | tr -dc '0-9' | awk '{printf "%d", $1/1024}')"
log "Free disk for build tree (\$HOME): ${BUILD_FREE_GB} GB"
[[ "${BUILD_FREE_GB}" -ge 12 ]] || die "need >= 12 GB free for the CUDA build, have ${BUILD_FREE_GB} GB"

log "Configuring (CUDA_ARCH_LIST=5.0, OpenBLAS, no MKL/DNNL)"
rm -rf "${SRC}/build"
cmake -S "${SRC}" -B "${SRC}/build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DCUDA_ARCH_LIST="5.0" \
  -DWITH_MKL=OFF \
  -DWITH_DNNL=OFF \
  -DWITH_OPENBLAS=ON \
  -DWITH_RUY=ON \
  -DOPENMP_RUNTIME=COMP \
  -DCMAKE_INSTALL_PREFIX=/usr/local

log "Compiling (this is the long part on ${JOBS} cores)"
cmake --build "${SRC}/build" -j "${JOBS}"

log "Installing libctranslate2 to /usr/local"
sudo cmake --install "${SRC}/build"
sudo ldconfig

# --- 4. Verify sm_50 SASS is actually in the .so ----------------------------
SO="$(find /usr/local/lib -name 'libctranslate2.so*' | head -1)"
[[ -n "${SO}" ]] || die "libctranslate2.so not found after install"
if command -v cuobjdump >/dev/null; then
  log "Checking embedded SASS arch in ${SO}"
  cuobjdump "${SO}" 2>/dev/null | grep -iE 'arch = sm_5|sm_50' | head -5 \
    || log "WARNING: could not confirm sm_50 SASS via cuobjdump (review manually)"
fi

# --- 5. Build the Python wheel into a venv ----------------------------------
log "Creating venv + building ctranslate2 wheel"
[[ -d "${VENV}" ]] || python3 -m venv "${VENV}"
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
pip install --upgrade pip wheel build
cd "${SRC}/python"
pip install -r install_requirements.txt
export CTRANSLATE2_ROOT=/usr/local
rm -rf "${SRC}/python/dist"   # ensure the only wheel present is the fresh one
python -m build --wheel --no-isolation
shopt -s nullglob
wheels=("${SRC}/python/dist/"*.whl)
[[ ${#wheels[@]} -eq 1 ]] || die "expected exactly 1 fresh wheel, got ${#wheels[@]}"
WHEEL="${wheels[0]}"
mkdir -p "${PROJ}/dist"
cp "${WHEEL}" "${PROJ}/dist/"
pip install --force-reinstall "${WHEEL}"

# --- 6. Install faster-whisper, then RE-ASSERT our wheel ---------------------
# faster-whisper depends on ctranslate2 and will pull the sm_50-less PyPI build;
# force-reinstall ours last so the patched library wins.
log "Installing faster-whisper (then restoring our sm_50 wheel)"
pip install faster-whisper
pip install --force-reinstall --no-deps "${WHEEL}"
python -c "import ctranslate2 as c; assert c.get_cuda_device_count() >= 1; print('ct2 from', c.__file__)"

log "=== BUILD DONE ==="
echo "  wheel: ${PROJ}/dist/$(basename "${WHEEL}")"
python -c "import ctranslate2 as c; print('ctranslate2', c.__version__, '| cuda devices:', c.get_cuda_device_count())"
log "Next: scripts/03_validate.py (inside ${VENV})"
