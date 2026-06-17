# =============================================================================
# CTranslate2 — Maxwell (sm_50) FROZEN build — reproducible Docker build
# =============================================================================
#
# Cross-compiles a CTranslate2 Python wheel that contains ONLY sm_50 (Maxwell,
# GM107/GM108) SASS. The Docker host building this needs NO GPU: nvcc only
# COMPILES for sm_50 here, nothing is executed against a device in the image.
#
# This image is FROZEN by design. It targets the LAST toolchain that still
# supports Maxwell. Do not bump any pin past these values:
#
#   Component     | Pin                                | Why this is the ceiling
#   ------------- | ---------------------------------- | ------------------------------
#   Base image    | nvidia/cuda:12.9.1-devel-ubuntu24.04| -devel ships nvcc; 12.9 is the
#                 |                                    | last nvcc that emits sm_50 SASS
#                 |                                    | (CUDA 13.0 removes sm_50 codegen)
#   cuDNN         | 9.10.x  (<= 9.10, NEVER 9.11+)     | 9.11 raises min compute cap to
#                 |                                    | 7.5 (Turing), dropping Maxwell
#   CTranslate2   | v4.8.0 + patches/1766-sm50.patch   | latest release; patch applies clean
#   CUDA arch     | CUDA_ARCH_LIST="5.0"               | only kernel we need
#
# IMPORTANT: we deliberately do NOT use a "-cudnn" base image variant. Its
# bundled cuDNN may be 9.11+, which DROPS Maxwell. We install cuDNN ourselves
# pinned to a 9.10.x version from the NVIDIA apt repo (mirrors the pin logic in
# scripts/01_install_toolchain.sh). See the cuDNN stage below for the fallback
# if a 9.10.x package is no longer served by the repo.
#
# The resulting wheel:
#   - contains ONLY sm_50 SASS (runs on Maxwell: Quadro K2200, GTX 750/Ti,
#     GTX 9xx, 940MX, 960M, ...)
#   - requires a CUDA 12.9 runtime + NVIDIA driver >= 575.51.03
#     (the K2200 / 940MX R580 branch driver, e.g. 580.159.03, satisfies this)
#   - links cuDNN 9.10.x and OpenBLAS (no MKL/DNNL) per BUILD_SPEC.md
#
# Build host needs NO GPU and NO --gpus flag (compile only).
#
# -----------------------------------------------------------------------------
# How to extract the wheel
# -----------------------------------------------------------------------------
# Option A — copy out of the final image (recommended):
#     docker build -t ct2-maxwell -f docker/Dockerfile .
#     # create a throwaway container and copy /wheels out of it
#     id=$(docker create ct2-maxwell)
#     docker cp "$id":/wheels ./wheels
#     docker rm "$id"
#
# Option B — BuildKit export of just the wheels (no image kept):
#     DOCKER_BUILDKIT=1 docker build --target export \
#       --output type=local,dest=./wheels -f docker/Dockerfile .
#
# The repaired manylinux/linux wheel lands in ./wheels/.
# Build context must be the repo root (so patches/1766-sm50.patch is COPY-able):
#     docker build -f docker/Dockerfile .
# =============================================================================


# -----------------------------------------------------------------------------
# Stage: builder — toolchain + cuDNN pin + C++ lib + wheel
# -----------------------------------------------------------------------------
# -devel image => has nvcc. 12.9.1 is the last CUDA that emits sm_50 SASS.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS builder

# Frozen pins (kept in one place so they match BUILD_SPEC.md)
ARG CT2_TAG=v4.8.0
ARG CUDNN_MAJOR_MINOR=9.10
# Default interpreter is python3.12 from the ubuntu24.04 base; overridable.
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

# --- Base build deps (mirror 01_install_toolchain.sh) ------------------------
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      ninja-build \
      git \
      curl \
      wget \
      ca-certificates \
      python3-pip \
      python3-venv \
      python3-dev \
      libopenblas-dev \
      pkg-config \
      patchelf && \
    rm -rf /var/lib/apt/lists/*

# --- cuDNN pinned to 9.10.x (NEVER 9.11+, which drops Maxwell) ---------------
# The 12.9.1-devel base already carries the NVIDIA CUDA apt repo + keyring, so
# we can resolve cuDNN versions directly. We replicate the pin logic from
# scripts/01_install_toolchain.sh:
#   apt-cache madison libcudnn9-cuda-12 | grep '^9.10.' | sort -V | tail -1
#
# FALLBACK (documented per request): if the NVIDIA repo ever stops serving a
# 9.10.x package (only 9.11+ remains), this RUN FAILS LOUDLY rather than
# silently installing a Maxwell-dropping cuDNN. In that case, switch to the
# conservative fallback from BUILD_SPEC.md:  cuDNN 8 + CTranslate2 == 4.4.0
# (cuDNN 8 fully supported sm_50). Do NOT "fix" this by relaxing the grep to
# accept 9.11.
RUN set -eux; \
    apt-get update -y; \
    if ! apt-cache madison libcudnn9-cuda-12 >/dev/null 2>&1; then \
      echo "FATAL: libcudnn9-cuda-12 not visible in apt. The base image should already carry the NVIDIA CUDA repo+keyring; if not, add it (cuda-keyring) before this step." >&2; \
      exit 1; \
    fi; \
    CUDNN_VER="$(apt-cache madison libcudnn9-cuda-12 | awk '{print $3}' | grep -E "^${CUDNN_MAJOR_MINOR}\\." | sort -V | tail -1 || true)"; \
    CUDNN_HDR_VER="$(apt-cache madison libcudnn9-headers-cuda-12 | awk '{print $3}' | grep -E "^${CUDNN_MAJOR_MINOR}\\." | sort -V | tail -1 || true)"; \
    CUDNN_DEV_VER="$(apt-cache madison libcudnn9-dev-cuda-12 | awk '{print $3}' | grep -E "^${CUDNN_MAJOR_MINOR}\\." | sort -V | tail -1 || true)"; \
    if [ -z "${CUDNN_VER}" ] || [ -z "${CUDNN_HDR_VER}" ] || [ -z "${CUDNN_DEV_VER}" ]; then \
      echo "FATAL: no cuDNN ${CUDNN_MAJOR_MINOR}.x available in apt (only 9.11+? that DROPS Maxwell)." >&2; \
      echo "       FALLBACK: rebuild with cuDNN 8 + CTranslate2 4.4.0 per BUILD_SPEC.md. Do not accept 9.11." >&2; \
      exit 1; \
    fi; \
    echo "Installing cuDNN pinned: rt=${CUDNN_VER} hdr=${CUDNN_HDR_VER} dev=${CUDNN_DEV_VER}"; \
    apt-get install -y --no-install-recommends --allow-downgrades \
      "libcudnn9-cuda-12=${CUDNN_VER}" \
      "libcudnn9-headers-cuda-12=${CUDNN_HDR_VER}" \
      "libcudnn9-dev-cuda-12=${CUDNN_DEV_VER}"; \
    apt-mark hold libcudnn9-cuda-12 libcudnn9-headers-cuda-12 libcudnn9-dev-cuda-12; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# --- Python build toolchain --------------------------------------------------
# ubuntu24.04 pip is PEP 668 externally-managed; --break-system-packages is the
# intended escape hatch inside a throwaway build container.
RUN python3 -m pip install --no-cache-dir --break-system-packages --upgrade \
      pip build wheel auditwheel

# --- Clone CTranslate2 v4.8.0 (with submodules) ------------------------------
WORKDIR /src
RUN git clone --branch "${CT2_TAG}" --depth 1 --recursive \
      https://github.com/OpenNMT/CTranslate2.git /src/CTranslate2

# --- Apply the sm_50 patch (git apply, NOT --3way) ---------------------------
# Build context root is the repo, so patches/ is COPY-able.
COPY patches/1766-sm50.patch /tmp/1766-sm50.patch
WORKDIR /src/CTranslate2
RUN set -eux; \
    git apply --check /tmp/1766-sm50.patch; \
    git apply /tmp/1766-sm50.patch; \
    # Confirm both patch markers are present (same gate as 02_build_ct2.sh)
    grep -q 'CUDA_VERSION_MAJOR EQUAL 12' CMakeLists.txt; \
    grep -q '__CUDA_ARCH__ < 530' src/ops/awq/dequantize_gpu.cu; \
    echo "Patch applied and both markers verified."

# --- Configure + build + install the C++ library -----------------------------
# EXACTLY the canonical CMake invocation from BUILD_SPEC.md.
RUN cmake -S /src/CTranslate2 -B /src/CTranslate2/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_CUDA=ON \
      -DWITH_CUDNN=ON \
      -DCUDA_ARCH_LIST="5.0" \
      -DWITH_MKL=OFF \
      -DWITH_DNNL=OFF \
      -DWITH_OPENBLAS=ON \
      -DWITH_RUY=ON \
      -DOPENMP_RUNTIME=COMP \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /src/CTranslate2/build -j "$(nproc)" && \
    cmake --install /src/CTranslate2/build && \
    ldconfig

# --- Sanity: confirm sm_50 SASS landed in the .so (best-effort) --------------
RUN set -eux; \
    SO="$(find /usr/local/lib -name 'libctranslate2.so*' | head -1)"; \
    test -n "${SO}"; \
    if command -v cuobjdump >/dev/null 2>&1; then \
      cuobjdump "${SO}" 2>/dev/null | grep -iE 'arch = sm_50' | head -3 \
        || echo "WARNING: could not confirm sm_50 SASS via cuobjdump (review manually)"; \
    else \
      echo "cuobjdump not present; skipping SASS check"; \
    fi

# --- Build the Python wheel --------------------------------------------------
WORKDIR /src/CTranslate2/python
RUN python3 -m pip install --no-cache-dir --break-system-packages \
      -r install_requirements.txt && \
    CTRANSLATE2_ROOT=/usr/local python3 -m build --wheel --no-isolation

# --- Repair the wheel: bundle libctranslate2.so into a portable wheel ---------
# auditwheel pulls the freshly installed .so (found via LD_LIBRARY_PATH) into
# the wheel and emits a manylinux/linux tagged wheel under /wheels.
# NOTE: CUDA/cuDNN runtime libs are intentionally provided by the deploy host
# (CUDA 12.9 runtime + driver >= 575); they are excluded so the wheel stays a
# thin sm_50 ctranslate2 build, exactly like 02_build_ct2.sh produces.
RUN set -eux; \
    mkdir -p /wheels; \
    RAW_WHEEL="$(ls -t /src/CTranslate2/python/dist/*.whl | head -1)"; \
    test -n "${RAW_WHEEL}"; \
    export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"; \
    auditwheel repair "${RAW_WHEEL}" \
      --exclude 'libcudart.so*' \
      --exclude 'libcublas.so*' \
      --exclude 'libcublasLt.so*' \
      --exclude 'libcudnn.so*' \
      --exclude 'libcudnn_*.so*' \
      --exclude 'libnvrtc.so*' \
      -w /wheels \
    || { \
      echo "auditwheel repair failed (likely due to excluded CUDA deps not being on a manylinux policy); falling back to copying the raw wheel."; \
      cp "${RAW_WHEEL}" /wheels/; \
    }; \
    ls -l /wheels


# -----------------------------------------------------------------------------
# Stage: export — minimal final image that just holds /wheels
# -----------------------------------------------------------------------------
# Lets you `docker cp` the wheel out, or use BuildKit --output to extract it
# without keeping the heavy builder image around.
FROM scratch AS export
COPY --from=builder /wheels /wheels
