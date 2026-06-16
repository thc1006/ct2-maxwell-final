# BUILD_SPEC — single source of truth

This file is the authoritative spec. The shell scripts, Dockerfile, CI workflow,
and README must all agree with the pins and flags below. Do not diverge.

## What this project is

A **final, frozen** build of CTranslate2 that re-introduces NVIDIA Compute
Capability **5.0 (sm_50, Maxwell GM107/GM108)** support, so that
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) runs on Maxwell GPUs
(Quadro K2200, GeForce GTX 750/Ti, GTX 9xx, 940MX, 960M, ...) without:

    cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device

It is **frozen by design**: CUDA 13 removes sm_50 codegen and cuDNN 9.11 drops
Maxwell, so this build targets the *last* toolchain that still supports Maxwell
and will not be updated past it.

Upstream fix: OpenNMT/CTranslate2 PR #1766 by @giuliopaci (still open). We carry
that patch in `patches/1766-sm50.patch` and credit it.

## Frozen pins (the ceiling that still supports sm_50)

| Component      | Pin                              | Why this is the last one |
|----------------|----------------------------------|--------------------------|
| CUDA Toolkit   | **12.9** (`cuda-toolkit-12-9`)   | last nvcc that emits sm_50 SASS; CUDA 13.0 removes it |
| cuDNN          | **9.10.x** (`<= 9.10`, never 9.11) | 9.11 raises min compute capability to 7.5 (Turing), dropping Maxwell |
| CTranslate2    | **v4.8.0** + `patches/1766-sm50.patch` | latest release; patch applies clean |
| CUDA arch      | `-DCUDA_ARCH_LIST="5.0"`         | only kernel we need; keeps build small/fast |
| NVIDIA driver  | **DO NOT TOUCH** (build host has 580.159.03) | R580 is the last Maxwell branch; 580 >= 575.51.03 floor for CUDA 12.9 |

Conservative fallback if the cuDNN 9.10 / Maxwell path proves fragile:
cuDNN 8 + CTranslate2 == 4.4.0 (cuDNN 8 fully supported sm_50).

## Build host (validation target)

- Host alias: `1030-dev` (hostname `thc1006-D630MT`), Ubuntu 24.04.4 LTS, x86_64
- GPU: Quadro K2200, compute cap 5.0, 4 GB, driver 580.159.03
- 4 cores, 23 GB RAM, ~31 GB free disk (TIGHT — keep the build lean)
- Deploy target downstream: GeForce 940MX (same sm_50, only 2 GB VRAM)

## Canonical CMake invocation (must match everywhere)

```
cmake .. -GNinja \
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
```

Rationale: OpenBLAS (not Intel MKL) for the CPU GEMM backend keeps the install
small on a disk-constrained host; RUY gives int8 CPU GEMM for the CPU benchmark
baseline; single arch 5.0 keeps nvcc time and disk down.

## Hard safety rules (enforced by adversarial-review before any run)

1. Install ONLY `cuda-toolkit-12-9`. NEVER `cuda`, `cuda-12-9`, or `cuda-drivers*`
   (those pull a driver and can break the working 580 driver).
2. Pin cuDNN to a `9.10.*` version explicitly and `apt-mark hold` it.
3. Capture `nvidia-smi` driver version before/after install; abort/alert on change.
4. Guard free disk (>= 12 GB) before installing; `apt-get clean` after.
5. Every script: `set -euo pipefail`, idempotent, re-runnable.
6. No script runs on the box until `code_review.md` is marked PASS.
