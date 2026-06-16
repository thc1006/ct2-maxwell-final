# ct2-maxwell-final

A **final, frozen** build of [CTranslate2](https://github.com/OpenNMT/CTranslate2) that
re-introduces NVIDIA Compute Capability **5.0 (sm_50, Maxwell GM107/GM108)** support, so
that [faster-whisper](https://github.com/SYSTRAN/faster-whisper) runs on Maxwell GPUs
instead of failing to launch any CUDA kernel. Stock CTranslate2 wheels (and any CUDA 12
build of upstream) no longer emit sm_50 SASS, so on a Maxwell card faster-whisper aborts
the moment it tries to run on the GPU. This repo carries the upstream sm_50 patch, pins
the *last* toolchain that can still compile it, and packages a working build.

It is **frozen by design.** CUDA 13.0 removes sm_50 codegen from `nvcc`, and cuDNN 9.11
raises the minimum compute capability to 7.5 (Turing), dropping Maxwell entirely. This
project therefore targets the last Maxwell-supporting toolchain (CUDA 12.9 + cuDNN 9.10.x)
and **will not be maintained past it.** There is no upgrade path; that is the point.

## The error this fixes

On a Maxwell GPU, a stock faster-whisper / CTranslate2 install dies on the GPU path with:

```
cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device
```

This means the loaded `libctranslate2.so` contains no kernels compiled for your card's
architecture (sm_50). It is not a driver problem and not a faster-whisper bug — the SASS
for Maxwell simply was never emitted.

### Who is affected

Any GPU with compute capability **5.0 (Maxwell GM107/GM108)**, including:

- Quadro K2200 (4 GB)
- GeForce GTX 750 / 750 Ti
- GeForce GTX 9xx (e.g. GTX 950/960-class GM-parts at cc 5.0)
- GeForce 940MX (2 GB)
- GeForce 960M (laptop)

(Higher Maxwell parts at cc 5.2, e.g. GTX 970/980, hit the same upstream gap; this build
targets cc 5.0 specifically — see `CUDA_ARCH_LIST` below.)

## Frozen pins

These are the ceiling that still supports sm_50. The shell scripts, CI workflow, and this
README all agree with [`BUILD_SPEC.md`](BUILD_SPEC.md), which is the single source of truth.

| Component      | Pin                                      | Why this is the last one |
|----------------|------------------------------------------|--------------------------|
| CUDA Toolkit   | **12.9** (`cuda-toolkit-12-9`)           | last `nvcc` that emits sm_50 SASS; CUDA 13.0 removes it |
| cuDNN          | **9.10.x** (`<= 9.10`, never 9.11)       | 9.11 raises min compute capability to 7.5 (Turing), dropping Maxwell |
| CTranslate2    | **v4.8.0** + [`patches/1766-sm50.patch`](patches/1766-sm50.patch) | latest release; patch applies clean |
| CUDA arch      | `-DCUDA_ARCH_LIST="5.0"`                 | only kernel we need; keeps build small and fast |
| NVIDIA driver  | **DO NOT TOUCH**                         | install the toolkit only; never `cuda`, `cuda-12-9`, or `cuda-drivers*`, which can replace your working driver |

The build/validation host runs driver 580.159.03 (R580, the last Maxwell driver branch;
>= the 575.51.03 floor for CUDA 12.9). The install script holds existing NVIDIA driver
packages, captures the driver version before and after, and aborts if it changes.

## Quickstart (build from source)

This must run on an **actual sm_50 Linux host** — the build emits and validates Maxwell
SASS, so the GPU has to be present. Target platform is **Ubuntu 24.04 x86_64** with a
working NVIDIA driver already installed (you need `nvidia-smi` to report cc 5.0).

```bash
git clone https://github.com/thc1006/ct2-maxwell-final.git
cd ct2-maxwell-final

# 1. Install the frozen toolchain: CUDA Toolkit 12.9 (no driver) + cuDNN 9.10.x.
#    Holds your NVIDIA driver, pins cuDNN, guards disk, aborts if the driver moves.
bash scripts/01_install_toolchain.sh

# 2. Clone CTranslate2 v4.8.0, apply patches/1766-sm50.patch, build the C++ lib,
#    build the Python wheel into ./venv, then install faster-whisper and
#    force-reinstall our patched wheel last so the sm_50 build wins.
bash scripts/02_build_ct2.sh

# 3. Validate: prove the GPU path works and benchmark GPU vs CPU.
source venv/bin/activate
source cuda-env.sh
python scripts/03_validate.py
```

Notes:

- The scripts are idempotent and re-runnable, and use `set -euo pipefail`.
- Step 1 installs **only** `cuda-toolkit-12-9`. It never installs driver metapackages.
- If apt offers no cuDNN 9.10.x (only 9.11+), step 1 aborts and tells you to use the
  cuDNN 8 fallback below.

## Using it with faster-whisper

After step 2, the project venv has faster-whisper plus the patched sm_50 `ctranslate2`:

```python
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cuda", compute_type="int8")
segments, info = model.transcribe("audio.wav", beam_size=1)
for s in segments:
    print(s.text)
```

**VRAM caveat.** Pick the model to fit the card:

- **Quadro K2200 (4 GB):** tiny / base / small int8 are comfortable.
- **GeForce 940MX (2 GB):** stick to **tiny / base / small int8**. `large` will not fit
  in 2 GB and will OOM. Do not assume a model that runs on the K2200 also runs on a 2 GB
  card.

## Is the GPU even worth it on Maxwell?

Be honest with yourself before committing to the GPU path. sm_50 has **no native FP16** and
**no `dp4a` int8 acceleration** (both arrived in later architectures). On Maxwell, int8 and
fp16 fall back to slower paths, so for small Whisper models the GPU may be only marginally
faster than — or even slower than — a decent CPU running int8. The GPU is not automatically
the right choice here.

`scripts/03_validate.py` exists precisely to settle this: it transcribes the same clip
across `cuda float32`, `cuda int8_float32`, `cuda int8`, `cpu int8`, and `cpu float32`,
prints load time, transcribe time, and real-time factor (RTF), and writes
`validation_results.json`. **Run it on your own hardware and compare** before deciding the
GPU is worth it.

Benchmark (to be filled in after validation on the Quadro K2200):

| device | compute_type   | transcribe_s | RTF  |
|--------|----------------|--------------|------|
| cuda   | float32        | TBD          | TBD  |
| cuda   | int8_float32   | TBD          | TBD  |
| cuda   | int8           | TBD          | TBD  |
| cpu    | int8           | TBD          | TBD  |
| cpu    | float32        | TBD          | TBD  |

## Prebuilt wheels

Prebuilt `ctranslate2` wheels are attached to the project's **GitHub Releases**, produced by
the CI workflow. They:

- contain **only sm_50 SASS** (no other architectures), and
- target **CUDA 12.9 + cuDNN 9.10** on **Linux x86_64**.

They will not run anywhere else. On a non-Maxwell card, on a different CUDA/cuDNN, or on a
different platform, build from source instead. If you are on a Maxwell sm_50 host with the
frozen toolchain installed (step 1 above), you can `pip install` the release wheel directly
rather than running the full build.

## Credit

The actual fix is **OpenNMT/CTranslate2 PR #1766 by Giulio Paci ([@giuliopaci](https://github.com/giuliopaci)),
which is still open.** It re-adds sm_50 to the CUDA 12 arch list and guards the AWQ
dequantize kernel for pre-sm_53 devices. All credit for the working code goes to him.

- PR: https://github.com/OpenNMT/CTranslate2/pull/1766
- Issue (Maxwell / CUDA 12 codegen): https://github.com/OpenNMT/CTranslate2/issues/1765
- Original Quadro K2200 report: https://github.com/OpenNMT/CTranslate2/issues/1666

**This repository merely packages and freezes that patch** against a known-good toolchain.
It contributes no kernel code of its own; see [`patches/1766-sm50.patch`](patches/1766-sm50.patch).

## Conservative fallback

If the cuDNN 9.10 / Maxwell path proves fragile (e.g. apt no longer offers a 9.10.x build,
or you hit cuDNN runtime issues), fall back to **cuDNN 8 + `ctranslate2==4.4.0`**. cuDNN 8
fully supported sm_50, and CTranslate2 4.4.0 predates the upstream changes that dropped it.
This is older and slower-moving, but rock-solid on Maxwell.

## License

The scripts, patch packaging, and configuration in this repository are released under the
**MIT License**. CTranslate2 itself is also **MIT-licensed**; the sm_50 patch is carried
under the same terms as upstream.
