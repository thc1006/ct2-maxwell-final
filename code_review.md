# code_review.md — adversarial review of ct2-maxwell-final

REVIEW STATUS: PASS

Reviewer stance: adversarial. This scaffold runs with `sudo` on a real, in-use
Ubuntu 24.04 box with a working NVIDIA 580.159.03 driver that must not be
disturbed, and ~31 GB free disk. One HIGH-severity issue exists (disk
exhaustion mid-build on a sudo `cmake --install` path), so per the rubric the
overall status is FAIL. The driver-safety design and the core patch are
otherwise sound and upstream-faithful.

Counts: HIGH = 1, MEDIUM = 5, LOW = 6.

---

## What I verified upstream (so the review is not vibes)

- Cloned `OpenNMT/CTranslate2` tag `v4.8.0` and inspected the real files.
- **CMake option names all exist with the stated semantics** in
  `CMakeLists.txt` (lines 10-16, 56):
  `WITH_MKL` (default ON), `WITH_DNNL` (OFF), `WITH_OPENBLAS` (OFF),
  `WITH_RUY` (OFF), `WITH_CUDA` (OFF), `WITH_CUDNN` (OFF), and
  `OPENMP_RUNTIME` is a CACHE STRING accepting `INTEL|COMP|NONE`
  (FATAL_ERROR on anything else). `CUDA_ARCH_LIST` is consumed at line 530-543
  and `"5.0"` is a valid value (passed straight to
  `cuda_select_nvcc_arch_flags`, the legacy FindCUDA helper, which emits
  `-gencode arch=compute_50,code=sm_50`). **No wrong/typo'd flag names.**
- **`CTRANSLATE2_ROOT` is the correct env var.** `python/setup.py` line 32-42:
  `_maybe_add_library_root("CTRANSLATE2")` reads `os.environ["CTRANSLATE2_ROOT"]`
  and tries `$ROOT/lib` then `$ROOT/lib64`. With `=/usr/local`, `/usr/local/lib`
  exists, and the linux rpath is `-Wl,-rpath,/usr/local/lib64:/usr/local/lib`.
- **`python -m build --wheel --no-isolation` is viable.** `python/pyproject.toml`
  declares `requires=["setuptools","wheel","pybind11==2.11.1"]`, and
  `install_requirements.txt` installs exactly those before the no-isolation build.
- **The carried patch is byte-for-byte identical to upstream PR #1766**
  (`gh pr diff 1766`: state OPEN, touches only `CMakeLists.txt` +2 and
  `src/ops/awq/dequantize_gpu.cu` +4). The `.cu` path is correct and exists.
  `git apply --check` is CLEAN against v4.8.0; reverse-check correctly fails
  before apply and passes after (idempotency works). Markers
  `CUDA_VERSION_MAJOR EQUAL 12` and `__CUDA_ARCH__ < 530` are present post-apply.
- **`cuda-toolkit-12-9` is toolkit-only and does NOT pull `cuda-drivers`.**
  The NVIDIA ubuntu2404 repo serves `cuda-toolkit-12-9_{12.9.0,12.9.1,12.9.2}`;
  only the umbrella `cuda` / `cuda-12-9` metapackages depend on `cuda-drivers`.
  `--no-install-recommends` additionally blocks recommended driver pulls.
- **`libcudnn9-cuda-12` / `libcudnn9-dev-cuda-12` are the correct cuDNN-9
  package names** (the `cudnn9-cuda-12` meta depends on them). They do not pull
  a driver.
- **cuDNN 9.11.0 really does drop Maxwell/Pascal/Volta** (min compute cap 7.5),
  confirmed in NVIDIA 9.11.0 release notes and the current support matrix
  (9.23.2 lists 7.5 as the floor for both CUDA 12.x and 13.x). The "`9.10.x` or
  die" pin and fallback note are justified and the empty-`madison` → `die` path
  is the correct loud failure.
- **faster-whisper requires `ctranslate2>=4.0,<5`**; our 4.8.0 satisfies it, so
  `pip install faster-whisper` will NOT replace our wheel, and the trailing
  `--force-reinstall --no-deps OURWHEEL` re-asserts it regardless. The clobber
  sequence is correct — no dependency-resolution hole.
- **faster-whisper API in 03 is correct**: `WhisperModel(size, device=,
  compute_type=)`; `transcribe()` returns `(Iterable[Segment],
  TranscriptionInfo)`; `TranscriptionInfo.duration` exists. Compute-type strings
  `int8`, `int8_float32`, `float32` are all valid (`src/types.cc` lines 47-49,
  `python/cpp/*.cc` docstrings list them).

Sources:
- https://raw.githubusercontent.com/OpenNMT/CTranslate2/v4.8.0/CMakeLists.txt
- https://github.com/OpenNMT/CTranslate2/pull/1766 (PR diff via `gh pr diff 1766`)
- https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/requirements.txt (`ctranslate2>=4.0,<5`)
- https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.0/release-notes.html (Maxwell dropped in 9.11.0)
- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html (min cc 7.5)
- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ (cuda-toolkit-12-9 vs cuda-drivers separation)

---

## Findings

### [HIGH] Disk can run out mid-build; `sudo cmake --install` failure leaves a half-installed lib

**Files:** `scripts/01_install_toolchain.sh:25`, `scripts/02_build_ct2.sh:52-69`;
spec `BUILD_SPEC.md:39,68`.

**Problem.** The only disk guard is a one-time `>= 12 GB` check in 01 *before*
the toolkit install. On a box with ~31 GB free the realistic peak is much
tighter than 12 GB of headroom:

- `cuda-toolkit-12-9` installed footprint is ~6.5-7.5 GB (it pulls
  `cuda-libraries-dev-12-9`, `cuda-nsight-systems-12-9`,
  `cuda-nsight-compute-12-9`, docs, etc. — Nsight alone is multiple GB).
- cuDNN 9.10 runtime + dev: ~3 GB.
- The CTranslate2 **CUDA** build tree (`-DWITH_CUDA=ON`, Release) produces very
  large `.o`/`.cu.o` objects; `rm -rf build` then a fresh full build is easily
  4-8 GB transiently.
- During apt, `.deb`s are cached under `/var/cache/apt/archives` *before*
  `apt-get clean` runs (01 only cleans at the very end, line 82), adding several
  GB to the simultaneous peak.

Summed peak can reach ~26-30 GB on a 31 GB-free disk. If `cmake --build` or, worse,
`sudo cmake --install` (02:69) fails on `ENOSPC`, you get a **partially
installed `libctranslate2.so` in `/usr/local`** plus a broken `ldconfig` state on
the production box — exactly the "messy, not graceful" failure the spec's
tightness warning is trying to avoid. 02 does no disk check at all, and the
`apt-get clean` that would free the `.deb` cache lives in 01, not between steps.

**Fix.** (a) Raise/parametrize the guard and re-check before the build; (b) trim
the toolkit to the compile-only subset; (c) clean the apt cache before the build,
not only at end of 01. Concretely:

In `01` step 4, prefer the lean metapackage set over the full toolkit:
```bash
# Nsight + docs are not needed to COMPILE for sm_50; this saves ~3-4 GB.
sudo apt-get install -y --no-install-recommends \
  cuda-nvcc-12-9 cuda-cudart-dev-12-9 cuda-libraries-dev-12-9 \
  cuda-nvtx-12-9 cuda-profiler-api-12-9 cuda-cccl-12-9
# (keep cuda-toolkit-12-9 only if you actually need nsight/profilers)
sudo apt-get clean          # free the .deb cache NOW, before cuDNN + build
```
And raise the guard to match reality and re-assert in `02`:
```bash
# 01:25  — 12 GB is too low for toolkit+cuDNN+CUDA build tree
[[ "${FREE_GB}" -ge 25 ]] || die "need >= 25 GB free on /, have ${FREE_GB} GB"
```
```bash
# 02, before 'rm -rf build' (line 52):
FREE_GB="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
[[ "${FREE_GB}" -ge 12 ]] || die "need >= 12 GB free for the CUDA build, have ${FREE_GB} GB"
```
If trimming the toolkit is undesirable, at minimum add the `02` pre-build disk
guard and move an `apt-get clean` to the end of `01` step 4 (it is already at
line 82 but *after* cuDNN; move a clean to right after the toolkit install too).

---

### [MEDIUM] 01:23 / 01:92 — `df -BG ... / ` rounds DOWN; a true 11.6 GB reads as "11" and a true 12.4 GB reads as "12"

**File:** `scripts/01_install_toolchain.sh:23,92`.

**Problem.** `df -BG` truncates to whole GiB (floor). Near the boundary the
guard is off by up to ~1 GB in the *unsafe* direction is not the issue (floor is
conservative for a `>=` check), but the *reported* "free disk now" at line 92 is
misleadingly coarse and, combined with the low 12 GB threshold, gives false
confidence. Also `--output=avail` measures `/` only; if `/usr/local`,
`/var`, or `$HOME` are separate mounts (common on workstations) the check
guards the wrong filesystem entirely.

**Fix.** Use MB precision and check the filesystem that actually backs the
install + build dirs:
```bash
avail_gb() { df -BM --output=avail "$1" | tail -1 | tr -dc '0-9' | awk '{printf "%d", $1/1024}'; }
FREE_GB="$(avail_gb /usr/local)"   # where cuda + libctranslate2 land
HOME_GB="$(avail_gb "${HOME}")"    # where the build tree + venv land
```

---

### [MEDIUM] 01:29 — driver-hold awk pattern misses `nvidia-firmware*` and `xserver-xorg-video-nvidia*`; relies on `|| true`

**File:** `scripts/01_install_toolchain.sh:29-34`.

**Problem.** The hold regex matches
`nvidia-driver|nvidia-dkms|nvidia-kernel|libnvidia|nvidia-compute|nvidia-utils`
but not `nvidia-firmware-*`, `nvidia-fabricmanager-*`, or
`xserver-xorg-video-nvidia-*`, any of which an errant `cuda-drivers` pull would
touch. More importantly the whole hold is best-effort (`|| true`, line 33): if
`apt-mark hold` silently fails, the script proceeds believing the driver is
protected. The *real* protection here is "never install a driver metapackage,"
which the script does correctly — but the hold is advertised (and in
BUILD_SPEC.md rule 1/3) as a guard, so it should not silently no-op.

**Fix.** Broaden the pattern and do not swallow the hold failure:
```bash
HOLD_PKGS="$(dpkg -l 2>/dev/null | awk '/^ii/ && $2 ~ /^(nvidia-(driver|dkms|kernel|compute|utils|firmware|fabricmanager)|libnvidia|xserver-xorg-video-nvidia)/ {print $2}')"
if [[ -n "${HOLD_PKGS}" ]]; then
  # shellcheck disable=SC2086
  sudo apt-mark hold ${HOLD_PKGS} || die "failed to hold driver packages; refusing to continue"
fi
```
(The before/after `nvidia-smi` driver check at 93 is the real backstop and is
correct — keep it.)

---

### [MEDIUM] 02:90 + 02:91 — wheel path `dist/*.whl` is non-empty-glob fragile; `ls -t | head -1` can grab a stale wheel on re-run

**File:** `scripts/02_build_ct2.sh:90-91,102`.

**Problem.** On a re-run, `python -m build` writes a new wheel into
`${SRC}/python/dist/` but does **not** clear old ones. `ls -t ... | head -1`
picks the newest by mtime, which is *usually* right, but if a previous run left
a wheel and the new build fails to overwrite (e.g. same version, build aborts
after metadata), you can silently package/install a **stale** wheel and never
notice. Also `WHEEL="$(ls -t .../dist/*.whl ...)"` under `set -e` with `pipefail`
will not error if `dist` is empty in the way the author expects — `ls` errors to
stderr and `head` succeeds with empty stdout, so `WHEEL` is empty and the
`[[ -n ... ]]` guard at 92 catches it; OK there, but the staleness is the real
trap.

**Fix.** Clear `dist/` before building so the only wheel present is the fresh one:
```bash
rm -rf "${SRC}/python/dist"
export CTRANSLATE2_ROOT=/usr/local
python -m build --wheel --no-isolation
shopt -s nullglob
wheels=("${SRC}/python/dist/"*.whl)
[[ ${#wheels[@]} -eq 1 ]] || die "expected exactly 1 fresh wheel, got ${#wheels[@]}"
WHEEL="${wheels[0]}"
```

---

### [MEDIUM] 03_validate.py:64 — `WhisperModel("tiny")` needs network to download from HF; a flaky/offline box reports FAIL that looks like an sm_50 failure

**File:** `scripts/03_validate.py:59-80,109-114`.

**Problem.** `WhisperModel(MODEL, ...)` downloads the model from Hugging Face on
first use. On the air-gapped or proxied workstation this raises (HF hub / network
error), the `except` catches it, `ok=False`, and since it is not a
"no kernel image" string it is reported as a generic error. With *all* cuda
cases failing for a **network** reason, `main()` prints
"FAIL: no CUDA case succeeded" — which a reader will misattribute to the sm_50
build, defeating the whole purpose of the validator (it would *lie* about the
cause). The sample-audio fetch has the same single-point-of-network-failure but
at least its message says "no network?".

**Fix.** Distinguish "model/network unavailable" from "GPU kernel missing", and
fail the *whole run* early with a clear message rather than mislabeling it a GPU
result:
```python
# in run_case, after building rec:
if "no kernel image" in msg or "NoKernelImage" in msg:
    rec["diagnosis"] = "sm_50 NOT in loaded ctranslate2 (wrong/clobbered build)"
elif any(k in msg.lower() for k in ("connection", "timed out", "huggingface", "couldn't find", "resolve")):
    rec["diagnosis"] = "MODEL DOWNLOAD/NETWORK failure — NOT a GPU/sm_50 result"
```
and in `main()`, treat an all-network-failure as a distinct non-PASS/non-FAIL
exit (e.g. exit 2 with "INCONCLUSIVE: could not load model") so it is not read as
a Maxwell failure. Optionally pre-cache the model (`huggingface-cli download
Systran/faster-whisper-tiny`) in 02.

---

### [MEDIUM] 03_validate.py:73 — RTF/print uses `getattr(dinfo,"duration",...)` but the column formatting will crash if a value is the string `"-"`

**File:** `scripts/03_validate.py:101-104`.

**Problem.** For a failed case `rec` has no `load_s`/`transcribe_s`/`rtf`, so the
print uses defaults `"-"`. The format spec `f'{r.get("load_s","-"):>7}'` formats a
str with `>7`, which is fine; but `f'{r.get("rtf","-"):>6}'` on a successful case
formats a `float` with `>6` — also fine. The actual latent bug: a *successful*
case stores numeric values, a *failed* case stores `"-"` (str); mixing `>7`
alignment of int/float vs str is legal in Python f-strings, so this does **not**
crash. Re-checked: no crash. Downgrading rationale below — this is LOW, not a
breakage. (Listed for completeness; see LOW-6.)

Reclassified to LOW — see LOW-6. (No fix required for correctness.)

---

### [LOW] BUILD_SPEC vs reality — CMake patch hunk is a **no-op** under the chosen `-DCUDA_ARCH_LIST="5.0"`

**Files:** `patches/1766-sm50.patch:5-13`, `BUILD_SPEC.md:49`, `scripts/02_build_ct2.sh:57`.

**Problem (informational, not a defect).** The patched
`elseif(CUDA_VERSION_MAJOR EQUAL 12)` block only executes when
`CUDA_ARCH_LIST STREQUAL "Common"` (CMakeLists.txt:532). The build passes an
explicit `"5.0"`, so that branch is skipped — but `"5.0"` is then handed
directly to `cuda_select_nvcc_arch_flags`, which emits the sm_50 gencode anyway.
**Net effect: sm_50 SASS is still produced correctly.** The CMake half of the
patch is redundant for *this* invocation; the load-bearing half is the `.cu`
`__CUDA_ARCH__ < 530` guard, which prevents the `sub.f16x2`/`fma.rn.f16x2` PTX
(nonexistent on sm_50) from being compiled for sm_50 and is what actually makes
the build succeed. The `grep -q 'CUDA_VERSION_MAJOR EQUAL 12'` marker check
(02:47) is therefore a *patch-applied* sanity check, not a guarantee the line
runs — which is fine, just worth stating so nobody "fixes" a non-bug.

**Note/optional fix.** None required. If you wanted the patch's CMake branch to
actually fire, you would build with `-DCUDA_ARCH_LIST="Common"` instead of
`"5.0"` — but that enlarges the build (more arches) against the disk-lean goal,
so keeping `"5.0"` is the right call. Leave as-is.

---

### [LOW] 02:39-44 — patch idempotency via `git apply --reverse --check` is correct, but a *partially* applied tree (one hunk applied, one not) is not handled

**File:** `scripts/02_build_ct2.sh:39-45`.

**Problem.** If a prior run applied the patch and was interrupted such that only
one of the two files changed (extremely unlikely with `git apply`'s atomicity,
but possible if someone hand-edited), `--reverse --check` fails (not fully
reverse-appliable) and `--check` also fails (not forward-appliable), so the
script `die`s with "patch does not apply cleanly". That is a *safe* failure but
the message misleads (it is actually "tree is in a mixed state").

**Fix.** Reset to a clean tag state before (re)applying, since the clone step
already checks out the tag:
```bash
git -C "${SRC}" checkout -f "${CT2_TAG}" -- CMakeLists.txt src/ops/awq/dequantize_gpu.cu
```
before the apply block, making the apply deterministic on every run.

---

### [LOW] 01:63 — `apt-cache madison` is not guaranteed sorted; relies on `sort -V | tail -1` (correct) but ignores epoch/`-1` revision in the pin

**File:** `scripts/01_install_toolchain.sh:63-67`.

**Problem.** `madison` prints e.g. `9.10.2.21-1`. The grep `^9\.10\.` + `sort -V`
+ `tail -1` correctly selects the highest 9.10.x, and the install pins the exact
string. Fine. One edge: if the repo lists the same version for both
`libcudnn9-cuda-12` and a different revision for `-dev`, pinning both to
`${CUDNN_VER}` (derived only from the runtime package's madison) can fail to
resolve if the `-dev` revision differs. In practice NVIDIA ships them lockstep,
so this is low risk.

**Fix (defensive).** Derive the dev version independently or drop to
major.minor matching with `--allow-downgrades` already present:
```bash
DEV_VER="$(apt-cache madison libcudnn9-dev-cuda-12 | awk '{print $3}' | grep -E '^9\.10\.' | sort -V | tail -1)"
sudo apt-get install -y --no-install-recommends --allow-downgrades \
  "libcudnn9-cuda-12=${CUDNN_VER}" "libcudnn9-dev-cuda-12=${DEV_VER}"
```

---

### [LOW] Dockerfile:104-122 — base image may NOT carry the NVIDIA apt repo/keyring; the cuDNN stage assumes it does

**File:** `docker/Dockerfile:104-122`.

**Problem.** The comment asserts "the 12.9.1-devel base already carries the
NVIDIA CUDA apt repo + keyring." This is true for `nvidia/cuda:*-devel-*` images
historically, but it is an assumption; if a future base drops the repo, the
`apt-cache madison libcudnn9-cuda-12` guard correctly fails loudly (it `exit 1`s
with a clear message), so this degrades gracefully. Listed only because the
fallback advice ("add cuda-keyring before this step") is in a comment, not code.

**Fix (optional).** Make the keyring add explicit and idempotent rather than
assumed, mirroring 01:
```dockerfile
RUN test -f /etc/apt/sources.list.d/cuda*.list || ( \
    curl -fsSL -o /tmp/k.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/k.deb && rm -f /tmp/k.deb )
```

---

### [LOW] Dockerfile:177-179 / 127 — `--break-system-packages` into the base system Python is intended, but `auditwheel` excludes may strip a lib the wheel genuinely needs at import

**File:** `docker/Dockerfile:187-204`.

**Problem.** `auditwheel repair` excludes `libcudart/libcublas/libcublasLt/`
`libcudnn*/libnvrtc`. That is correct for a "thin" wheel whose CUDA runtime is
host-provided. But `auditwheel` will then likely *fail* the manylinux policy
(those excluded libs are non-whitelisted external deps), which is why the
`|| cp raw wheel` fallback exists — meaning the shipped wheel is frequently the
**raw, unrepaired** wheel with an rpath of `/usr/local/lib`. That raw wheel only
imports if `libctranslate2.so` is found at runtime (rpath + host
`LD_LIBRARY_PATH`). For the documented "install on the frozen host" flow this is
fine, but the README's "pip install the release wheel directly" (README:144)
omits that the wheel may be unrepaired and needs `libctranslate2.so` present.

**Fix (doc).** State in README that the release wheel bundles `libctranslate2.so`
only if `auditwheel repair` succeeded; otherwise it requires the C++ lib
installed (i.e. you ran `02`). No code change required.

---

### [LOW-6] 03_validate.py:101-104 — mixed numeric/str column values (re-classified from the MEDIUM above)

Confirmed **not** a crash in Python 3 f-string formatting (`f'{x:>6}'` works for
both `float` and `str`). No fix needed. Recorded so the earlier MEDIUM entry is
not double-counted: it is LOW/no-op.

---

## Bottom line

- The **driver-safety design is sound**: toolkit-only install, no
  `cuda`/`cuda-drivers` metapackage, `--no-install-recommends`, before/after
  `nvidia-smi` check with `die` on change, cuDNN pinned to 9.10.x with a loud
  `die` if only 9.11+ is served. Verified that `cuda-toolkit-12-9` does not pull
  a driver.
- The **patch is upstream-faithful (identical to PR #1766), applies cleanly to
  v4.8.0, targets the right file, and is idempotent.** The `.cu` guard is the
  load-bearing piece and is correct.
- The **wheel/clobber flow is correct** (ct2 4.8.0 satisfies
  `faster-whisper`'s `>=4.0,<5`; final `--force-reinstall --no-deps` wins).
- **The one HIGH is disk:** the 12 GB guard is too low for toolkit + cuDNN + a
  CUDA build tree on a 31 GB-free box, and a `sudo cmake --install` ENOSPC would
  leave `/usr/local` in a messy half-installed state on the production machine.
  Trim the toolkit and/or raise the guard and re-check before the build, and
  clean the apt cache before building — then this scaffold is safe to run.

Because a HIGH existed, the original status was **FAIL**. The author has since
addressed the disk HIGH (and all 5 MEDIUMs); see the **Re-review** section below.
Post-fix status: **REVIEW STATUS: PASS**.

---

## Re-review (2026-06-17)

Adversarial re-review of the author's fixes. Each prior finding was re-verified
against the edited `01_install_toolchain.sh`, `02_build_ct2.sh`, and
`03_validate.py`, and the edited regions were re-read as shell/Python looking for
regressions. Key shell/Python mechanisms were executed in isolation to confirm
runtime behavior (MB->GB awk math, `nullglob` array under `set -euo pipefail`,
the driver-hold awk regex, `madison_910` empty/non-empty paths, and the full
`main()` exit-code decision tree).

**Outcome: no HIGH remains and no new HIGH/MEDIUM regression was introduced.**
Remaining HIGH = 0, remaining MEDIUM = 0. New HIGH = 0, new MEDIUM = 0.
One new LOW (cosmetic duplication) noted below; it does not affect status.

### Per prior finding

- **[HIGH] Disk exhaustion mid-build — RESOLVED.**
  - `01:25` defines `avail_gb() { df -BM --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9' | awk '{printf "%d", $1/1024}'; }` — defined *before* first use at `01:26`. The MB->GB conversion is correct (floor division by 1024; verified 30000M->29, 25600M->25, 11878M->11). It measures `/usr/local` (the install target) and `${HOME}` (build tree), addressing the "wrong filesystem" sub-concern of the MEDIUM below as well.
  - `01:29` now guards `>= 25 GB` on `/usr/local`, with a clear `die`.
  - `01:71` runs `sudo apt-get clean` immediately after the toolkit install and *before* cuDNN, so toolkit `.deb`s + cuDNN `.deb`s + build objects never stack at peak. (A second `apt-get clean` remains at end-of-script `01:95`.)
  - `02:51-53` re-checks `>= 12 GB` on `${HOME}` right before `rm -rf build`, with `die`. The inline `df -BM ... awk` matches the `01` helper's math.
  - Verdict: RESOLVED.

- **[MEDIUM] `df -BG` floor + wrong-filesystem — RESOLVED.** Switched to `df -BM` with MB precision and now probes both `/usr/local` and `${HOME}` separately (`01:26`), so a split-mount workstation is guarded on the filesystems that actually fill. Verdict: RESOLVED.

- **[MEDIUM] Driver-hold regex too narrow + silent `|| true` — RESOLVED.** `01:33` regex now also matches `nvidia-firmware`, `nvidia-fabricmanager`, and `xserver-xorg-video-nvidia` (in addition to driver/dkms/kernel/compute/utils/libnvidia). Executed against a simulated `dpkg -l`: all 9 relevant `ii` lines matched, the `rc` (config-only) line was correctly excluded, and `build-essential`/`libcudnn9` were not matched. The hold at `01:37` now ends in `|| die ...` (no silent no-op). Verdict: RESOLVED.

- **[MEDIUM] cuDNN `-dev` version pinned to runtime's madison — RESOLVED.** `01:74` `madison_910()` is a correct helper (`apt-cache madison "$1" | awk '{print $3}' | grep -E '^9\.10\.' | sort -V | tail -1`). `CUDNN_DEV_VER` is derived *independently* from `libcudnn9-dev-cuda-12` (`01:76`) and used for the `-dev` package at `01:80`. The empty-result `die` at `01:77` now requires *both* runtime and dev to be non-empty. Verified under `set -euo pipefail`: the `$(... || true)` capture does not abort on a no-9.10 match, and the `die` fires correctly. Verdict: RESOLVED.

- **[MEDIUM] Stale wheel via `ls -t | head -1` — RESOLVED.** `02:94` `rm -rf "${SRC}/python/dist"` before `python -m build`; `02:96-99` use `shopt -s nullglob`, glob into a `wheels=(...)` array, and `die` unless `${#wheels[@]} -eq 1`. Executed under `set -euo pipefail`: the empty-array assignment does NOT trigger `set -e` (array assignment of an empty glob is exit-0), the 0/2-wheel cases hit the `die` branch, and exactly-1 selects `wheels[0]`. Valid. Verdict: RESOLVED.

- **[MEDIUM] Validator mislabels network failure as a Maxwell/sm_50 FAIL — RESOLVED.** `03:79-85` `run_case` now sets `kind` on every error branch: `no_sm50` for "no kernel image"/"nokernelimage", `network` for a broadened keyword set (connection/timed out/huggingface/couldn't find/resolve/max retries/network/offline), else `other`. `main()` (`03:115-132`) returns `0`/PASS if any cuda case is ok; `2`/INCONCLUSIVE only when `cuda and all(kind=="network")`; `1`/FAIL `no_sm50` if any cuda case is `no_sm50`; else `1`/FAIL generic. Traced all branches in Python: all-network->2, all-no_sm50->1, **mixed network+no_sm50 correctly ->1 (NOT INCONCLUSIVE)** because `all(kind=="network")` is False, one-ok->0, single-other->1. The success path leaves `kind` unset but `ok=True` short-circuits first, so `r.get("kind")` returning `None` is harmless. Verdict: RESOLVED.

- **[LOW] Patch idempotency / mixed-tree — RESOLVED.** `02:41` now `git checkout -f -- CMakeLists.txt src/ops/awq/dequantize_gpu.cu` restores both patched files to the checked-out tag state before `git apply --check`/`git apply`, so re-runs and partially-applied trees are deterministic. Verdict: RESOLVED.

- **[LOW] `madison` ignores `-dev` revision differences — RESOLVED** (folded into the cuDNN-dev MEDIUM fix above; `CUDNN_DEV_VER` is now independent). Verdict: RESOLVED.

- **[LOW] CMake hunk is a no-op under `-DCUDA_ARCH_LIST="5.0"`** — informational only; unchanged and correct as-is. Verdict: N/A (no fix was required).

- **[LOW] Dockerfile keyring assumption / [LOW] auditwheel raw-wheel doc / [LOW-6] mixed numeric-str column** — not in scope of the three edited scripts; unchanged. The LOW-6 format-string non-crash was re-confirmed in Python 3.12 (`f'{x:>6}'` formats both `float` and `'-'`). Verdict: unchanged / N/A.

### Regression scan of edited regions

- `01` `avail_gb` and the `02` inline disk check: correct awk; both define/inline before use; no unbound-var risk (awk `printf "%d"` always emits at least `0`, so the captured var is never empty on a real mounted path). No regression.
- `01` driver-hold and `madison_910`: both safe under `set -euo pipefail` (the `$(... || true)` captures and `[[ -n ... ]]` guards prevent premature abort). No regression.
- `02` `git checkout -f --` then `git apply`: ordering is correct (restore, check, apply, marker-grep). No regression.
- `02` `nullglob` array: valid under `set -e`; no regression.
- `03` control flow: every `run_case` error branch sets `kind`; `main()` decision tree is exhaustive and ordered correctly (PASS -> INCONCLUSIVE -> no_sm50 FAIL -> generic FAIL). No regression.

### New issue introduced by the fixes

- **[LOW] Disk-check math is duplicated rather than shared.** `02:51` inlines the same `df -BM ... | awk '{printf "%d", $1/1024}'` pipeline that `01:25` factors into `avail_gb()`. Cosmetic only (the two copies currently agree). **Exact fix (optional):** add `avail_gb() { df -BM --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9' | awk '{printf "%d", $1/1024}'; }` near the top of `02_build_ct2.sh` and replace the `02:51` inline with `BUILD_FREE_GB="$(avail_gb "${HOME}")"`. Not status-affecting.

### Verdict

All 1 HIGH and 5 MEDIUM prior findings are RESOLVED; the 2 actionable LOWs
(idempotency, cuDNN-dev pin) are RESOLVED; the remaining LOWs were informational
and unchanged. No regressions of MEDIUM-or-higher severity. One new cosmetic LOW.
**REVIEW STATUS: PASS.**
