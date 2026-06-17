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

---

## Re-review #3 (benchmark + publish-readiness)

Adversarial review of `bench/run_bench.py` (NEW, not yet run), the Phase-4
single-run validation numbers, the author's intended published conclusion, and
the two `(cd /tmp && python -c ...)` cwd-shadow verify lines in `02_build_ct2.sh`.
Scope: would the PUBLISHED numbers/claims be wrong or misleading under the
author's real name. I verified the load-bearing facts upstream rather than
asserting from memory (sources at the end of this section).

RE-REVIEW #3 STATUS: PASS
(was FAIL; the author re-ran a rigorous median-of-5 benchmark and rewrote the
README. All 3 HIGH and 5 MEDIUM concerns are addressed in the published numbers.
See "## Re-review #3 FINAL (numbers verified)" at the end of this file.)

Counts (original): HIGH = 3, MEDIUM = 5, LOW = 4.
(FAIL because three HIGHs are present: the published "~2.3x" and "~4s CUDA init"
claims are derived from a single unverified run and a mislabeled cause, and the
single-11s-clip throughput claim is not supported by the data the bench produces.)

---

### What I verified upstream (so this is not vibes)

- **faster-whisper `transcribe()` returns a LAZY generator.** Return type is
  `Tuple[Iterable[Segment], TranscriptionInfo]`. The actual encoder/decoder GPU
  work runs *during iteration* of the segments generator (inside
  `_*segments_generator` -> `self.forward()`), NOT when `transcribe()` returns.
  `TranscriptionInfo` (incl. `.duration`) is computed eagerly before the
  generator is yielded (VAD/feature-extract/lang-detect run up front).
  => Consequence for the bench: `" ".join(s.text for s in segments)` **does**
  force the full transcription, so the timed region is real work, not a no-op.
  This part is correct in both `03` and `run_bench.py`. (Source: faster-whisper
  `transcribe.py`, master.)
- **CTranslate2 GPU execution is synchronous from Python's view per call.** The
  Python `generate`/Whisper path returns materialized host-side results
  (token ids / `Segment.text`), which forces a device->host copy and therefore a
  stream sync before each `Segment` is yielded; consuming the generator to
  completion blocks on all GPU compute. => The wall-clock around the `for s in
  segments` consumption captures real GPU compute; no explicit
  `torch.cuda.synchronize()` is needed here because faster-whisper/CT2 do not
  expose a lazy CUDA tensor to Python. This is the one thing that, had it been
  false, would have invalidated every GPU number; it holds.
- **`cpu_threads` default = 0 -> CT2 `intra_threads=0`.** faster-whisper
  `WhisperModel.__init__` defaults: `device="auto"`, `compute_type="default"`,
  `cpu_threads=0`, `num_workers=1`; `cpu_threads` is passed straight to CT2
  `intra_threads`. CT2 `intra_threads=0` does NOT mean "all cores": it honors
  `OMP_NUM_THREADS` if set, else picks a small default (historically capped at
  ~4). **CT2 4.8.0 (our frozen pin) has a known `intra_threads=0` thread bug
  (OpenNMT/CTranslate2#2063): on some platforms it oversubscribes (~1470% CPU)
  and runs pathologically slower than `intra_threads=1`.** This directly
  threatens the CPU baseline's representativeness and is unreported by the bench
  (see [HIGH-3]). (Sources: faster-whisper `transcribe.py`; CT2 docs
  `parallel.html`/`performance.md`; CT2 issue #2063.)

---

### Findings

#### [HIGH-1] The published "~2.3x faster" comes from a SINGLE Phase-4 run, not from the median-of-N the bench produces — the README must not publish the 0.59/1.36 single-shot pair

**Where:** intended README conclusion ("0.59 vs 1.36s ... ~2.3x"); data source is
the Phase-4 `03_validate.py` run (single transcription, no warmup, beam_size=1).

**Problem.** `03_validate.py` times exactly one transcription per case with **no
warmup**. The 0.59s GPU number therefore includes first-call effects (cuDNN
algorithm selection / autotune, first-kernel load, lazy CUDA module load), and
the 1.36s CPU number includes cold model-weight page-in and the CT2 thread
ramp. A ratio of two cold single shots is not a steady-state throughput ratio,
and "~2.3x" is stated to 2 significant figures off of n=1 with no dispersion.
The whole reason `run_bench.py` exists (warmup + median-of-N + separated load)
is to replace exactly these numbers — so publishing the *validation* numbers
pre-empts and contradicts the rigorous bench.

**Fix.** Publish ONLY `run_bench.py` median-of-N numbers, and report the speedup
as `median(cpu)/median(gpu)` with the actual N and the per-run spread (min-max or
all `runs[]`). Do not print a 2-sig-fig "2.3x" from n=1. If the bench's median
ratio lands near 2.3x, state it as e.g. "GPU float32 ~2.0-2.5x faster than
CPU-int8 in steady state (median of N=… runs, tiny model, 11s clip)". Raise
`CT2_BENCH_REPEATS` to >= 5 for a publishable median (3 is the floor; with 3,
median == the middle of 3 and is noise-sensitive).

#### [HIGH-2] "~4s one-time CUDA init" is a mislabeled cause — `load_s` is WhisperModel construction (weights to GPU + context + first cuDNN/cuBLAS handle), not "CUDA init"; do not attribute it to one thing

**Where:** intended README conclusion ("pays a ~4s one-time CUDA init").

**Problem.** The 4.04s `load_s` from Phase-4 is the wall time of
`WhisperModel("tiny", device="cuda", ...)` plus the **first** transcription is
NOT inside it — but `load_s` itself bundles: (a) CUDA context / primary-context
creation, (b) loading + casting model weights and copying them host->device,
(c) lazy `libcudart`/`libcublas`/`libcudnn` load and first handle creation, and
on some stacks (d) JIT/PTX work if any kernel lacks the exact SASS. Calling all
4s "CUDA init" is a specific causal claim that the data does not isolate — most
of a tiny-model GPU load on a 4GB Maxwell over PCIe is plausibly weight transfer
+ cuBLAS/cuDNN handle creation, not bare context init (bare `cuInit`/context is
typically a few hundred ms). Publishing "4s CUDA init" under a real name invites
a correct "that's not what that measures" reply.

**Fix.** Describe it as measured, not as mechanism: "a one-time model-load /
GPU-warmup cost of ~Xs (CUDA context + weights to VRAM + first cuDNN/cuBLAS
setup), measured as `load_s`, paid once per process." Report the bench's actual
median `load_s`, not the 4.04s single sample. If the author wants to claim it is
*mostly* context init, that requires isolating it (e.g. time a bare
`ctranslate2`/CUDA context create with no model) — which the bench does not do,
so the claim should be dropped, not asserted.

#### [HIGH-3] An 11s clip + unpinned/unreported CPU thread count makes the CPU baseline (and therefore the GPU-vs-CPU ratio) non-reproducible and possibly pathological on CT2 4.8.0

**Where:** `bench/run_bench.py` (single `sample.wav`, `CONFIGS` cpu cases,
no thread pinning, no `OMP_NUM_THREADS` set/reported); intended README throughput
claim ("for sustained/batch/long audio the GPU wins on throughput").

**Problem (two compounding issues).**
1. **Per-call fixed overhead dominates an 11s clip.** At RTF ~0.05-0.14 the
   actual compute is 0.6-1.6s, a large fraction of which is fixed per-call cost
   (VAD, feature extraction, generator setup, a single tiny encoder pass).
   A throughput/"GPU wins on long audio" claim **cannot be supported by an 11s
   clip** — that is precisely the regime where fixed overhead, not steady
   throughput, is being measured. The author's own conclusion ("for
   sustained/batch/long audio the GPU wins on throughput") is an *extrapolation*
   the bench does not measure.
2. **CPU thread count is neither pinned nor reported.** `cpu_threads` defaults to
   0 -> CT2 `intra_threads=0`. On the frozen pin **CT2 4.8.0 this is the
   #2063-buggy path** (oversubscribe / pathological slowdown on some platforms;
   elsewhere it silently uses `OMP_NUM_THREADS` or a small cap). So the published
   CPU seconds depend on an environment variable the bench never sets or records,
   and could be a worst-case oversubscribed number on a different machine — which
   would make "GPU 2.3x faster than CPU" an artifact of a mis-threaded CPU run,
   not a hardware fact.

**Fix.**
- For an honest throughput claim, bench at least one **long / concatenated clip**
  (e.g. 60-300s; concatenate the JFK clip xN or use a longer public speech
  sample) and report RTF there; keep the 11s clip only as a "short-clip latency"
  data point and label it as such. State explicitly that the GPU-wins-on-long-
  audio claim is supported (or not) by the long-clip RTF, not the 11s one.
- Pin and report CPU threads: set `cpu_threads=os.cpu_count()` (or a fixed value)
  explicitly in `WhisperModel(...)` for cpu cases, OR `export OMP_NUM_THREADS=N`,
  and print the effective thread count + `OMP_NUM_THREADS` in the bench header
  and the README caveat. Given #2063, do NOT leave `intra_threads=0` for a
  published CT2-4.8.0 CPU number — set it explicitly.

#### [MEDIUM-1] `runs` records the timed list but the median can be the warmup-excluded but still cold-ish first repeat; with REPEATS=3 the published median is noise-sensitive

**Where:** `run_bench.py:50-52` (one warmup, then `REPEATS=3` timed).

**Problem.** One warmup is good, but a single warmup on a 4GB Maxwell may not
fully settle cuDNN autotune / clocks; and `statistics.median` of 3 is the middle
value — one slow GC/throttle blip makes the published median that blip. The math
is *correct*; the sample size is just thin for publication.

**Fix.** Default `CT2_BENCH_REPEATS` to >= 5 (ideally 10) for the published run;
report `min`/`median`/`max` (or all `runs[]`, which it already stores — just
surface them in the README table). Keep `transcribe_s` = median but show spread.

#### [MEDIUM-2] No thermal/throttle or clock guard, and no cold-vs-warm filesystem-cache control for the model load

**Where:** `run_bench.py` overall; `load_s` semantics.

**Problem.** On a small passively/throttle-prone workstation GPU, back-to-back
configs can warm the card so later configs (or later models) run at different
clocks; and the FIRST `WhisperModel(...)` of a model size pays cold-page model
read from disk while the second pays warm page cache — so comparing `load_s`
across the run is apples-to-oranges. Not a correctness bug, but it biases the
*one-time cost* story the README wants to tell.

**Fix.** (a) Note in the README that `load_s` is cold-cache-sensitive and was
measured warm/cold (state which); (b) optionally log GPU temp/clocks
(`nvidia-smi --query-gpu=temperature.gpu,clocks.sm`) before each case; (c) run
GPU and CPU cases in separate process invocations if you want clean,
non-cross-contaminated context state. At minimum, disclose the order-dependence.

#### [MEDIUM-3] The bench summary's GPU-vs-CPU verdict compares GPU float32 only against CPU **int8**, silently dropping CPU float32 — and the README conclusion says "vs CPU" (1.36 = cpu float32) while the bench computes vs cpu int8 (1.57)

**Where:** `run_bench.py:95-100` (`cpu = row.get("cpu/int8")`); vs the intended
README "0.59 vs 1.36" (1.36 is cpu **float32**, not int8).

**Problem.** Inconsistent baseline. The bench's printed verdict divides by
`cpu/int8` (1.57s in Phase-4 -> ~2.66x), but the author's prose uses `cpu/float32`
(1.36s -> ~2.31x). A reader cross-checking the JSON against the prose will find
the "2.3x" doesn't match the script's own "vs CPU-int8" line. Pick ONE baseline
and be explicit, or report both. (CPU int8 is the *faster* CPU path to beat for a
fair "is the GPU worth it" framing; CPU float32 is the apples-to-apples compute
type. Both are defensible; mixing them silently is not.)

**Fix.** In the README state the baseline compute type every time
("GPU float32 vs CPU int8: …x; vs CPU float32: …x"), and make the bench print
both ratios. Do not write "vs CPU" unqualified.

#### [MEDIUM-4] `x_realtime`/`rtf` use `info.duration` (post-VAD *content* duration), which can be shorter than the wall audio — fine, but it must be disclosed so RTF isn't mistaken for wall-clock-over-file-length

**Where:** `run_bench.py:48,57-59`; `info.duration`.

**Problem.** faster-whisper's `TranscriptionInfo.duration` is the processed audio
duration; with VAD it can differ from the file's wall length. RTF computed
against it is a legitimate metric but is "compute-time per second of *processed*
audio," not "per second of *file*." For an 11s clip with speech throughout these
nearly coincide, but the README should say which, or a careful reader will
recompute RTF from the 11s file length and get a different number.

**Fix.** Either disable VAD for the bench (faster-whisper default has VAD off
unless `vad_filter=True`, so likely already fine — verify and state "VAD off,
duration == file length") or print both `audio_s` and the raw file seconds and
define RTF against the file length in the README.

#### [MEDIUM-5] `bench_one` swallows ALL exceptions into `ok=False` with a 180-char string — a partial/garbage transcription or an OOM mid-stream can be recorded as a clean FAIL or, worse, a short truthy `warm_text` can mark `ok=True` on a truncated result

**Where:** `run_bench.py:54,62-63`; `ok=bool(warm_text)`.

**Problem.** `ok` is set from the *warmup* text being non-empty. If the warmup
produces one stray token (e.g. partial decode before an OOM on the 4GB card on a
larger model), `ok=True` is recorded and the timed runs may then differ or fail
silently inside the median. Also `small` is in the default `MODELS` and may OOM
or behave differently on 4GB under float32 GPU — a degraded result could be
published as a valid row. The broad `except` hides the kind (OOM vs no_sm50 vs
network), unlike `03` which classifies.

**Fix.** (a) Assert the transcript is non-trivial (e.g. len > some chars, or
compare against the known JFK text for the canonical clip) before `ok=True`;
(b) classify the exception kind like `03` does (OOM / no_sm50 / network / other)
so a published "FAIL" row is attributable; (c) consider dropping `small` from the
*default published* set unless you confirm it fits float32 in 4GB, or label per-
model VRAM headroom.

#### [LOW-1] `02` cwd-shadow fix via `(cd /tmp && python -c ...)` is CORRECT and sufficient

**Where:** `02_build_ct2.sh:112,116`.

**Assessment (not a defect).** Both verify lines run from `/tmp`, so `sys.path[0]`
is `/tmp`, not `${SRC}/python` (which contains a `ctranslate2/` source dir that
would shadow the installed wheel and lack the compiled `_ext`). `/tmp` has no
`ctranslate2` dir, so `import ctranslate2` resolves to the force-reinstalled
site-packages wheel — which is exactly what we want to assert. The subshell also
avoids leaking the `cd` into the rest of the script. **Correct and sufficient.**
One nit: it assumes the venv `python` is on PATH inside the subshell; since
`source venv/bin/activate` ran earlier in the same shell and `(...)` inherits the
environment (PATH included), this holds. No fix required.

#### [LOW-2] `CT2_PROJ` default in the bench (`~/projects/ct2-maxwell-final`) differs from where the repo appears to live; a published run must pin it

**Where:** `run_bench.py:29`; `03_validate.py:25`.

**Problem.** Default `PROJ` is `~/projects/ct2-maxwell-final`. If the author runs
from elsewhere without `CT2_PROJ` set, `sample.wav`/`benchmark_results.json` land
in a different tree than expected and the published JSON path is ambiguous.

**Fix.** State the exact `CT2_PROJ`/cwd used for the published run in the README,
and have the bench print `PROJ` in its header.

#### [LOW-3] `audio_s` rounded to 2 decimals then used as RTF denominator — negligible, but `max(audio_s, 1e-6)` guard means a failed-duration case yields RTF ~1e6 silently

**Where:** `run_bench.py:48,58`.

**Problem.** If `info.duration` is missing (`getattr(... , 0.0)`), RTF becomes
`med / 1e-6` = a huge number rather than an obvious error. The `ok` gate mostly
prevents this from being printed, but a successful transcribe with a 0 duration
(shouldn't happen, but) would publish a nonsense RTF.

**Fix.** If `audio_s <= 0`, mark the row `ok=False`/`rtf=None` rather than
emitting a 1e6 RTF.

#### [LOW-4] K2200-4GB -> 940MX-2GB extrapolation honesty

**Where:** README "Who is affected" / VRAM caveat; BUILD_SPEC deploy target.

**Problem.** The bench/validation run on a Quadro K2200 (4GB). The README already
warns 940MX has only 2GB and "do not assume a model that runs on the K2200 also
runs on a 2GB card" — good. But the *performance* numbers are K2200-only; the
940MX is a different SKU (different SM count, memory bandwidth, boost clocks) and
will not match the K2200 RTF.

**Fix.** Add one line: "All benchmark numbers are from the Quadro K2200 (4GB);
the 940MX (2GB) is the same sm_50 ISA but a different, slower part — expect
different RTF and tighter VRAM. Numbers are not transferable between cards; run
`bench/run_bench.py` on your own card."

---

### Recommended exact wording for the README benchmark caveat

> **Benchmark caveat (read before trusting these numbers).** All numbers are from
> a single Quadro K2200 (Maxwell sm_50, 4 GB, driver 580.159.03) on Ubuntu 24.04,
> CTranslate2 v4.8.0 + PR #1766, faster-whisper, beam_size=1, VAD off. Each figure
> is the **median of N=… timed runs after one warmup**; the one-time
> `WhisperModel(...)` load (CUDA context + weights to VRAM + first cuDNN/cuBLAS
> setup) is reported **separately** as `load_s` and is *not* a per-clip cost.
> CPU runs used `cpu_threads=…` (`OMP_NUM_THREADS=…`) — CTranslate2 4.8.0's
> default `intra_threads=0` can mis-thread (issue #2063), so we pin it. On sm_50
> there is no native FP16 and no `dp4a` int8, so CUDA int8/float16 fall back to
> float32 and are not benchmarked. The short-clip (11 s) figures measure
> **latency** (fixed per-call overhead dominates); the long-clip (… s) figures
> measure **throughput** — only the latter supports any "GPU wins on long/batch
> audio" statement. The 940MX (2 GB) is the same ISA but a different, slower part:
> these numbers do not transfer. Reproduce on your own card with
> `bench/run_bench.py` before deciding the GPU is worth it.

And the conclusion line should read (only after the bench is actually run):

> On the K2200, in steady state GPU float32 transcribes a short clip about
> **<median-ratio>x faster than CPU <int8|float32>** (median of N runs:
> <gpu_med>s vs <cpu_med>s), but it pays a one-time ~<load_med>s model-load/
> GPU-warmup cost per process. For a single short clip that one-time cost can make
> CPU win end-to-end; whether the GPU wins on sustained/long audio is answered by
> the long-clip RTF row, not the 11 s clip.

It should **NOT** claim: a bare "2.3x faster than CPU" (ambiguous baseline,
n=1); "~4s CUDA init" (mislabeled cause, unisolated); or any throughput / "wins on
long audio" statement backed only by the 11 s clip.

---

### Re-review #3 sources

- faster-whisper `transcribe.py` (master): `transcribe()` returns
  `Tuple[Iterable[Segment], TranscriptionInfo]`, segments lazy, info eager;
  `WhisperModel.__init__` defaults `cpu_threads=0`, `num_workers=1`,
  `device="auto"`, `compute_type="default"`; `cpu_threads -> intra_threads`.
- CTranslate2 docs `parallel.html` / `performance.md`: `intra_threads`/
  `inter_threads` semantics; `0` -> default (honors `OMP_NUM_THREADS`, small cap),
  not all cores; total threads should not exceed physical cores.
- OpenNMT/CTranslate2 issue #2063: `intra_threads=0` on 4.8.0 oversubscribes
  (~1470% CPU) / pathologically slow vs `intra_threads=1` on some platforms —
  the frozen pin's CPU-baseline risk.

### Re-review #3 verdict

**RE-REVIEW #3 STATUS (original): FAIL** (3 HIGH). The build, the cwd-shadow verify lines,
and the *mechanics* of `run_bench.py` (lazy-generator forcing, median math,
load/steady-state separation) are sound. The FAIL is about **what gets
published**: the intended "2.3x / 4s CUDA init / GPU wins on long audio" sentence
is built on a single un-warmed validation run, a mislabeled load cause, and an
11 s clip that cannot support a throughput claim, with an unpinned CT2-4.8.0 CPU
thread count underneath. Run `run_bench.py` (REPEATS>=5, pinned CPU threads, plus
a long clip), publish only its medians with the baseline named and the caveat
above, drop the "4s CUDA init" mechanism claim, and this becomes publishable.

---

## Re-review #3 FINAL (numbers verified)

The author re-ran a rigorous benchmark (median of 5 timed runs after one warmup,
K2200 sm_50 4 GB, Ubuntu 24.04, CT2 v4.8.0 + PR #1766, faster-whisper, beam=1,
VAD off, `cpu_threads=4`) and rewrote the README "Is the GPU worth it on Maxwell?
(measured)" section. I transcription-checked every published number against the
measured dataset and re-derived every arithmetic value. All checks OK.

### Number transcription (throughput table, 66 s clip) — OK

Re-derived `round(med,2)`, `round(rtf,3)`, `round(x_rt,1)` for all 6 rows; all
loads cross-checked too. Zero typos.

- tiny  cuda f32: load 0.86 / 7.40 / 0.112 / 8.9x — OK
- tiny  cpu  int8: load 0.53 / 32.92 / 0.499 / 2.0x — OK
- tiny  cpu  f32: load 0.44 / 29.35 / 0.445 / 2.2x — OK
- small cuda f32: load 9.88 / 27.36 / 0.415 / 2.4x — OK
- small cpu  int8: load 0.89 / 140.51 / 2.129 / 0.5x — OK
- small cpu  f32: load 1.15 / 113.74 / 1.723 / 0.6x — OK

### Number transcription (latency table, 11 s clip) — OK

Displayed transcribe medians `round(med,2)`: 0.19 (0.186), 0.82 (0.819),
1.20 (1.198), 5.33 (5.331) — all OK. Loads match.

### End-to-end (load + 1 clip) column recompute — OK

- tiny  cuda f32: 0.86 + 0.186 = 1.046 -> ~1.0s — OK
- tiny  cpu  f32: 0.44 + 0.819 = 1.259 -> ~1.3s — OK
- small cuda f32: 9.88 + 1.198 = 11.078 -> ~11.1s — OK
- small cpu  int8: 0.89 + 5.331 = 6.221 -> ~6.2s — OK

### Headline ratios — OK

32.922 / 7.401 = 4.45x (tiny), 140.508 / 27.363 = 5.13x (small). README states
"tiny 4.45x, small 5.13x," correctly labeled "GPU vs CPU-int8 on the 66 s clip."
Both exact.

### Claim audit — all literally supported, none overstated

- **"4–5x faster than the CPU baseline on a 66 s clip" / "GPU wins decisively
  (4–5x)"** — OK. Anchored explicitly to the CPU-int8 baseline (4.45x / 5.13x,
  which bracket 4–5x). vs CPU-float32 the ratios are 3.97x / 4.16x, but the
  README names the int8 baseline every time, so the 4–5x figure is the
  conservative-direction claim. Not overstated. Confined to the 66 s row, which
  is the only one that supports a throughput claim.
- **"CPU only wins for a single one-off short clip with a larger model"** — OK,
  and the "larger model" qualifier is CORRECT and load-bearing. Verified against
  the trap the prompt flagged: tiny short end-to-end is GPU 1.05s vs CPU 1.26s,
  so **GPU wins tiny short too**. CPU only wins end-to-end for `small` (GPU
  11.08s vs CPU 6.22s). Without "larger model" the claim would be false for tiny;
  with it, it is exactly right.
- **"int8 was slower than float32 on this CPU"** — OK. int8 > float32 transcribe
  time in all four CPU pairs (32.92>29.35, 140.51>113.74, 0.868>0.819,
  5.331>4.880). README correctly hedges "measure it."
- **"small CPU cannot keep up with real time (RTF > 1) while the GPU stays at
  ~0.4 RTF"** — OK. small cpu int8 RTF 2.129>1, small cpu f32 RTF 1.723>1, small
  cuda RTF 0.415 ≈ 0.4.
- **"~10 s one-time load" (small GPU) / "amortizes immediately over any repeated
  use"** — OK. Measured small cuda load 9.88s ≈ 10s. Per-clip, small GPU
  (1.198s) beats CPU-int8 (5.331s), so only the one-time load makes CPU win
  end-to-end; the amortization claim is correct.
- **sm_50 lede: "no native FP16, no `dp4a` int8, float32 the only supported CUDA
  compute type, int8/float16 fall back to float32"** — OK. Matches
  `get_supported_compute_types('cuda') == {'float32'}`. The bench only runs
  `(cuda, float32)` for GPU, consistent with the fall-back statement; no
  benchmarked cuda-int8 number is claimed.

### Methodology provenance (run_bench.py) matches the README header — OK

median-of-5 after one warmup (lines 89, 94-95); `cpu_threads=4` passed for CPU
cases (line 74) + `OMP_NUM_THREADS` pinned pre-import (line 40); `beam_size=1,
vad_filter=False` (line 85); RTF = med/dur, x_rt = dur/med (lines 100-101); long
clip = short concatenated to ~66 s (LONG_REPEAT=6, lines 45/51-61); ratio printed
as cpu_int8/gpu on the long clip (line 148). All consistent with the published
numbers and prose.

### Caveat paragraph — sufficient and accurate (OK)

Covers: single-machine/single-K2200 scope; "940MX is the same sm_50 ISA but a
different, slower, 2 GB part — these numbers do not transfer"; "short-clip rows
measure latency; only the 66 s rows support a throughput claim" (matches the
bench's latency-vs-throughput design); and "CPU threads were pinned (CT2 4.8.0
default `intra_threads=0` can oversubscribe, issue #2063)." This closes the three
original HIGHs (single-run -> median-of-5; "4s CUDA init" mechanism claim dropped
in favor of measured `load`/"CUDA context + weights to VRAM + first cuDNN/cuBLAS
setup"; throughput now backed by the 66 s clip, not the 11 s clip) and the
relevant MEDIUMs (baseline named every time; threads pinned/reported; VAD off so
RTF is per file second; K2200->940MX extrapolation disclaimed).

### Residual overclaim scan — none found

No remaining unsupported statement in the README. "frozen by design," the pins
table, "DO NOT TOUCH" driver guidance, and the credit/PR-#1766 attribution are
descriptive and consistent with the prior verified review. The benchmark section
no longer contains the un-rigorous "2.3x / 4s CUDA init" claims that failed #3.

### FINAL verdict

**RE-REVIEW #3 FINAL: PASS.** Every published number is exactly backed by the
measured median-of-5 data with no transcription error, every arithmetic value
(end-to-end column, RTF, x_rt, headline ratios) re-derives correctly, and every
claim is literally supported and conservatively stated (the load-bearing "larger
model" qualifier is correct). The README is safe to publish as-is.
