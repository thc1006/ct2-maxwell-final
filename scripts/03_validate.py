#!/usr/bin/env python3
"""
03_validate.py — prove the sm_50 build works on the Maxwell GPU, then benchmark
GPU vs CPU across compute types.

Run inside the project venv AFTER 02_build_ct2.sh and after faster-whisper is
installed with our patched ctranslate2 kept (see scripts/02c note / README):

    source ~/projects/ct2-maxwell-final/venv/bin/activate
    source ~/projects/ct2-maxwell-final/cuda-env.sh
    python scripts/03_validate.py

A real cudaErrorNoKernelImageForDevice on the GPU path means the loaded
ctranslate2 does NOT contain sm_50 kernels (e.g. a PyPI build clobbered ours).
"""
from __future__ import annotations
import os
import sys
import time
import json
import urllib.request
from pathlib import Path

MODEL = os.environ.get("CT2_VAL_MODEL", "tiny")  # fits 4GB K2200 and 2GB 940MX
PROJ = Path(os.environ.get("CT2_PROJ", Path(__file__).resolve().parent.parent))
SAMPLE = PROJ / "sample.wav"
SAMPLE_URLS = [
    "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav",
    "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
]


def info() -> None:
    import ctranslate2
    print("== environment ==")
    print("  python           :", sys.version.split()[0])
    print("  ctranslate2 ver  :", ctranslate2.__version__)
    print("  ctranslate2 path :", ctranslate2.__file__)
    print("  cuda device count:", ctranslate2.get_cuda_device_count())
    if ctranslate2.get_cuda_device_count() < 1:
        sys.exit("FATAL: no CUDA device visible to ctranslate2")


def fetch_sample() -> Path:
    if SAMPLE.exists() and SAMPLE.stat().st_size > 1000:
        return SAMPLE
    for url in SAMPLE_URLS:
        try:
            print(f"  downloading sample: {url}")
            urllib.request.urlretrieve(url, SAMPLE)
            if SAMPLE.stat().st_size > 1000:
                return SAMPLE
        except Exception as e:  # noqa: BLE001
            print(f"    failed: {e}")
    sys.exit("FATAL: could not fetch a sample audio (no network?). "
             "Place a short speech clip at " + str(SAMPLE))


def run_case(device: str, compute_type: str, audio: str) -> dict:
    from faster_whisper import WhisperModel
    rec = {"device": device, "compute_type": compute_type}
    try:
        t0 = time.perf_counter()
        model = WhisperModel(MODEL, device=device, compute_type=compute_type)
        t_load = time.perf_counter() - t0
        t1 = time.perf_counter()
        segments, dinfo = model.transcribe(audio, beam_size=1)
        text = " ".join(s.text for s in segments).strip()
        t_xcribe = time.perf_counter() - t1
        rec.update(ok=bool(text), load_s=round(t_load, 2),
                   transcribe_s=round(t_xcribe, 2),
                   audio_s=round(getattr(dinfo, "duration", 0.0), 2),
                   rtf=round(t_xcribe / max(getattr(dinfo, "duration", 1.0), 1e-6), 3),
                   text=text[:80])
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        low = msg.lower()
        rec.update(ok=False, error=msg[:160])
        if "no kernel image" in low or "nokernelimage" in low:
            rec.update(kind="no_sm50", diagnosis="sm_50 NOT in loaded ctranslate2 (wrong/clobbered build)")
        elif any(k in low for k in ("connection", "timed out", "huggingface", "couldn't find",
                                    "couldn t find", "resolve", "max retries", "network", "offline")):
            rec.update(kind="network", diagnosis="MODEL DOWNLOAD / NETWORK failure -- NOT a GPU/sm_50 result")
        else:
            rec.update(kind="other")
    return rec


def main() -> int:
    info()
    audio = str(fetch_sample())
    cases = [
        ("cuda", "float32"),
        ("cuda", "int8_float32"),
        ("cuda", "int8"),
        ("cpu", "int8"),
        ("cpu", "float32"),
    ]
    results = [run_case(d, c, audio) for d, c in cases]

    print("\n== results ==")
    hdr = f'{"device":5} {"compute":13} {"ok":3} {"load_s":7} {"xcribe_s":9} {"rtf":6}  text/err'
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        tail = r.get("text") or r.get("diagnosis") or r.get("error", "")
        print(f'{r["device"]:5} {r["compute_type"]:13} '
              f'{"Y" if r.get("ok") else "n":3} '
              f'{r.get("load_s", "-"):>7} {r.get("transcribe_s", "-"):>9} '
              f'{r.get("rtf", "-"):>6}  {tail}')

    (PROJ / "validation_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {PROJ / 'validation_results.json'}")

    cuda = [r for r in results if r["device"] == "cuda"]
    gpu_ok = any(r.get("ok") for r in cuda)
    if gpu_ok:
        print("\nPASS: sm_50 GPU path works on this Maxwell GPU.")
        return 0
    # If EVERY cuda case failed for a network/model-download reason, the result
    # is INCONCLUSIVE (the sm_50 build was never actually exercised) -- do not
    # report this as a Maxwell/GPU failure.
    if cuda and all(r.get("kind") == "network" for r in cuda):
        print("\nINCONCLUSIVE: could not load the model (network/download) -- "
              "GPU path was never exercised. Pre-cache the model and re-run.")
        return 2
    if any(r.get("kind") == "no_sm50" for r in cuda):
        print("\nFAIL: GPU reports 'no kernel image' -- the loaded ctranslate2 "
              "lacks sm_50 kernels (wrong/clobbered build).")
        return 1
    print("\nFAIL: no CUDA case succeeded -- see diagnosis above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
