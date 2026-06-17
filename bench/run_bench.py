#!/usr/bin/env python3
"""
run_bench.py — rigorous, publication-grade GPU(sm_50 float32) vs CPU benchmark
for the frozen Maxwell CTranslate2 build, via faster-whisper.

Design (so the published numbers are defensible):
- LATENCY vs THROUGHPUT are measured separately: a short clip (~11s) shows
  per-call latency where fixed overhead dominates; a long clip (short clip
  concatenated to ~66s) shows steady-state throughput. Only the long-clip
  numbers support any "GPU wins on long/batch audio" statement.
- One-time model load is reported separately as load_s (CUDA context + weights
  to VRAM + first cuDNN/cuBLAS setup) and is NOT charged per clip. We do not
  claim a specific mechanism for it.
- Each timed figure is the MEDIAN of REPEATS runs after one untimed warmup;
  we also report min/max so the spread is visible.
- CPU threads are PINNED and reported (CTranslate2 4.8.0's default
  intra_threads=0 can oversubscribe, issue #2063), so the CPU baseline is
  reproducible.
- VAD is OFF, so info.duration == file length and RTF is per real second.
- On sm_50 there is no native FP16 and no dp4a int8, so cuda int8/float16 fall
  back to float32; we benchmark only the configs the hardware actually runs.

Run inside the project venv with cuda-env.sh sourced:
    source venv/bin/activate && source cuda-env.sh
    python bench/run_bench.py
Env overrides: CT2_BENCH_MODELS=tiny,small  CT2_BENCH_REPEATS=5
               CT2_CPU_THREADS=4  CT2_LONG_REPEAT=6
"""
from __future__ import annotations
import os
import sys
import time
import json
import wave
import statistics
from pathlib import Path

CPU_THREADS = int(os.environ.get("CT2_CPU_THREADS", os.cpu_count() or 4))
# Pin CPU math threads BEFORE importing faster_whisper / ctranslate2.
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))

PROJ = Path(os.environ.get("CT2_PROJ", Path.home() / "projects" / "ct2-maxwell-final"))
SHORT = PROJ / "sample.wav"
LONG = PROJ / "sample_long.wav"
LONG_REPEAT = max(2, int(os.environ.get("CT2_LONG_REPEAT", "6")))
MODELS = [m for m in os.environ.get("CT2_BENCH_MODELS", "tiny,small").split(",") if m]
REPEATS = max(3, int(os.environ.get("CT2_BENCH_REPEATS", "5")))
CONFIGS = [("cuda", "float32"), ("cpu", "int8"), ("cpu", "float32")]


def build_long_audio() -> None:
    """Concatenate SHORT LONG_REPEAT times into LONG (stdlib wave, no ffmpeg)."""
    if LONG.exists() and LONG.stat().st_size > SHORT.stat().st_size:
        return
    with wave.open(str(SHORT), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    with wave.open(str(LONG), "wb") as out:
        out.setparams(params)
        for _ in range(LONG_REPEAT):
            out.writeframes(frames)


def duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def bench_model_config(model_size: str, device: str, compute_type: str, audios: list) -> list:
    from faster_whisper import WhisperModel
    base = {"model": model_size, "device": device, "compute_type": compute_type}
    try:
        t0 = time.perf_counter()
        kw = {"cpu_threads": CPU_THREADS} if device == "cpu" else {}
        model = WhisperModel(model_size, device=device, compute_type=compute_type, **kw)
        load_s = round(time.perf_counter() - t0, 2)
    except Exception as e:  # noqa: BLE001
        return [{**base, "ok": False, "stage": "load", "error": str(e)[:180]}]

    results = []
    for label, path, dur in audios:
        rec = {**base, "audio": label, "audio_s": round(dur, 2), "load_s": load_s}
        try:
            def once():
                segs, info = model.transcribe(str(path), beam_size=1, vad_filter=False)
                t = time.perf_counter()
                text = " ".join(s.text for s in segs).strip()  # consume = real work
                return time.perf_counter() - t, text, float(getattr(info, "duration", 0.0))
            _, warm_text, _ = once()  # untimed warmup
            if not warm_text:
                rec.update(ok=False, error="empty transcription (possible OOM / truncation)")
                results.append(rec)
                continue
            times = sorted(once()[0] for _ in range(REPEATS))
            med = statistics.median(times)
            rec.update(ok=True,
                       transcribe_med_s=round(med, 3),
                       transcribe_min_s=round(times[0], 3),
                       transcribe_max_s=round(times[-1], 3),
                       rtf=round(med / max(dur, 1e-6), 4),
                       x_realtime=round(dur / max(med, 1e-6), 1))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            kind = "oom" if ("out of memory" in msg.lower() or "cudaerrormemoryallocation" in msg.lower()) else "other"
            rec.update(ok=False, error=msg[:180], kind=kind)
        results.append(rec)
    return results


def main() -> int:
    import ctranslate2
    if not SHORT.exists() or SHORT.stat().st_size < 1000:
        sys.exit(f"short sample not found at {SHORT} -- stage it first")
    build_long_audio()
    audios = [("short", SHORT, duration_s(SHORT)), ("long", LONG, duration_s(LONG))]

    print("ctranslate2", ctranslate2.__version__,
          "| cuda devices", ctranslate2.get_cuda_device_count(),
          "| cuda compute types", ctranslate2.get_supported_compute_types("cuda"))
    print(f"cpu_threads(pinned)={CPU_THREADS}  models={MODELS}  repeats={REPEATS}  "
          f"vad=off beam=1")
    print(f"audio: short={audios[0][2]:.1f}s long={audios[1][2]:.1f}s\n")

    results = []
    for m in MODELS:
        for dev, ct in CONFIGS:
            recs = bench_model_config(m, dev, ct, audios)
            results.extend(recs)
            for r in recs:
                if r.get("ok"):
                    tag = (f'load {r["load_s"]}s | med {r["transcribe_med_s"]}s '
                           f'[{r["transcribe_min_s"]}-{r["transcribe_max_s"]}] '
                           f'| RTF {r["rtf"]} | {r["x_realtime"]}x rt')
                else:
                    tag = f'FAIL({r.get("stage","run")}): {r.get("error", "")}'
                print(f'{m:6} {dev:4} {ct:8} {r.get("audio","-"):5} | {tag}')

    (PROJ / "benchmark_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {PROJ / 'benchmark_results.json'}")

    # Honest summary: report GPU-vs-CPU on the LONG clip (throughput) explicitly.
    print("\n== throughput on long clip: transcribe median (s), lower=faster ==")
    for m in MODELS:
        row = {f'{r["device"]}/{r["compute_type"]}': r.get("transcribe_med_s")
               for r in results if r["model"] == m and r.get("audio") == "long" and r.get("ok")}
        gpu = row.get("cuda/float32")
        cpu_i8 = row.get("cpu/int8")
        v = f' -> GPU {round(cpu_i8 / gpu, 2)}x vs CPU-int8' if (gpu and cpu_i8) else ""
        print(f'  {m}: {row}{v}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
