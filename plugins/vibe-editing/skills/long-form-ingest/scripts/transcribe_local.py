#!/usr/bin/env python3
"""Local, key-free transcription via faster-whisper — the no-Groq-key fallback.

This is the canonical offline transcriber for the pipeline. It produces word-level
JSON in the same shape the Groq path emits, so mining/cutting/captions all consume it
unchanged:
    {"source","start","end","duration","language","text",
     "words":[{"word","start","end","prob"}, ...]}

Whole file:        transcribe_local.py audio.wav --out words.json
A [start,end] clip: transcribe_local.py source.mp4 --out clip.json --start 0 --end 47.2
                    (word timestamps are rebased to the clip — i.e. start at 0)

Model size defaults to 'small' (good speed/quality on CPU). Override with --model or
the VIBE_WHISPER_MODEL env var (e.g. 'large-v3' for max quality, slower on CPU).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib" / "_shared"))
from audio_decode import load_audio  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Local key-free transcription (faster-whisper).")
    ap.add_argument("input")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--model", default=os.environ.get("VIBE_WHISPER_MODEL", "small"),
                    help="faster-whisper model size (small|medium|large-v3|...). Env: VIBE_WHISPER_MODEL")
    ap.add_argument("--device", default=os.environ.get("VIBE_WHISPER_DEVICE", "cpu"))
    ap.add_argument("--compute-type", default=os.environ.get("VIBE_WHISPER_COMPUTE", "int8"))
    a = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper not installed. Run: pip install faster-whisper  "
                 "(or set GROQ_API_KEY to use the cloud path).")

    seg = a.start is not None and a.end is not None
    # ffmpeg-decode (just [start,end] if given — word times then start at 0) instead of
    # faster-whisper's PyAV decoder, which breaks across PyAV releases.
    audio = load_audio(a.input, start=a.start if seg else None, end=a.end if seg else None)

    print(f"[transcribe_local] model={a.model} device={a.device} → transcribing"
          f"{' segment' if seg else ''}…", file=sys.stderr)
    model = WhisperModel(a.model, device=a.device, compute_type=a.compute_type)
    segments, info = model.transcribe(
        audio, word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    words, texts = [], []
    for s in segments:
        texts.append(s.text)
        for w in (s.words or []):
            words.append({
                "word": w.word.strip(),
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
                "prob": round(float(w.probability or 0.0), 3),
            })

    out = {
        "source": a.input,
        "start": a.start if seg else 0.0,
        "end": a.end if seg else round(float(getattr(info, "duration", 0.0) or 0.0), 3),
        "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 3),
        "language": getattr(info, "language", None),
        "text": "".join(texts).strip(),
        "words": words,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"[transcribe_local] {len(words)} words → {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
