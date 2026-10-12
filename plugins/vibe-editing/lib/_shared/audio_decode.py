"""Decode any audio/video file to the float32 mono 16 kHz array faster-whisper expects.

faster-whisper's own decoder goes through PyAV, and PyAV releases regularly break it
(e.g. PyAV 19 removed av.open(metadata_errors=...) → TypeError). The kit already
depends on ffmpeg, so decode with ffmpeg and hand WhisperModel.transcribe() a numpy
array — it then never touches PyAV.
"""
from __future__ import annotations

import subprocess

import numpy as np

SAMPLE_RATE = 16000


def load_audio(path, sr: int = SAMPLE_RATE, start: float | None = None,
               end: float | None = None) -> np.ndarray:
    """Return mono float32 PCM in [-1, 1] at `sr` Hz, optionally just [start, end] seconds."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start}"]
    if end is not None:
        cmd += ["-to", f"{end}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode audio from {path}: "
                           f"{r.stderr.decode(errors='replace').strip()[-500:]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()
