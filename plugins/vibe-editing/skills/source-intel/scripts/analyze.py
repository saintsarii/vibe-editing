#!/usr/bin/env python3
"""
Source Intelligence — analyze raw footage before editing.

Produces source_intel.json with face positions, scene changes, audio quality,
and recommendations for every downstream pipeline step.

Usage:
    python3 analyze.py --source 00_SOURCE/video.mp4 --out 10_WORK/source_intel.json
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, timeout=120):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout, r.stderr, r.returncode


def probe_video(source):
    """Get basic video metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", source,
    ]
    out, _, _ = run(cmd)
    data = json.loads(out)

    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    duration = float(data["format"].get("duration", 0))
    coded_w = int(video_stream["width"]) if video_stream else 0
    coded_h = int(video_stream["height"]) if video_stream else 0
    rotation = _stream_rotation_cw(video_stream) if video_stream else 0
    # Phones store portrait video as landscape pixels + a rotation flag; report what viewers see.
    disp_w, disp_h = (coded_h, coded_w) if rotation in (90, 270) else (coded_w, coded_h)
    result = {
        "source": source,
        "duration_s": round(duration, 1),
        "resolution": f"{disp_w}x{disp_h}" if video_stream else "unknown",
        "width": disp_w,
        "height": disp_h,
        "coded_width": coded_w,
        "coded_height": coded_h,
        "rotation": rotation,
        "fps": eval_fps(video_stream.get("r_frame_rate", "30/1")) if video_stream else 30,
        "codec": video_stream.get("codec_name", "unknown") if video_stream else "unknown",
        "audio_channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
        "audio_codec": audio_stream.get("codec_name", "unknown") if audio_stream else "none",
    }
    return result


def _stream_rotation_cw(stream):
    """Clockwise degrees (0/90/180/270) needed to display the stream upright.

    ffprobe reports a display-matrix rotation (counter-clockwise, e.g. -90 for iPhone
    portrait) and/or a legacy `rotate` tag (clockwise, e.g. 90)."""
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(round(-float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        try:
            return int(round(float(tag))) % 360
        except (TypeError, ValueError):
            pass
    return 0


def eval_fps(fps_str):
    try:
        if "/" in fps_str:
            n, d = fps_str.split("/")
            return round(int(n) / int(d), 2)
        return round(float(fps_str), 2)
    except Exception:
        return 30.0


YUNET_NAMES = ("face_detection_yunet_2023mar.onnx", "yunet.onnx")


def _kit_root():
    """The vibe-editing plugin root (the folder holding .claude-plugin/)."""
    r = os.environ.get("VIBE_PIPELINE_ROOT") or os.environ.get("CLAUDE_PLUGIN_ROOT")
    if r and os.path.isdir(os.path.join(r, ".claude-plugin")):
        return r
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, ".claude-plugin")):
            return d
        d = os.path.dirname(d)
    return None


def _find_yunet_model(cv2):
    """Locate a YuNet face model: env override, next to this script, the kit's bundled
    horizontal-to-vertical/scripts/yunet.onnx, then cv2's data dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.environ.get("VIBE_YUNET_MODEL")]
    candidates += [os.path.join(here, n) for n in YUNET_NAMES]
    root = _kit_root()
    if root:
        candidates.append(os.path.join(root, "skills", "horizontal-to-vertical", "scripts", "yunet.onnx"))
    data_dir = os.path.join(os.path.dirname(cv2.__file__), "data")
    candidates += [os.path.join(data_dir, n) for n in YUNET_NAMES]
    return next((c for c in candidates if c and os.path.isfile(c)), None)


def _open_upright(cv2, source, rotation):
    """Open `source` and return (cap, rotate_fn) so rotate_fn(frame) is display-upright.

    Auto-orientation is switched off where supported so we rotate exactly once, from the
    ffprobe rotation; if the backend ignores that switch, rotate only when the decoded
    frame is still in coded (sideways) orientation."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return cap, None
    rot_code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotation)
    if rot_code is None:
        return cap, lambda f: f
    prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    manual = prop is not None and cap.set(prop, 0) and not cap.get(prop)
    coded_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    coded_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def rotate(frame):
        if manual:
            return cv2.rotate(frame, rot_code)
        h, w = frame.shape[:2]
        if rotation in (90, 270) and (w, h) == (max(coded_w, coded_h), min(coded_w, coded_h)) \
                and coded_w > coded_h:
            return cv2.rotate(frame, rot_code)  # backend left it sideways
        return frame  # backend already applied orientation (180° can't be told apart)
    return cap, rotate


def _sample_faces(cv2, source, rotation, sample_interval, detect):
    """Seek through `source` every `sample_interval` s, run detect(frame) -> [(x,y,w,h)]
    on upright frames, and cluster. Frame size comes from the decoded (upright) frame."""
    cap, rotate = _open_upright(cv2, source, rotation)
    if rotate is None:
        return {"error": f"cannot open {source}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = max(1, int(sample_interval * fps))
    all_faces = []
    sample_count = 0
    frame_w = frame_h = 0

    for frame_idx in range(0, total_frames, step_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = rotate(frame)
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h

        sample_count += 1
        for box in detect(frame):
            x, y, w, h = (float(v) for v in box[:4])
            all_faces.append({
                "time_s": round(frame_idx / fps, 1),
                "cx": round((x + w / 2) / frame_w, 3),
                "cy": round((y + h / 2) / frame_h, 3),
                "size_pct": round((w * h) / frame_area, 4),
                "w": int(w), "h": int(h),
            })

    cap.release()
    result = _cluster_faces(all_faces, frame_w, frame_h, sample_count)
    result["frame_size"] = f"{frame_w}x{frame_h}"
    return result


def detect_faces(source, sample_interval=10, duration=0, rotation=0):
    """Sample frames (upright, honoring rotation metadata) and detect faces with YuNet."""
    try:
        import cv2
    except ImportError:
        return {"error": "opencv not installed — run: pip3 install opencv-python"}

    model_path = _find_yunet_model(cv2)
    if model_path is None or not hasattr(cv2, "FaceDetectorYN"):
        return _detect_faces_haar(source, sample_interval, duration, rotation)

    detector = None

    def detect(frame):
        nonlocal detector
        h, w = frame.shape[:2]
        if detector is None:
            detector = cv2.FaceDetectorYN.create(model_path, "", (w, h), 0.7)
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        return [] if faces is None else [tuple(f[:4]) for f in faces]

    result = _sample_faces(cv2, source, rotation, sample_interval, detect)
    if "error" not in result:
        result["detector"] = "yunet"
    return result


def _detect_faces_haar(source, sample_interval, duration, rotation=0):
    """Fallback face detection using Haar cascades — only if this OpenCV build has them
    (opencv-python 5.x dropped CascadeClassifier). Never raises: returns an error dict."""
    try:
        import cv2
    except ImportError:
        return {"error": "opencv not installed"}

    no_faces = {"count": 0, "detections": 0, "samples": 0, "detector": "none",
                "error": "face detection unavailable — no YuNet model found and this OpenCV "
                         "build has no Haar cascades; reframe will use defaults"}
    cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not hasattr(cv2, "CascadeClassifier") or not cascade_dir:
        return no_faces
    try:
        cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_frontalface_default.xml"))
        if cascade.empty():
            return no_faces

        def detect(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        result = _sample_faces(cv2, source, rotation, sample_interval, detect)
    except Exception as e:  # noqa: BLE001 — a broken fallback must not kill the analysis
        return {**no_faces, "error": f"Haar fallback failed: {e}"}
    if "error" not in result:
        result["detector"] = "haar"
    return result


def _cluster_faces(all_faces, frame_w, frame_h, sample_count):
    """Cluster detected faces by spatial position into distinct persons."""
    if not all_faces:
        return {
            "count": 0,
            "detections": 0,
            "samples": sample_count,
            "note": "no faces detected — b-roll, text-on-screen, or very wide shot",
        }

    clusters = []
    for f in all_faces:
        placed = False
        for c in clusters:
            if abs(f["cx"] - c["sum_cx"] / c["n"]) < 0.20:
                c["faces"].append(f)
                c["sum_cx"] += f["cx"]
                c["sum_cy"] += f["cy"]
                c["sum_size"] += f["size_pct"]
                c["n"] += 1
                placed = True
                break
        if not placed:
            clusters.append({
                "faces": [f],
                "sum_cx": f["cx"],
                "sum_cy": f["cy"],
                "sum_size": f["size_pct"],
                "n": 1,
            })

    clusters.sort(key=lambda c: c["n"], reverse=True)

    def classify_size(s):
        if s > 0.05:
            return "tight"
        elif s > 0.02:
            return "medium"
        elif s > 0.005:
            return "wide"
        return "extreme_wide"

    def summarize(c, label):
        faces = c["faces"]
        avg_cx = c["sum_cx"] / c["n"]
        avg_cy = c["sum_cy"] / c["n"]
        avg_size = c["sum_size"] / c["n"]
        xs = [f["cx"] for f in faces]
        ys = [f["cy"] for f in faces]
        return {
            "label": label,
            "avg_x_pct": round(avg_cx, 3),
            "avg_y_pct": round(avg_cy, 3),
            "avg_size_pct": round(avg_size, 4),
            "movement_range_x_pct": round(max(xs) - min(xs), 3),
            "movement_range_y_pct": round(max(ys) - min(ys), 3),
            "classification": classify_size(avg_size),
            "detections": c["n"],
        }

    result = {
        "count": min(len(clusters), 5),
        "detections": len(all_faces),
        "samples": sample_count,
    }

    labels = ["primary", "secondary", "tertiary", "face_4", "face_5"]
    for i, c in enumerate(clusters[:5]):
        if c["n"] >= max(2, sample_count * 0.05):
            result[labels[i]] = summarize(c, labels[i])

    return result


def detect_scenes(source, threshold=0.3):
    """Detect scene changes using ffmpeg's scene filter."""
    cmd = [
        "ffmpeg", "-i", source, "-vf",
        f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr", "-f", "null", "-",
    ]
    _, stderr, _ = run(cmd, timeout=300)

    times = [0.0]
    for line in stderr.split("\n"):
        if "pts_time:" in line:
            try:
                pt = line.split("pts_time:")[1].split()[0]
                times.append(float(pt))
            except (IndexError, ValueError):
                pass

    is_multicam = False
    if len(times) > 5:
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval < 30 and len(times) > 10:
                is_multicam = True

    return {
        "count": len(times),
        "changes_at_s": [round(t, 1) for t in times[:50]],
        "is_multicam": is_multicam,
        "avg_duration_s": round(sum(
            times[i + 1] - times[i] for i in range(len(times) - 1)
        ) / max(len(times) - 1, 1), 1) if len(times) > 1 else 0,
    }


def analyze_audio(source):
    """Analyze audio quality: levels, noise floor, hum detection."""
    cmd = [
        "ffmpeg", "-i", source, "-af",
        "volumedetect", "-vn", "-f", "null", "-",
    ]
    _, stderr, _ = run(cmd, timeout=120)

    mean_vol = -25.0
    max_vol = -3.0
    for line in stderr.split("\n"):
        if "mean_volume:" in line:
            try:
                mean_vol = float(line.split("mean_volume:")[1].split("dB")[0].strip())
            except (ValueError, IndexError):
                pass
        if "max_volume:" in line:
            try:
                max_vol = float(line.split("max_volume:")[1].split("dB")[0].strip())
            except (ValueError, IndexError):
                pass

    cmd2 = [
        "ffmpeg", "-i", source, "-af",
        "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
        "-vn", "-f", "null", "-",
    ]
    out2, _, _ = run(cmd2, timeout=120)

    rms_values = []
    for line in out2.split("\n"):
        if "RMS_level" in line:
            try:
                val = float(line.split("=")[1].strip())
                if val > -80:
                    rms_values.append(val)
            except (ValueError, IndexError):
                pass

    noise_floor = -50.0
    speech_ratio = 0.7
    if rms_values:
        sorted_rms = sorted(rms_values)
        bottom_10 = sorted_rms[: max(1, len(sorted_rms) // 10)]
        noise_floor = sum(bottom_10) / len(bottom_10)
        speech_threshold = mean_vol - 15
        speech_frames = sum(1 for v in rms_values if v > speech_threshold)
        speech_ratio = round(speech_frames / len(rms_values), 2)

    hum_detected = False
    cmd3 = [
        "ffmpeg", "-i", source, "-t", "10", "-af",
        "asplit=2[a][b],[a]bandpass=f=60:width_type=h:w=20,volumedetect[a60],[b]volumedetect",
        "-vn", "-f", "null", "-",
    ]
    _, stderr3, rc3 = run(cmd3, timeout=30)
    if rc3 == 0 and "mean_volume" in stderr3:
        volumes = []
        for line in stderr3.split("\n"):
            if "mean_volume:" in line:
                try:
                    volumes.append(float(line.split("mean_volume:")[1].split("dB")[0].strip()))
                except (ValueError, IndexError):
                    pass
        if len(volumes) >= 2:
            band_60 = volumes[0]
            overall = volumes[1]
            if band_60 > overall - 20:
                hum_detected = True

    has_clipping = max_vol > -0.5

    quality = "clean"
    if hum_detected and noise_floor > -35:
        quality = "poor"
    elif hum_detected:
        quality = "needs_filter"
    elif noise_floor > -35:
        quality = "needs_denoise"
    elif has_clipping:
        quality = "clipping"

    return {
        "mean_volume_db": round(mean_vol, 1),
        "max_volume_db": round(max_vol, 1),
        "noise_floor_db": round(noise_floor, 1),
        "has_hum_60hz": hum_detected,
        "has_clipping": has_clipping,
        "speech_ratio": speech_ratio,
        "silence_ratio": round(1 - speech_ratio, 2),
        "quality": quality,
    }


def make_recommendations(probe, faces, scenes, audio):
    """Generate actionable recommendations for downstream pipeline steps."""
    recs = {
        "reframe_preset": "talking-head",
        "reframe_zoom": 1.6,
        "caption_height": 0.50,
        "needs_notch_filter": audio.get("has_hum_60hz", False),
        "needs_denoise": audio.get("quality") == "needs_denoise",
        "mic_routing": "single",
        "notes": [],
    }

    if probe.get("audio_channels", 1) >= 2:
        recs["mic_routing"] = "mix_both"
        recs["notes"].append(f"{probe['audio_channels']} audio channels — mix both mics")

    primary = faces.get("primary", {})
    face_size = primary.get("avg_size_pct", 0)
    face_class = primary.get("classification", "unknown")

    if face_size > 0.05:
        recs["reframe_zoom"] = 1.2
        recs["reframe_preset"] = "talking-head"
        recs["caption_height"] = 0.50
        recs["notes"].append(f"Face is {face_size*100:.1f}% of frame (tight) — low zoom")
    elif face_size > 0.02:
        recs["reframe_zoom"] = 1.6
        recs["reframe_preset"] = "stage"
        recs["caption_height"] = 0.50
        recs["notes"].append(f"Face is {face_size*100:.1f}% of frame (medium) — default zoom")
    elif face_size > 0.005:
        recs["reframe_zoom"] = 2.2
        recs["reframe_preset"] = "stage"
        recs["caption_height"] = 0.45
        recs["notes"].append(f"Face is {face_size*100:.1f}% of frame (wide) — high zoom needed")
    elif face_size > 0:
        recs["reframe_zoom"] = 2.8
        recs["reframe_preset"] = "stage"
        recs["caption_height"] = 0.45
        recs["notes"].append(
            f"Face is {face_size*100:.2f}% of frame (extreme wide) — very high zoom, quality may degrade"
        )

    face_count = faces.get("count", 0)
    if face_count >= 2:
        recs["notes"].append(f"{face_count} speakers detected — use dual-color captions")
    elif faces.get("error"):
        recs["notes"].append(f"{faces['error']} — using default reframe zoom")
    elif face_count == 0:
        recs["notes"].append("No faces detected — b-roll or text content, skip reframe")

    if scenes.get("is_multicam"):
        recs["notes"].append(
            f"Multicam detected ({scenes['count']} scene changes) — use per-segment reframe"
        )

    if audio.get("has_hum_60hz"):
        recs["notes"].append("60Hz hum detected — apply notch filter before transcription")
    if audio.get("has_clipping"):
        recs["notes"].append("Audio clipping detected — check levels, may need limiter")
    if audio.get("quality") == "needs_denoise":
        recs["notes"].append(f"High noise floor ({audio['noise_floor_db']}dB) — may need denoise")

    movement = primary.get("movement_range_x_pct", 0)
    if movement > 0.25:
        recs["notes"].append(
            f"High face movement ({movement*100:.0f}% X range) — face tracking will work hard"
        )

    return recs


def main():
    parser = argparse.ArgumentParser(description="Analyze source footage")
    parser.add_argument("--source", required=True, help="Path to source video")
    parser.add_argument("--out", default="source_intel.json", help="Output JSON path")
    parser.add_argument("--sample-interval", type=int, default=10, help="Seconds between frame samples")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--faces-only", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"ERROR: source file not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {args.source}")

    print("  Probing video metadata...")
    probe = probe_video(args.source)
    rot = f" (rotated {probe['rotation']}° from {probe['coded_width']}x{probe['coded_height']})" \
        if probe.get("rotation") else ""
    print(f"  → {probe['resolution']}{rot}, {probe['duration_s']}s, {probe['fps']}fps")

    faces = {}
    scenes = {}
    audio = {}

    if not args.audio_only:
        print(f"  Detecting faces (sampling every {args.sample_interval}s)...")
        faces = detect_faces(args.source, args.sample_interval, probe["duration_s"],
                             probe.get("rotation", 0))
        print(f"  → {faces.get('count', 0)} face(s) from {faces.get('detections', 0)} detections")

        print("  Detecting scene changes...")
        scenes = detect_scenes(args.source)
        print(f"  → {scenes['count']} scenes, multicam={scenes['is_multicam']}")

    if not args.faces_only:
        print("  Analyzing audio quality...")
        audio = analyze_audio(args.source)
        print(f"  → quality={audio['quality']}, mean={audio['mean_volume_db']}dB, noise floor={audio['noise_floor_db']}dB")

    print("  Generating recommendations...")
    recs = make_recommendations(probe, faces, scenes, audio)
    for note in recs["notes"]:
        print(f"    • {note}")

    result = {
        "source": args.source,
        "duration_s": probe["duration_s"],
        "resolution": probe["resolution"],
        "width": probe["width"],
        "height": probe["height"],
        "rotation": probe.get("rotation", 0),
        "fps": probe["fps"],
        "codec": probe["codec"],
        "faces": faces,
        "scenes": scenes,
        "audio": audio,
        "recommendations": recs,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSource intel written: {args.out}")


if __name__ == "__main__":
    main()
