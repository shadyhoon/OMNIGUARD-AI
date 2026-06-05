"""
OmniGuard AI - OpenCV forensic vision layer
============================================
Pixel-level forensic analysis for images and video frames. This is
the "real" vision path that supplements the lightweight byte-level
heuristics in :mod:`utils.verifier` when the heavy stack is
installed.

Heavy imports (cv2 / numpy) are lazy: the module is import-safe
even on machines that don't have OpenCV.

Public functions
----------------
* :func:`analyse_image_bytes` - OpenCV-based forensic pass on raw
  image bytes (PNG, JPEG, etc.).
* :func:`analyse_video_path` - sample frames from a local video
  file and run the same checks per frame, plus inter-frame
  consistency.
* :func:`extract_frames` - helper to write evenly-spaced frames
  from a video to a temp directory.
"""

from __future__ import annotations

import io
import logging
import math
import os
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Lazy-loaded numpy / cv2 so the module imports cheaply.
_np = None
_cv2 = None


def _get_cv():
    """Lazy-import OpenCV and numpy. Returns (np, cv2) or raises."""
    global _np, _cv2
    if _cv2 is None:
        import numpy as np  # noqa: F401
        import cv2

        _np = np
        _cv2 = cv2
    return _np, _cv2


# =====================================================================
# Image forensics
# =====================================================================
def _decode_image(data: bytes):
    """Decode a byte buffer to an OpenCV BGR image, or None on failure."""
    _, cv2 = _get_cv()
    arr = _np.frombuffer(data, dtype=_np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _lighting_consistency(img) -> Dict[str, float]:
    """
    Compare the lighting colour cast of the foreground (largest
    central region) to the background (outer ring). Big mismatches
    can indicate composited faces or AI-spliced imagery.
    """
    _, cv2 = _get_cv()
    h, w = img.shape[:2]
    # Centre 50% vs outer 25% border
    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    cy0, cy1 = int(h * 0.25), int(h * 0.75)
    centre = img[cy0:cy1, cx0:cx1]
    border = _np.concatenate(
        [
            img[: int(h * 0.25), :].reshape(-1, 3),
            img[int(h * 0.75):, :].reshape(-1, 3),
            img[:, : int(w * 0.25)].reshape(-1, 3),
            img[:, int(w * 0.75):].reshape(-1, 3),
        ],
        axis=0,
    )
    if centre.size == 0 or border.size == 0:
        return {"mismatch_pct": 0.0, "centre_lab": [0, 0, 0], "border_lab": [0, 0, 0]}
    centre_lab = cv2.cvtColor(
        centre.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).mean(axis=0)
    border_lab = cv2.cvtColor(
        border.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).mean(axis=0)
    # Use only the chromaticity channels (a, b). Luminance differences
    # are normal; hue differences are the suspicious bit.
    diff = float(_np.linalg.norm(centre_lab[1:] - border_lab[1:]))
    mismatch_pct = min(1.0, diff / 30.0)
    return {
        "mismatch_pct": round(mismatch_pct, 3),
        "centre_lab": [round(float(x), 1) for x in centre_lab],
        "border_lab": [round(float(x), 1) for x in border_lab],
    }


def _jpeg_artifact_score(img) -> float:
    """
    Estimate JPEG blockiness in the 8x8 DCT domain. A high score
    may indicate double-compression or splicing.
    """
    _, cv2 = _get_cv()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Sum of squared differences at 8-pixel boundaries, normalised
    diffs = []
    for y in range(0, h - 8, 8):
        diffs.append(float(_np.mean(_np.abs(gray[y, :].astype(int) - gray[y + 1, :].astype(int)))))
    for x in range(0, w - 8, 8):
        diffs.append(float(_np.mean(_np.abs(gray[:, x].astype(int) - gray[:, x + 1].astype(int)))))
    if not diffs:
        return 0.0
    score = sum(diffs) / len(diffs) / 30.0  # rough normalisation
    return float(min(1.0, max(0.0, score)))


def _edge_density(img) -> float:
    """Quick proxy: density of Canny edges. Smooth / over-smoothed = suspicious."""
    _, cv2 = _get_cv()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(edges.mean() / 255.0)


def analyse_image_bytes(data: bytes) -> Dict[str, Any]:
    """
    Run the OpenCV forensic pipeline on raw image bytes.

    Returns a dict with the same multimodal shape used elsewhere
    in the project: ``lighting_anomaly_score``, ``artifact_score``,
    plus raw measurements.
    """
    img = _decode_image(data)
    if img is None:
        return {
            "engine": "opencv",
            "ok": False,
            "error": "decode_failed",
            "lighting_anomaly_score": 0.0,
            "artifact_score": 0.0,
        }
    lighting = _lighting_consistency(img)
    artifact = _jpeg_artifact_score(img)
    edges = _edge_density(img)
    return {
        "engine": "opencv",
        "ok": True,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "lighting_mismatch": lighting["mismatch_pct"],
        "lighting_anomaly_score": lighting["mismatch_pct"],
        "artifact_score": artifact,
        "edge_density": round(edges, 4),
        "centre_lab": lighting["centre_lab"],
        "border_lab": lighting["border_lab"],
    }


# =====================================================================
# Video frame extraction
# =====================================================================
def extract_frames(
    video_path: str, max_frames: int = 8
) -> List["_np.ndarray"]:
    """
    Extract up to ``max_frames`` evenly-spaced BGR frames from a
    local video file. Returns an empty list on any failure.
    """
    _, cv2 = _get_cv()
    if not os.path.exists(video_path):
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    if max_frames >= total:
        indices = list(range(total))
    else:
        step = total / float(max_frames)
        indices = [int(i * step) for i in range(max_frames)]
    frames: List = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def analyse_video_path(video_path: str, max_frames: int = 6) -> Dict[str, Any]:
    """
    Sample frames and run the image forensic pass on each, then
    compute a single per-video score from the frame-level results.
    """
    frames = extract_frames(video_path, max_frames=max_frames)
    if not frames:
        return {
            "engine": "opencv",
            "ok": False,
            "error": "no_frames",
            "lighting_anomaly_score": 0.0,
            "artifact_score": 0.0,
            "frame_count": 0,
        }
    per_frame = [analyse_image_bytes(_frame_to_bytes(f)) for f in frames]
    lighting = sum(p.get("lighting_anomaly_score", 0.0) for p in per_frame) / len(per_frame)
    artifact = sum(p.get("artifact_score", 0.0) for p in per_frame) / len(per_frame)
    edges = sum(p.get("edge_density", 0.0) for p in per_frame) / len(per_frame)
    # Inter-frame variance on luminance: rapid changes = possible splicing.
    lumas = [p["centre_lab"][0] for p in per_frame if p.get("centre_lab")]
    inter_var = 0.0
    if len(lumas) > 1:
        mean = sum(lumas) / len(lumas)
        inter_var = sum((l - mean) ** 2 for l in lumas) / len(lumas)
    inter_anomaly = min(1.0, inter_var / 50.0)
    return {
        "engine": "opencv",
        "ok": True,
        "frame_count": len(per_frame),
        "lighting_anomaly_score": round(0.6 * lighting + 0.4 * inter_anomaly, 3),
        "artifact_score": round(artifact, 3),
        "edge_density": round(edges, 4),
        "inter_frame_luma_variance": round(inter_var, 3),
        "per_frame": per_frame,
    }


def _frame_to_bytes(frame) -> bytes:
    """Encode a BGR frame to PNG bytes (lossless, no re-encoding artefacts)."""
    _, cv2 = _get_cv()
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        return b""
    return _np.asarray(buf).tobytes()


# =====================================================================
# Health probe
# =====================================================================
def vision_health() -> Dict[str, Any]:
    """Cheap check: can we import OpenCV + numpy?"""
    try:
        _get_cv()
        return {"opencv_ok": True, "error": None}
    except Exception as exc:
        return {"opencv_ok": False, "error": str(exc)[:120]}
