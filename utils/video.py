"""
OmniGuard AI - Video downloader
================================
Thin wrapper around yt-dlp that downloads a short clip from a
public video URL (YouTube, Vimeo, Twitter, TikTok, etc.) into a
temporary directory. We cap the file size and length so the
download never blows past a sensible budget on an 8 GB machine.

Heavy imports (yt-dlp) are lazy: this module is safe to import
even if yt-dlp is not installed.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ytdl_module = None


def _get_ytdl():
    global _ytdl_module
    if _ytdl_module is None:
        import yt_dlp

        _ytdl_module = yt_dlp
    return _ytdl_module


DEFAULT_MAX_DURATION_SEC = 60
DEFAULT_MAX_FILESIZE_MB = 80
DEFAULT_VIDEO_FORMAT = (
    # Prefer a moderate-res mp4; fall back to whatever is best.
    "bv*[height<=720][ext=mp4][filesize<80M]+ba[ext=m4a]/"
    "bv*[height<=720]+ba/best"
)


def is_ytdlp_available() -> bool:
    try:
        _get_ytdl()
        return True
    except Exception as exc:
        logger.info("yt-dlp unavailable: %s", exc)
        return False


def download_video_clip(
    url: str,
    *,
    max_duration_sec: int = DEFAULT_MAX_DURATION_SEC,
    max_filesize_mb: int = DEFAULT_MAX_FILESIZE_MB,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download a short clip from ``url`` and return:

        * ``ok``           - bool
        * ``path``         - local file path on success
        * ``title``        - video title (from yt-dlp metadata)
        * ``duration``     - duration in seconds
        * ``filesize_mb``  - downloaded size
        * ``error``        - short error label on failure

    The download is bounded by both duration and filesize to keep
    the footprint small.
    """
    if not url:
        return {"ok": False, "error": "empty_url"}
    if not is_ytdlp_available():
        return {"ok": False, "error": "yt_dlp_not_installed"}

    yt_dlp = _get_ytdl()

    tmpdir = output_dir or tempfile.mkdtemp(prefix="omniguard_")
    outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")

    ydl_opts: Dict[str, Any] = {
        "format": DEFAULT_VIDEO_FORMAT,
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 15,
        "retries": 1,
        "merge_output_format": "mp4",
        "max_filesize": max_filesize_mb * 1024 * 1024,
        # External downloader is the bottleneck on Windows for some
        # sites; letting yt-dlp use its internal ffmpeg-free path
        # means we don't need a system ffmpeg.
        "external_downloader_args": {"ffmpeg_o": ["-t", str(max_duration_sec)]},
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        # Best-effort cleanup
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        return {
            "ok": False,
            "error": str(exc).split("\n")[0][:160] or type(exc).__name__,
        }

    if not info:
        return {"ok": False, "error": "no_info_returned"}

    # Resolve the downloaded file path.
    file_path = None
    if info.get("requested_downloads"):
        file_path = info["requested_downloads"][0].get("filepath")
    if not file_path:
        vid_id = info.get("id", "video")
        ext = info.get("ext", "mp4")
        candidate = os.path.join(tmpdir, f"{vid_id}.{ext}")
        if os.path.exists(candidate):
            file_path = candidate

    if not file_path or not os.path.exists(file_path):
        return {"ok": False, "error": "file_not_found_after_download"}

    filesize = os.path.getsize(file_path)
    return {
        "ok": True,
        "path": file_path,
        "title": info.get("title", ""),
        "duration": int(info.get("duration") or 0),
        "filesize_mb": round(filesize / (1024 * 1024), 2),
        "site": info.get("extractor", "?"),
        "uploader": info.get("uploader", ""),
    }


def cleanup_download(path_or_dir: str) -> None:
    """Remove a downloaded clip or its containing directory."""
    if not path_or_dir:
        return
    try:
        if os.path.isdir(path_or_dir):
            shutil.rmtree(path_or_dir, ignore_errors=True)
        elif os.path.isfile(path_or_dir):
            os.remove(path_or_dir)
            # Also try to clean the parent temp dir
            parent = os.path.dirname(path_or_dir)
            if parent and os.path.isdir(parent) and "omniguard_" in parent:
                shutil.rmtree(parent, ignore_errors=True)
    except Exception as exc:
        logger.debug("cleanup_download: %s", exc)
