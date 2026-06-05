"""
Regression tests for the bugs found in the audit pass.

Each test name maps to a bug ID from the audit (see the commit
"Fix bug-a … bug-m").
"""

import re
from unittest import mock

from utils.verifier import (
    _extract_main_text,
    _hallucination_signals,
    _split_sentences,
    _veracity_score,
    analyze_content,
)
from utils.rag import _chroma_query


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise AssertionError(label)


def _fake_features(word_count: int) -> dict:
    """Build a features dict with all the keys _hallucination_signals reads."""
    return {
        "word_count": word_count,
        "sentence_count": max(word_count, 1),
        "avg_sentence_length": float(word_count),
        "type_token_ratio": 0.5,
        "punctuation_density": 0.0,
        "number_density": 0.0,
        "unique_word_count": word_count,
        "llm_phrase_hits": [],
        "stopword_ratio": 0.3,
    }


# ---------------------------------------------------------------------
# Bug A: _chroma_query must not crash on empty distances
# ---------------------------------------------------------------------
def test_chroma_query_handles_none_distances() -> None:
    print("test_chroma_query_handles_none_distances")
    fake_collection = mock.MagicMock()
    fake_collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": None,  # Chroma returns this for an empty store
    }
    with mock.patch("utils.rag._get_chroma", return_value=fake_collection), \
         mock.patch("utils.rag._get_sentence_model") as m:
        m.return_value.encode.return_value.tolist.return_value = [[0.0, 0.0]]
        out = _chroma_query("anything", threshold=0.0)
    _check(out is None, "None distances -> None (not a TypeError)")


# ---------------------------------------------------------------------
# Bug D: empty / whitespace / null-byte text is rejected
# ---------------------------------------------------------------------
def test_empty_text_input_raises() -> None:
    print("test_empty_text_input_raises")
    for bad in ("", "   ", "\n\n\n", "\t\t", " \r\n \t "):
        try:
            analyze_content("text", bad)
        except ValueError as exc:
            _check(True, f"'{bad!r}' -> ValueError")
            continue
        raise AssertionError(f"'{bad!r}' should have raised")


def test_null_byte_text_input_raises() -> None:
    print("test_null_byte_text_input_raises")
    for bad in ("\x00", "hello\x00world", "\x00\x00\x00"):
        try:
            analyze_content("text", bad)
        except ValueError:
            _check(True, f"{bad!r} -> ValueError")
            continue
        raise AssertionError(f"{bad!r} should have raised")


# ---------------------------------------------------------------------
# Bug C: URL branch of analyze_content must fetch the page, not
# assign the URL string to text_payload.
# ---------------------------------------------------------------------
def test_url_branch_rejects_non_http_scheme() -> None:
    print("test_url_branch_rejects_non_http_scheme")
    for bad in ("file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)"):
        try:
            analyze_content("url", bad)
        except ValueError:
            _check(True, f"{bad!r} -> ValueError")
            continue
        raise AssertionError(f"{bad!r} should have raised")


# ---------------------------------------------------------------------
# Bug F: sentence splitter must not split on decimal points
# ---------------------------------------------------------------------
def test_sentence_splitter_keeps_decimals_intact() -> None:
    print("test_sentence_splitter_keeps_decimals_intact")
    sents = _split_sentences("Pi is 3.14 and light is fast. The end.")
    # 2 sentences, not 3 - the 3.14 should not be a boundary.
    _check(len(sents) == 2, f"got {len(sents)} sentences, want 2: {sents}")
    _check("3.14" in sents[0], "decimal preserved in first sentence")


def test_sentence_splitter_keeps_abbreviations_intact() -> None:
    print("test_sentence_splitter_keeps_abbreviations_intact")
    sents = _split_sentences("We met in the U.S. and it was fun. Goodbye.")
    _check(len(sents) == 2, f"got {len(sents)} sentences, want 2: {sents}")


# ---------------------------------------------------------------------
# Bug I: hallucination detector must not fire on a single word
# ---------------------------------------------------------------------
def test_hallucination_skips_assertive_on_short_input() -> None:
    print("test_hallucination_skips_assertive_on_short_input")
    sigs = _hallucination_signals("Never.", _fake_features(1))
    joined = " ".join(sigs)
    _check(
        "Assertive" not in joined,
        f"assertive note should not appear on 1-word input: {sigs}",
    )


def test_hallucination_still_fires_assertive_on_real_input() -> None:
    print("test_hallucination_still_fires_assertive_on_real_input")
    txt = (
        "I can guarantee that this will never happen. Everyone knows it. "
        "The science is 100% proven and definitive."
    )
    sigs = _hallucination_signals(txt, _fake_features(len(txt.split())))
    joined = " ".join(sigs)
    _check(
        "Assertive" in joined,
        f"assertive note should appear on a 19-word input: {sigs}",
    )


# ---------------------------------------------------------------------
# Bug H: cross-reference hits with relevance < 0.45 must not affect score
# ---------------------------------------------------------------------
def test_low_relevance_hits_excluded_from_score() -> None:
    print("test_low_relevance_hits_excluded_from_score")
    base_features = {
        "word_count": 50, "sentence_count": 3, "avg_sentence_length": 16,
        "number_density": 0.0, "punctuation_density": 0.0,
        "uppercase_ratio": 0.0, "hedge_density": 0.0,
    }
    refs = [
        {"status": "false", "relevance": 0.10, "claim": "off-topic sidebar"},
        {"status": "false", "relevance": 0.05, "claim": "unrelated link"},
    ]
    score = _veracity_score("text", base_features, {}, refs, [])
    # A neutral text with two off-topic "false" hits should not be
    # pulled all the way down to 0.
    _check(score > 30, f"off-topic hits dragged score to {score}, want > 30")


# ---------------------------------------------------------------------
# Bug K: download_video_clip clamps the duration and size args
# ---------------------------------------------------------------------
def test_download_video_clip_clamps_oversize_args() -> None:
    print("test_download_video_clip_clamps_oversize_args")
    from utils import video as video_mod
    captured_opts: dict = {}
    fake_ydl_instance = mock.MagicMock()
    fake_ydl_instance.extract_info.return_value = {
        "id": "video",
        "ext": "mp4",
        "title": "x",
        "duration": 1,
        "requested_downloads": [{"filepath": "C:/fake/clip.mp4"}],
    }
    fake_ydl_instance.__enter__.return_value = fake_ydl_instance
    fake_ydl_instance.__exit__.return_value = False

    def _fake_youtube_dl(opts):
        captured_opts.update(opts)
        return fake_ydl_instance

    fake_ytdl_mod = mock.MagicMock()
    fake_ytdl_mod.YoutubeDL.side_effect = _fake_youtube_dl
    with mock.patch.object(video_mod, "_get_ytdl", return_value=fake_ytdl_mod), \
         mock.patch.object(video_mod, "is_ytdlp_available", return_value=True), \
         mock.patch.object(video_mod.os.path, "exists", return_value=True), \
         mock.patch.object(video_mod.os.path, "getsize", return_value=1024):
        out = video_mod.download_video_clip(
            "https://example.com/v",
            max_duration_sec="999999",  # string, not int
            max_filesize_mb="not-a-number",
        )
    _check(out.get("ok") is True, f"ok=True after clamping (got {out!r})")
    dur = int(captured_opts["external_downloader_args"]["ffmpeg_o"][1])
    _check(dur <= 600, f"duration clamped to <= 600 (got {dur})")
    _check(
        captured_opts["max_filesize"] <= 200 * 1024 * 1024,
        f"filesize capped at 200 MB (got {captured_opts['max_filesize']})",
    )


# ---------------------------------------------------------------------
# Bug L: /analyze/upload rejects oversize uploads
# ---------------------------------------------------------------------
def test_upload_rejects_oversize() -> None:
    print("test_upload_rejects_oversize")
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    big = b"\x00" * (51 * 1024 * 1024)
    resp = client.post(
        "/analyze/upload",
        files={"file": ("x.png", big, "image/png")},
        data={"content_type": "image"},
    )
    _check(resp.status_code == 413, f"got {resp.status_code}, want 413")


def test_upload_rejects_bad_suffix() -> None:
    print("test_upload_rejects_bad_suffix")
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app)
    resp = client.post(
        "/analyze/upload",
        files={"file": ("x.exe", b"hello", "application/octet-stream")},
        data={"content_type": "image"},
    )
    _check(resp.status_code == 400, f"got {resp.status_code}, want 400")


# ---------------------------------------------------------------------
# Bug M: chip-render fields are HTML-escaped
# ---------------------------------------------------------------------
def test_chip_render_escapes_html_injection() -> None:
    print("test_chip_render_escapes_html_injection")
    from app import _build_report_chip_html
    out = _build_report_chip_html(
        content_type="text",
        generated_at='<script>alert(1)</script>',
        analysis_duration_ms=12,
    )
    _check(
        "<script>" not in out,
        f"injection survived: {out!r}",
    )
    _check("&lt;script&gt;" in out, "injection was escaped")


def main() -> None:
    test_chroma_query_handles_none_distances()
    test_empty_text_input_raises()
    test_null_byte_text_input_raises()
    test_url_branch_rejects_non_http_scheme()
    test_sentence_splitter_keeps_decimals_intact()
    test_sentence_splitter_keeps_abbreviations_intact()
    test_hallucination_skips_assertive_on_short_input()
    test_hallucination_still_fires_assertive_on_real_input()
    test_low_relevance_hits_excluded_from_score()
    test_download_video_clip_clamps_oversize_args()
    test_upload_rejects_oversize()
    test_upload_rejects_bad_suffix()
    test_chip_render_escapes_html_injection()
    print("\nAll bug-fix regression tests passed.")


if __name__ == "__main__":
    main()
