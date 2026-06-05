"""
OmniGuard AI - Content Verifier
===============================
Forensic-style analysis pipeline that simulates multimodal
verification of text, image, and video/audio content.

The module is intentionally lightweight so it can run on an
8 GB-RAM machine: all "AI" heuristics are implemented with
deterministic, well-commented algorithms that operate on the
raw bytes/text the caller provides. No heavy deep-learning
dependencies are required.

Public entry point:
    analyze_content(content_type, user_input) -> dict

Supported content_type values:
    "text"   - free-form text / claim
    "image"  - path to a local image file OR raw bytes
    "video"  - path to a local video file (metadata only)
    "audio"  - path to a local audio file (metadata only)
    "url"    - a publicly reachable URL
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import random
import re
import struct
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from bs4 import BeautifulSoup


# =====================================================================
# Tunable configuration
# =====================================================================
HTTP_TIMEOUT_SECONDS = 10
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 OmniGuard-AI/0.1"
)

# Lightweight English stop-word list used for text feature extraction.
# It is intentionally short - we only need it for the heuristic signals
# in this module, not for full NLP.
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "while", "of", "at",
    "by", "for", "with", "about", "to", "in", "on", "as", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "from", "into", "over", "under", "than",
    "then", "so", "such", "i", "you", "he", "she", "we", "they",
    "them", "my", "your", "our", "their", "not", "no", "do", "does",
    "did", "done", "have", "has", "had", "can", "could", "should",
    "would", "will", "shall", "may", "might", "must", "also", "very",
    "just", "more", "most", "some", "any", "all", "each", "every",
}

# Phrases that are commonly used in LLM / generative-AI outputs. They
# are not conclusive on their own, but multiple hits raise the
# "hallucination / synthetic" flag in the report.
_LLM_TELL_TALES = (
    "as an ai", "as a language model", "i cannot", "i don't have access",
    "in conclusion", "it is important to note", "furthermore",
    "moreover", "in summary", "delve into", "navigate the complexities",
    "tapestry of", "in the realm of", "a testament to",
    "it is worth noting", "first and foremost", "last but not least",
    "undeniable", "pivotal moment", "game-changer", "leverage",
    "synergy", "paradigm shift", "disrupt", "revolutionize",
    "unprecedented", "groundbreaking", "cutting-edge",
)

# Claims that are commonly used as canonical "live facts" for the
# cross-reference simulation. The real system would query search APIs;
# here we use a deterministic lookup so the dashboard always returns
# a useful, reproducible signal.
_LIVE_FACTS: List[Dict[str, str]] = [
    {
        "claim": "the earth is round",
        "status": "confirmed",
        "source": "NASA - https://science.nasa.gov/earth/",
        "summary": "Earth is an oblate spheroid, confirmed by satellite imagery.",
    },
    {
        "claim": "water boils at 100 degrees celsius at sea level",
        "status": "confirmed",
        "source": "NIST Chemistry WebBook",
        "summary": "Boiling point of water at 1 atm is 99.97 °C.",
    },
    {
        "claim": "humans have landed on mars",
        "status": "false",
        "source": "NASA Mars Exploration Program",
        "summary": "No crewed mission has reached Mars; only robotic landers.",
    },
    {
        "claim": "the great wall of china is visible from space with the naked eye",
        "status": "false",
        "source": "NASA astronaut observations",
        "summary": "Astronauts report the wall is not easily visible unaided.",
    },
    {
        "claim": "vaccines cause autism",
        "status": "false",
        "source": "WHO / CDC consensus",
        "summary": "The original Wakefield study was retracted; no causal link.",
    },
    {
        "claim": "the sun is a star",
        "status": "confirmed",
        "source": "European Space Agency",
        "summary": "The Sun is a G-type main-sequence star.",
    },
    {
        "claim": "lightning never strikes the same place twice",
        "status": "false",
        "source": "NOAA / National Geographic",
        "summary": "Lightning frequently strikes the same place, e.g. the Empire State Building.",
    },
    {
        "claim": "sharks are mammals",
        "status": "false",
        "source": "NOAA Fisheries",
        "summary": "Sharks are cartilaginous fish, not mammals.",
    },
]


# =====================================================================
# Low-level helpers
# =====================================================================
def _now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 when the denominator is zero."""
    return numerator / denominator if denominator else 0.0


def _shannon_entropy(data: bytes) -> float:
    """
    Compute Shannon entropy (bits per byte) of a byte sequence.

    Natural photographic / audio content tends to fall in the 4-8
    bits/byte range. Values approaching 8 suggest near-uniform random
    noise, while values near 0 indicate highly repetitive (potentially
    synthetic) content. Either extreme is mildly suspicious.
    """
    if not data:
        return 0.0
    counts = [0] * 256
    for byte in data:
        counts[byte] += 1
    length = len(data)
    entropy = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / length
        entropy -= p * math.log2(p)
    return entropy


def _seeded_random(*keys: Any) -> random.Random:
    """
    Build a deterministic RNG seeded from the input.

    The same input always produces the same forensic "noise" so that
    the UI shows stable results for a given piece of content.
    """
    hasher = hashlib.sha256()
    for k in keys:
        hasher.update(str(k).encode("utf-8", errors="replace"))
    return random.Random(int(hasher.hexdigest(), 16))


# =====================================================================
# Text analysis
# =====================================================================
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _split_words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _analyse_text(text: str) -> Dict[str, Any]:
    """
    Run heuristic text analysis and return a feature dict.

    The metrics produced are deliberately explainable:
      * sentence / word counts
      * average sentence length
      * type-token ratio (lexical diversity)
      * punctuation density
      * number density
      * tell-tale LLM phrase matches
    """
    sentences = _split_sentences(text)
    words = _split_words(text)

    if not words:
        return {
            "sentence_count": 0,
            "word_count": 0,
            "avg_sentence_length": 0.0,
            "type_token_ratio": 0.0,
            "punctuation_density": 0.0,
            "number_density": 0.0,
            "unique_word_count": 0,
            "llm_phrase_hits": [],
            "stopword_ratio": 0.0,
        }

    unique = set(words)
    punctuation = sum(1 for ch in text if ch in ".,;:!?\"'()[]{}")
    numbers = sum(1 for w in words if any(c.isdigit() for c in w))
    stopwords = sum(1 for w in words if w in _STOP_WORDS)

    text_lower = text.lower()
    llm_hits = sorted({phrase for phrase in _LLM_TELL_TALES if phrase in text_lower})

    return {
        "sentence_count": len(sentences),
        "word_count": len(words),
        "avg_sentence_length": _safe_div(len(words), len(sentences)),
        "type_token_ratio": _safe_div(len(unique), len(words)),
        "punctuation_density": _safe_div(punctuation, len(words)),
        "number_density": _safe_div(numbers, len(words)),
        "unique_word_count": len(unique),
        "llm_phrase_hits": llm_hits,
        "stopword_ratio": _safe_div(stopwords, len(words)),
    }


def _hallucination_signals(text: str, features: Dict[str, Any]) -> List[str]:
    """
    Build a human-readable list of reasons the text *might* be
    synthetic or contain unsupported claims.

    Each entry is one observation - the UI can render them as a
    bulleted list.
    """
    signals: List[str] = []

    if features["llm_phrase_hits"]:
        joined = ", ".join(f'"{p}"' for p in features["llm_phrase_hits"])
        signals.append(
            f"Detected common generative-AI phrasing: {joined}."
        )

    if features["avg_sentence_length"] > 28:
        signals.append(
            "Average sentence length is unusually long "
            f"({features['avg_sentence_length']:.1f} words), a pattern "
            "often seen in LLM-generated prose."
        )

    if features["type_token_ratio"] < 0.35 and features["word_count"] > 60:
        signals.append(
            f"Low lexical diversity (TTR={features['type_token_ratio']:.2f}); "
            "vocabulary repeats more than a typical human author."
        )

    if features["punctuation_density"] > 0.3:
        signals.append(
            "Punctuation density is high; check whether quotes / lists "
            "are being used to mask thin substance."
        )

    if features["number_density"] > 0.15:
        signals.append(
            "Many numeric tokens appear without attributable sources; "
            "numbers without citations are a common hallucination pattern."
        )

    if features["stopword_ratio"] < 0.25 and features["word_count"] > 30:
        signals.append(
            "Stop-word ratio is below expected natural prose baseline; "
            "consider verifying the writing style."
        )

    # Look for assertive "guarantee" / "absolute" language without citation.
    assertive_re = re.compile(
        r"\b(guarantee[sd]?|always|never|everyone|nobody|100%|proven|definitive[ly]?)\b",
        re.IGNORECASE,
    )
    assertive_hits = sorted(set(assertive_re.findall(text)))
    if assertive_hits:
        signals.append(
            "Assertive absolute language used: "
            + ", ".join(assertive_hits)
            + ". Verify with primary sources."
        )

    if not signals:
        signals.append(
            "No obvious generative-AI tells detected. Always cross-check "
            "specific claims against authoritative sources."
        )

    return signals


# =====================================================================
# Image / video / audio forensic helpers
# =====================================================================
def _read_file_head(path: str, max_bytes: int = 1_048_576) -> bytes:
    """
    Read up to ``max_bytes`` from a local file. We only ever need the
    first ~1 MB to compute entropy / magic bytes, which keeps memory
    pressure low on 8 GB machines.
    """
    with open(path, "rb") as f:
        return f.read(max_bytes)


def _fetch_url_head(url: str, max_bytes: int = 1_048_576) -> Tuple[bytes, str]:
    """
    Fetch the first ``max_bytes`` of a URL with a custom User-Agent.
    Returns (bytes, final_url) - the final URL lets us surface the
    canonical page even after redirects.
    """
    response = requests.get(
        url,
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": HTTP_USER_AGENT},
        stream=True,
        allow_redirects=True,
    )
    response.raise_for_status()
    buf = io.BytesIO()
    for chunk in response.iter_content(chunk_size=16_384):
        if chunk:
            buf.write(chunk)
        if buf.tell() >= max_bytes:
            break
    return buf.getvalue(), response.url


def _image_forensics(data: bytes) -> Dict[str, Any]:
    """
    Lightweight forensic checks on raw image bytes.

    Notes
    -----
    * We never decode pixels (PIL is intentionally avoided to keep
      memory low). Instead we use:
        - magic-byte detection (PNG, JPEG, GIF, WebP, BMP)
        - byte-level Shannon entropy per region
        - a "block boundary" anomaly score that flags unusual
          alignment patterns sometimes seen in AI-generated images.
    """
    magic = data[:16]
    magic_hex = magic.hex()

    if magic.startswith(b"\x89PNG\r\n\x1a\n"):
        fmt = "PNG"
    elif magic.startswith(b"\xff\xd8\xff"):
        fmt = "JPEG"
    elif magic.startswith(b"GIF87a") or magic.startswith(b"GIF89a"):
        fmt = "GIF"
    elif magic.startswith(b"RIFF") and magic[8:12] == b"WEBP":
        fmt = "WEBP"
    elif magic.startswith(b"BM"):
        fmt = "BMP"
    else:
        fmt = "UNKNOWN"

    if len(data) < 64:
        return {
            "format": fmt,
            "size_bytes": len(data),
            "overall_entropy": 0.0,
            "block_entropy_variance": 0.0,
            "lighting_anomaly_score": 0.0,
            "artifact_score": 0.0,
            "block_alignment_anomaly": 0.0,
            "notes": "File too small for meaningful analysis.",
        }

    block_size = 4096
    blocks = [data[i : i + block_size] for i in range(0, len(data), block_size)]
    entropies = [_shannon_entropy(b) for b in blocks if b]
    overall = _shannon_entropy(data)
    if len(entropies) > 1:
        mean = sum(entropies) / len(entropies)
        variance = sum((e - mean) ** 2 for e in entropies) / len(entropies)
    else:
        variance = 0.0

    # The "block alignment anomaly" looks for 16-byte aligned runs of
    # zero padding, which can appear in synthetic or containerised
    # content. It's a weak signal on its own, hence the low weight.
    zero_runs = 0
    run = 0
    for byte in data[:65536]:
        if byte == 0:
            run += 1
        else:
            if run >= 16 and run % 16 == 0:
                zero_runs += 1
            run = 0

    block_alignment_anomaly = min(1.0, zero_runs / 32.0)

    # Simulated perceptual signals. These are not real photometric
    # measurements; they're derived from byte-level statistics and a
    # deterministic per-input noise, which is documented in the UI.
    rng = _seeded_random(fmt, len(data), overall)
    lighting_anomaly = round(min(1.0, max(0.0, 0.15 + rng.random() * 0.35)), 3)
    artifact_score = round(
        min(1.0, max(0.0, 0.10 + variance / 8.0 + block_alignment_anomaly * 0.2)),
        3,
    )

    notes: List[str] = []
    if fmt == "UNKNOWN":
        notes.append("Unrecognised image format - bytes do not match known headers.")
    if overall < 3.0:
        notes.append("Byte entropy is unusually low; image may be largely uniform.")
    if overall > 7.85:
        notes.append("Byte entropy is near-maximum; possible encrypted or noisy data.")
    if variance > 4.0:
        notes.append(
            "High variance between file blocks - could indicate composited regions."
        )
    if not notes:
        notes.append("Byte-level statistics within normal photographic range.")

    return {
        "format": fmt,
        "size_bytes": len(data),
        "overall_entropy": round(overall, 3),
        "block_entropy_variance": round(variance, 3),
        "lighting_anomaly_score": lighting_anomaly,
        "artifact_score": artifact_score,
        "block_alignment_anomaly": round(block_alignment_anomaly, 3),
        "notes": "; ".join(notes),
        "magic_hex": magic_hex,
    }


def _video_forensics(data: bytes, path: str) -> Dict[str, Any]:
    """
    Best-effort metadata-level forensic pass for video files.

    We deliberately do NOT decode frames (avoids ffmpeg/opencv).
    Instead we read the container header and use entropy signals.
    """
    head = data[:4096]
    container = "UNKNOWN"
    if b"ftyp" in head[:64]:
        container = "MP4 / MOV (ISO Base Media)"
    elif head.startswith(b"RIFF") and head[8:12] == b"AVI ":
        container = "AVI"
    elif head.startswith(b"\x1aE\xdf\xa3"):
        container = "Matroska (MKV/WebM)"
    elif head.startswith(b"FLV\x01"):
        container = "Flash Video"

    entropy = _shannon_entropy(data)
    rng = _seeded_random(container, len(data), entropy)

    return {
        "container": container,
        "size_bytes": len(data),
        "overall_entropy": round(entropy, 3),
        "frame_consistency_score": round(min(1.0, 0.7 + rng.random() * 0.25), 3),
        "lighting_anomaly_score": round(min(1.0, 0.1 + rng.random() * 0.4), 3),
        "artifact_score": round(min(1.0, 0.1 + rng.random() * 0.5), 3),
        "audio_visual_alignment": round(min(1.0, 0.6 + rng.random() * 0.3), 3),
        "notes": (
            "Frame-level decoding disabled to stay within 8 GB RAM budget. "
            "Analysis limited to container header and byte-level statistics."
        ),
        "source_path": os.path.basename(path) if path else "",
    }


def _audio_forensics(data: bytes, path: str) -> Dict[str, Any]:
    """
    Lightweight audio forensic pass. We read the RIFF/WAVE header
    if present and use byte-level entropy as a coarse proxy.
    """
    fmt = "UNKNOWN"
    sample_rate = 0
    channels = 0
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        fmt = "WAV"
        # fmt chunk starts with "fmt " at byte 12; we look for the
        # 'fmt ' chunk and read its first fields.
        try:
            chunk_id = data[12:16]
            if chunk_id == b"fmt ":
                channels = struct.unpack("<H", data[22:24])[0]
                sample_rate = struct.unpack("<I", data[24:28])[0]
        except struct.error:
            pass
    elif data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
        fmt = "MP3"
    elif data[:4] == b"OggS":
        fmt = "OGG"
    elif data[:4] == b"fLaC":
        fmt = "FLAC"

    entropy = _shannon_entropy(data)
    rng = _seeded_random(fmt, len(data), entropy)

    return {
        "format": fmt,
        "size_bytes": len(data),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "overall_entropy": round(entropy, 3),
        "voice_consistency_score": round(min(1.0, 0.65 + rng.random() * 0.3), 3),
        "spectral_anomaly_score": round(min(1.0, 0.1 + rng.random() * 0.4), 3),
        "artifact_score": round(min(1.0, 0.1 + rng.random() * 0.45), 3),
        "notes": (
            "Spectrogram decoding disabled to stay within 8 GB RAM budget. "
            "Signals are derived from header + byte-level entropy only."
        ),
        "source_path": os.path.basename(path) if path else "",
    }


# =====================================================================
# Web / URL analysis
# =====================================================================
def _extract_main_text(soup: BeautifulSoup) -> str:
    """
    Extract the *main* content of an HTML document, dropping nav,
    sidebars, comments, scripts, and styles. We try, in order:

      1. ``<article>`` blocks joined together (most blogs/news)
      2. ``<main>`` block
      3. The largest text density in any single ``<div>`` / ``<section>``
      4. The whole document as a last resort

    Returns a lowercased string suitable for keyword matching.
    """
    # Drop noise first.
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    # 1. <article>
    articles = soup.find_all("article")
    if articles:
        text = " ".join(a.get_text(separator=" ", strip=True) for a in articles)
        if len(text) > 200:
            return text.lower()

    # 2. <main>
    main = soup.find("main")
    if main:
        text = main.get_text(separator=" ", strip=True)
        if len(text) > 200:
            return text.lower()

    # 3. Largest <p> density block. We score by paragraph char count.
    candidates = soup.find_all(["div", "section"])
    best_text = ""
    best_score = 0
    for c in candidates:
        ps = c.find_all("p")
        score = sum(len(p.get_text(" ", strip=True)) for p in ps)
        if score > best_score:
            best_score = score
            best_text = " ".join(p.get_text(" ", strip=True) for p in ps)
    if best_score > 200:
        return best_text.lower()

    # 4. Fallback: whole document. The caller will down-weight hits
    #    that come from this branch by checking total length.
    return soup.get_text(separator=" ", strip=True).lower()


def _word_boundary_match(claim: str, text: str) -> Optional[int]:
    """
    Find ``claim`` in ``text`` using word boundaries so that e.g.
    "mars" does not match inside "remarkable" or "Marsala".

    Returns the index of the first match, or ``None``.
    """
    if not claim or not text:
        return None
    pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(claim) + r"(?![A-Za-z0-9])")
    m = pattern.search(text)
    return m.start() if m else None


def _relevance_score(claim: str, text: str, hit_index: int) -> float:
    """
    Compute a 0.0-1.0 relevance score for a claim inside a body of text.

    Heuristics, in order of weight:
      * position in the document (earlier = more likely the topic)
      * text length vs claim length (a longer body dilutes relevance)
      * count of occurrences in the body

    The result is meant for the UI to show alongside each cross-ref
    hit, not as a hard classifier.
    """
    if not text or not claim:
        return 0.0
    text_len = max(1, len(text))
    # Position: 0.0 at start, -0.5 at end.
    position_factor = max(0.0, 1.0 - (hit_index / text_len))
    # Body length: very long pages dilute relevance.
    length_factor = min(1.0, 1000.0 / text_len)
    # Occurrence count.
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(claim) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    count = len(pattern.findall(text))
    count_factor = min(1.0, count / 3.0)
    return round(
        0.4 * position_factor + 0.3 * length_factor + 0.3 * count_factor, 3
    )


def _parse_url_for_facts(url: str) -> List[Dict[str, str]]:
    """
    Fetch a URL, extract its main content, and look for canonical
    claims in :data:`_LIVE_FACTS` using *word-boundary* matches so
    that substring noise (e.g. "mars" inside "remarkable") is ignored.

    Each returned hit carries a ``relevance`` field (0.0-1.0) so the
    UI can rank and down-weight noisy matches.
    """
    try:
        body, final_url = _fetch_url_head(url)
    except requests.RequestException as exc:
        return [
            {
                "claim": "(fetch failed)",
                "status": "error",
                "source": url,
                "summary": f"Could not retrieve page: {exc}",
                "context": "",
                "relevance": 0.0,
            }
        ]

    try:
        soup = BeautifulSoup(body, "html.parser")
        text = _extract_main_text(soup)
    except Exception as exc:
        return [
            {
                "claim": "(parse failed)",
                "status": "error",
                "source": final_url,
                "summary": f"HTML parse error: {exc}",
                "context": "",
                "relevance": 0.0,
            }
        ]

    matches: List[Dict[str, str]] = []
    for fact in _LIVE_FACTS:
        idx = _word_boundary_match(fact["claim"], text)
        if idx is None:
            continue
        start = max(0, idx - 60)
        end = min(len(text), idx + len(fact["claim"]) + 60)
        matches.append(
            {
                "claim": fact["claim"],
                "status": fact["status"],
                "source": fact["source"],
                "summary": fact["summary"],
                "context": ("..." + text[start:end] + "...").strip(),
                "relevance": _relevance_score(fact["claim"], text, idx),
            }
        )

    # Sort highest-relevance first so the UI shows the most relevant
    # hits at the top.
    matches.sort(key=lambda m: m.get("relevance", 0.0), reverse=True)
    return matches


# =====================================================================
# Cross-reference engine
# =====================================================================
def _cross_reference(content_type: str, user_input: Union[str, bytes, Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build a list of simulated cross-reference hits.

    The real system would query search/news APIs; for the dashboard
    demo we mix deterministic local knowledge with optional live URL
    parsing when the input is a URL.
    """
    results: List[Dict[str, str]] = []

    # 1. Local knowledge base
    if content_type in ("text", "url"):
        text_to_scan = ""
        if isinstance(user_input, str):
            text_to_scan = user_input.lower()
        if isinstance(user_input, dict):
            text_to_scan = (user_input.get("text") or "").lower()

        for fact in _LIVE_FACTS:
            idx = _word_boundary_match(fact["claim"], text_to_scan)
            if idx is None:
                continue
            results.append(
                {
                    "claim": fact["claim"],
                    "status": fact["status"],
                    "source": fact["source"],
                    "summary": fact["summary"],
                    "context": "Matched inside submitted content.",
                    "relevance": _relevance_score(
                        fact["claim"], text_to_scan, idx
                    ),
                }
            )

    # 2. Live URL parsing
    if content_type == "url" and isinstance(user_input, str):
        parsed = urllib.parse.urlparse(user_input)
        if parsed.scheme in ("http", "https"):
            results.extend(_parse_url_for_facts(user_input))

    # 3. No fabricated fallbacks. If no canonical claim was matched
    #    in the input, we return an empty list and let the UI tell
    #    the user "no cross-reference hits". Filling the panel with
    #    unrelated facts (e.g. showing the sun, water, vaccines when
    #    the input is about the moon) was misleading and is removed.
    return results


# =====================================================================
# Scoring
# =====================================================================
def _veracity_score(
    content_type: str,
    text_features: Dict[str, Any],
    multimodal: Dict[str, Any],
    cross_refs: List[Dict[str, str]],
    hallucination_signals: List[str],
) -> int:
    """
    Combine the signals into a single 0-100 veracity score.

    The score is *higher* = more trustworthy. The heuristic is
    intentionally simple and explainable - we are not claiming a
    full probabilistic inference.
    """
    score = 70  # neutral baseline

    if content_type == "text":
        ttr = text_features.get("type_token_ratio", 0.5)
        avg_len = text_features.get("avg_sentence_length", 15)
        if ttr > 0.55:
            score += 5
        elif ttr < 0.35:
            score -= 5
        if 12 <= avg_len <= 22:
            score += 5
        elif avg_len > 28:
            score -= 5

        score -= min(20, len(text_features.get("llm_phrase_hits", [])) * 4)
        score -= min(15, max(0, len(hallucination_signals) - 1) * 3)

    if content_type in ("image", "video", "audio"):
        artifact = multimodal.get("artifact_score", 0.2)
        lighting = multimodal.get("lighting_anomaly_score", 0.2)
        score -= int(artifact * 25)
        score -= int(lighting * 15)

    if content_type == "video":
        alignment = multimodal.get("audio_visual_alignment", 0.7)
        score += int((alignment - 0.5) * 20)

    if content_type == "audio":
        voice = multimodal.get("voice_consistency_score", 0.7)
        score += int((voice - 0.5) * 20)

    # Cross-reference adjustments
    confirmed = sum(1 for r in cross_refs if r["status"] == "confirmed")
    falses = sum(1 for r in cross_refs if r["status"] == "false")
    errors = sum(1 for r in cross_refs if r["status"] == "error")
    score += min(15, confirmed * 5)
    score -= min(30, falses * 10)
    score -= min(10, errors * 3)

    return max(0, min(100, score))


def _multimodal_consistency(content_type: str, multimodal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a multimodal consistency block.

    For text-only inputs we still return a structured object so the
    UI can render the same widget shape across all content types.
    """
    if content_type == "text":
        return {
            "applicable": False,
            "audio_visual_alignment_pct": None,
            "lighting_anomaly_pct": None,
            "artifact_score_pct": None,
            "summary": "Multimodal checks are not applicable for plain text.",
        }

    if content_type == "image":
        lighting = multimodal.get("lighting_anomaly_score", 0.0)
        artifact = multimodal.get("artifact_score", 0.0)
        return {
            "applicable": True,
            "audio_visual_alignment_pct": None,
            "lighting_anomaly_pct": round((1.0 - lighting) * 100, 2),
            "artifact_score_pct": round((1.0 - artifact) * 100, 2),
            "summary": (
                f"Image format: {multimodal.get('format', 'UNKNOWN')}; "
                f"entropy={multimodal.get('overall_entropy', 0):.2f}."
            ),
        }

    if content_type == "video":
        alignment = multimodal.get("audio_visual_alignment", 0.0)
        lighting = multimodal.get("lighting_anomaly_score", 0.0)
        artifact = multimodal.get("artifact_score", 0.0)
        return {
            "applicable": True,
            "audio_visual_alignment_pct": round(alignment * 100, 2),
            "lighting_anomaly_pct": round((1.0 - lighting) * 100, 2),
            "artifact_score_pct": round((1.0 - artifact) * 100, 2),
            "summary": (
                f"Container: {multimodal.get('container', 'UNKNOWN')}; "
                f"frame-consistency={multimodal.get('frame_consistency_score', 0):.2f}."
            ),
        }

    if content_type == "audio":
        voice = multimodal.get("voice_consistency_score", 0.0)
        spectral = multimodal.get("spectral_anomaly_score", 0.0)
        artifact = multimodal.get("artifact_score", 0.0)
        return {
            "applicable": True,
            "audio_visual_alignment_pct": round(voice * 100, 2),
            "lighting_anomaly_pct": None,
            "artifact_score_pct": round((1.0 - artifact) * 100, 2),
            "summary": (
                f"Audio format: {multimodal.get('format', 'UNKNOWN')} at "
                f"{multimodal.get('sample_rate_hz', 0)} Hz; spectral-anomaly={spectral:.2f}."
            ),
        }

    return {
        "applicable": False,
        "audio_visual_alignment_pct": None,
        "lighting_anomaly_pct": None,
        "artifact_score_pct": None,
        "summary": "Unknown content type.",
    }


# =====================================================================
# Public entry point
# =====================================================================
def analyze_content(
    content_type: str,
    user_input: Union[str, bytes, os.PathLike, Dict[str, Any], None],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run the full verification pipeline and return a structured report.

    Parameters
    ----------
    content_type:
        One of "text", "image", "video", "audio", "url".
    user_input:
        * str  - the text itself, a URL, or a local file path
        * bytes / PathLike - raw bytes or a local file path
        * dict  - e.g. {"text": "...", "image": "/path/to/img.png"}

    Returns
    -------
    A dict with the following top-level keys:

        * ``veracity_score``              : int (0-100)
        * ``multimodal_consistency``      : dict (see ``_multimodal_consistency``)
        * ``cross_reference_results``     : list[dict]
        * ``hallucination_report``        : list[str]
        * ``content_type``                : str
        * ``generated_at``                : ISO timestamp
        * ``diagnostics``                 : raw measurements used
    """
    content_type = (content_type or "").strip().lower()
    started_at = time.time()

    # ------------------------------------------------------------------
    # Normalise the input into the shape each branch expects.
    # ------------------------------------------------------------------
    text_payload: str = ""
    file_path: Optional[str] = None
    raw_bytes: Optional[bytes] = None
    url_value: Optional[str] = None

    if content_type == "url":
        if not isinstance(user_input, str):
            raise TypeError("For content_type='url', user_input must be a string URL.")
        url_value = user_input
        text_payload = user_input
    elif content_type == "text":
        if isinstance(user_input, str):
            text_payload = user_input
        elif isinstance(user_input, dict):
            text_payload = str(user_input.get("text", ""))
        else:
            raise TypeError("For content_type='text', user_input must be str or dict.")
    else:
        # image / video / audio - accept str (file path or URL), bytes,
        # or dict with a "path"/"bytes"/"url" key.
        if isinstance(user_input, (bytes, bytearray)):
            raw_bytes = bytes(user_input)
        elif isinstance(user_input, str):
            parsed = urllib.parse.urlparse(user_input)
            if parsed.scheme in ("http", "https"):
                url_value = user_input
                try:
                    raw_bytes, _ = _fetch_url_head(user_input)
                except requests.RequestException as exc:
                    raw_bytes = b""
                    text_payload = f"(remote fetch failed: {exc})"
            else:
                file_path = user_input
        elif isinstance(user_input, dict):
            if "url" in user_input and user_input["url"]:
                url_value = user_input["url"]
                try:
                    raw_bytes, _ = _fetch_url_head(url_value)
                except requests.RequestException as exc:
                    raw_bytes = b""
                    text_payload = f"(remote fetch failed: {exc})"
            if "path" in user_input and user_input["path"]:
                file_path = user_input["path"]
            if "bytes" in user_input and user_input["bytes"]:
                raw_bytes = bytes(user_input["bytes"])
        else:
            raise TypeError(
                f"Unsupported user_input type for content_type='{content_type}'."
            )

        if raw_bytes is None and file_path is not None:
            try:
                raw_bytes = _read_file_head(file_path)
            except OSError as exc:
                raw_bytes = b""
                text_payload = f"(file read failed: {exc})"

    # ------------------------------------------------------------------
    # Per-modality analysis
    # ------------------------------------------------------------------
    text_features = _analyse_text(text_payload) if text_payload else {
        "sentence_count": 0,
        "word_count": 0,
        "avg_sentence_length": 0.0,
        "type_token_ratio": 0.0,
        "punctuation_density": 0.0,
        "number_density": 0.0,
        "unique_word_count": 0,
        "llm_phrase_hits": [],
        "stopword_ratio": 0.0,
    }

    if content_type == "image":
        multimodal = _image_forensics(raw_bytes or b"")
    elif content_type == "video":
        multimodal = _video_forensics(raw_bytes or b"", file_path or url_value or "")
    elif content_type == "audio":
        multimodal = _audio_forensics(raw_bytes or b"", file_path or url_value or "")
    else:
        multimodal = {}

    # ------------------------------------------------------------------
    # Cross-reference
    # ------------------------------------------------------------------
    # Lightweight path: substring / word-boundary match against the
    # built-in fact table. Always runs.
    if content_type == "url" and url_value:
        cross_refs = _cross_reference("url", url_value)
    elif content_type in ("text",):
        cross_refs = _cross_reference("text", text_payload)
    else:
        cross_refs = _cross_reference(content_type, user_input if user_input is not None else "")

    # Optional RAG path: Chroma + DuckDuckGo + sentence-transformers.
    # Heavier, slower, but covers claims outside the built-in fact
    # table. Disabled by default; enabled via ``use_rag=True``.
    use_rag = bool(kwargs.get("use_rag", False))
    rag_engine = str(kwargs.get("rag_engine", "chroma+ddg"))
    if use_rag and content_type in ("text", "url") and (text_payload or url_value):
        try:
            from utils.rag import rag_cross_reference  # lazy import

            rag_input = text_payload or url_value or ""
            rag_hits = rag_cross_reference(rag_input)
            # Merge: RAG hits first (they're richer), then lightweight hits
            # we don't already cover, deduped by claim text.
            seen_claims = {(h.get("claim") or "").strip().lower() for h in rag_hits}
            for h in cross_refs:
                key = (h.get("claim") or "").strip().lower()
                if key and key not in seen_claims:
                    rag_hits.append(h)
                    seen_claims.add(key)
            cross_refs = rag_hits
        except Exception:
            # RAG is optional; never let it break the lightweight path.
            pass

    # ------------------------------------------------------------------
    # Hallucination / synthesis report
    # ------------------------------------------------------------------
    if content_type == "text":
        hallucination_report = _hallucination_signals(text_payload, text_features)
    elif content_type in ("image", "video", "audio"):
        notes = multimodal.get("notes", "")
        hallucination_report = []
        if notes:
            hallucination_report.append(notes)
        if multimodal.get("artifact_score", 0) > 0.6:
            hallucination_report.append(
                "Elevated artifact score suggests possible generative or composited content."
            )
        if multimodal.get("lighting_anomaly_score", 0) > 0.55:
            hallucination_report.append(
                "Lighting anomalies detected; inconsistent shadows / highlights can be a "
                "deepfake tell."
            )
        if not hallucination_report:
            hallucination_report.append(
                "No elevated synthetic-content signals detected from header / entropy checks."
            )
    else:  # url
        hallucination_report = _hallucination_signals(text_payload or (url_value or ""), text_features)

    # ------------------------------------------------------------------
    # Final scoring
    # ------------------------------------------------------------------
    score = _veracity_score(
        content_type, text_features, multimodal, cross_refs, hallucination_report
    )
    mm_block = _multimodal_consistency(content_type, multimodal)

    elapsed_ms = int((time.time() - started_at) * 1000)

    return {
        "veracity_score": score,
        "multimodal_consistency": mm_block,
        "cross_reference_results": cross_refs,
        "hallucination_report": hallucination_report,
        "content_type": content_type,
        "generated_at": _now_iso(),
        "analysis_duration_ms": elapsed_ms,
        "diagnostics": {
            "text_features": text_features,
            "multimodal_raw": multimodal,
            "input_summary": {
                "is_url": bool(url_value),
                "file_path": os.path.basename(file_path) if file_path else None,
                "byte_length": len(raw_bytes) if raw_bytes else 0,
                "url": url_value,
            },
        },
    }


# =====================================================================
# Quick smoke test
# =====================================================================
if __name__ == "__main__":
    sample = (
        "Breaking news! The sun is a star and humans have landed on mars. "
        "Furthermore, this groundbreaking discovery is a testament to science."
    )
    report = analyze_content("text", sample)
    import json

    print(json.dumps(report, indent=2, default=str))
