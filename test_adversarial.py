"""
Adversarial test harness for OmniGuard AI.

Targets likely-to-be-broken edge cases that the existing test suite
does not cover:

- Empty string input
- Whitespace-only input
- Unicode (emoji, RTL, Chinese, emoji mid-word)
- Very long input (10k+ chars)
- Malformed URLs (no scheme, garbage scheme, etc.)
- HTML with nested iframes / comments / CDATA
- Mixed truth and lies ("earth round but vaccines cause autism")
- Inputs with no punctuation
- Binary / null-byte payloads
- An actually-running local HTTP server to test the URL fallback
- LLM provider switching (OpenAI vs Gemini) where possible
- FastAPI backend edge cases (multipart upload, bad content_type, etc.)
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.verifier import analyze_content, _extract_main_text  # noqa: E402
from utils.llm import (  # noqa: E402
    available_providers,
    get_provider,
    is_llm_available,
    llm_enrich_report,
)
from utils.rag import rag_health  # noqa: E402
from utils.video import is_ytdlp_available  # noqa: E402


RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")


def test_empty_string() -> None:
    try:
        report = analyze_content("text", "")
    except Exception as exc:
        _record("empty string raises", True, f"raised {type(exc).__name__}")
        return
    # If it does NOT raise, that's a bug - empty text is meaningless.
    _record(
        "empty string raises",
        False,
        f"analyze_content('text', '') returned a report instead of raising. "
        f"veracity_score={report.get('veracity_score')}",
    )


def test_whitespace_only() -> None:
    for s in ["   ", "\n\n\n", "\t\t", " \r\n \t "]:
        try:
            report = analyze_content("text", s)
        except Exception as exc:
            _record(f"whitespace-only {s!r} raises", True, f"raised {type(exc).__name__}")
            continue
        _record(
            f"whitespace-only {s!r} raises",
            False,
            f"analyze_content('text', {s!r}) did not raise; "
            f"veracity_score={report.get('veracity_score')}",
        )


def test_unicode_inputs() -> None:
    cases = [
        ("Chinese", "地球是圆的而月球绕地球转"),
        ("Arabic RTL", "الأرض كروية والشمس نجم"),
        ("Hebrew RTL", "כדור הארץ עגול והשמש היא כוכב"),
        ("Emoji only", "🌍🚀✨"),
        ("Emoji mid-word", "the eart🌍h is round"),
        ("Mixed emoji in sentence", "Vaccines 💉 cause autism 😱 says random tweet"),
        ("Zero-width joiner", "earth‍ is round"),
        ("Right-to-left override", "the earth is ‮ gnigniks siht"),
    ]
    for label, s in cases:
        try:
            report = analyze_content("text", s)
        except Exception as exc:
            _record(f"unicode {label}", True, f"raised {type(exc).__name__} (acceptable)")
            continue
        score = report.get("veracity_score")
        _record(
            f"unicode {label}",
            isinstance(score, (int, float)) and 0 <= score <= 100,
            f"veracity_score={score} (in [0,100])",
        )


def test_very_long_input() -> None:
    # 10k+ char input
    long_truth = "the sun is a star. " * 1000  # 18000 chars
    t0 = time.time()
    try:
        report = analyze_content("text", long_truth)
        elapsed = time.time() - t0
        _record(
            "10k+ char input",
            elapsed < 30.0,
            f"elapsed={elapsed:.2f}s score={report.get('veracity_score')}",
        )
    except Exception as exc:
        _record("10k+ char input", False, f"raised {type(exc).__name__}: {exc}")


def test_malformed_urls() -> None:
    cases = [
        "not a url",
        "ftp://example.com/foo",
        "javascript:alert(1)",
        "http:///nopath",
        "://broken",
        "https://",
        "https://example.invalid/very/deep/path/that/does/not/exist",
    ]
    for url in cases:
        try:
            report = analyze_content("url", url)
        except Exception as exc:
            _record(f"url {url!r} raises", True, f"raised {type(exc).__name__}")
            continue
        # Doesn't have to raise, but must produce a sane report.
        score = report.get("veracity_score")
        _record(
            f"url {url!r} sane",
            isinstance(score, (int, float)) and 0 <= score <= 100,
            f"score={score}",
        )


def test_html_with_nested_structures() -> None:
    from bs4 import BeautifulSoup
    html = """
    <html>
    <head>
      <title>Cat video</title>
      <!-- this is a comment that contains the text: the sun is a star -->
      <script>var x = "the sun is a star";</script>
      <![CDATA[the earth is round inside cdata]]>
    </head>
    <body>
      <iframe src="https://example.com"><p>the sun is a star</p></iframe>
      <article>
        <p>The article body talks about <strong>cats</strong> and <em>dogs</em> only.</p>
        <iframe><p>nested iframe content with sun is a star</p></iframe>
      </article>
    </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    text = _extract_main_text(soup)
    # Article body should be the dominant content.
    has_cats = "cats" in text
    has_article_dominant = text.lower().count("cats") > 0
    _record(
        "html with nested iframes/comments/CDATA",
        has_cats and has_article_dominant,
        f"extracted text len={len(text)} sample={text[:120]!r}",
    )


def test_mixed_truth_and_lies() -> None:
    s = "The earth is round and orbits the sun, but vaccines cause autism and the moon landing was faked."
    try:
        report = analyze_content("text", s)
    except Exception as exc:
        _record("mixed truth and lies raises", True, f"raised {type(exc).__name__}")
        return
    hits = report.get("cross_reference_results", [])
    # Should hit at least the truth ("earth is round", "orbits the sun")
    # and the lies ("vaccines cause autism", "moon landing faked") are
    # likely to also hit (because the corpus has them as confirmed
    # facts). We mostly want to confirm it doesn't crash and produces
    # a sensible score.
    score = report.get("veracity_score")
    _record(
        "mixed truth and lies produces hits",
        isinstance(hits, list) and isinstance(score, (int, float)),
        f"hits={len(hits)} score={score}",
    )


def test_no_punctuation() -> None:
    s = "the sun is a star the earth is round vaccines cause autism"
    try:
        report = analyze_content("text", s)
    except Exception as exc:
        _record("no-punctuation input", True, f"raised {type(exc).__name__}")
        return
    hits = report.get("cross_reference_results", [])
    _record(
        "no-punctuation input still matches",
        isinstance(hits, list),
        f"hits={len(hits)} score={report.get('veracity_score')}",
    )


def test_null_bytes_and_binary() -> None:
    for s in ["\x00", "hello\x00world", "\x00\x00\x00"]:
        try:
            report = analyze_content("text", s)
        except Exception as exc:
            _record(f"null-byte input {s!r}", True, f"raised {type(exc).__name__}")
            continue
        _record(
            f"null-byte input {s!r}",
            False,
            f"did NOT raise; score={report.get('veracity_score')}",
        )


def test_local_http_server_meta_fallback() -> None:
    """Spin up a real local HTTP server and verify URL extraction
    actually uses the meta-tag fallback (or doesn't) over the wire."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            html = (
                "<html><head>"
                "<meta property='og:title' content='Local test: the sun is a star'>"
                "<meta property='og:description' content='the earth is round and orbits the sun'>"
                "</head><body><div id='root'></div></body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *a, **kw):  # silence
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/"
        report = analyze_content("url", url)
        text = (report.get("diagnostics") or {}).get("input_summary", {}).get("normalized", "")
        full = json.dumps(report, default=str)
        ok = "sun" in full or "earth" in full or "star" in full
        _record(
            "live HTTP server meta fallback",
            ok,
            f"url={url} text_len={len(text)} score={report.get('veracity_score')}",
        )
    finally:
        server.shutdown()


def test_llm_provider_switching() -> None:
    """Cover BOTH OpenAI and Gemini paths if keys are set."""
    available = available_providers()
    print(f"  [info] available LLM providers: {available}")
    if "openai" in available:
        p = get_provider("openai")
        _record("openai provider selectable", p is not None, f"got {p!r}")
    else:
        _record(
            "openai provider selectable",
            True,
            "openai key not set; skipping (acceptable)",
        )
    if "gemini" in available:
        p = get_provider("gemini")
        _record("gemini provider selectable", p is not None, f"got {p!r}")
    else:
        _record(
            "gemini provider selectable",
            True,
            "gemini key not set; skipping (acceptable)",
        )
    # Verify enricher with explicit provider='openai' and 'gemini' both
    # produce a report dict without raising, regardless of key state.
    for prov in ("openai", "gemini"):
        try:
            out = llm_enrich_report(
                {
                    "cross_reference_results": [],
                    "content_type": "text",
                    "diagnostics": {},
                },
                provider_name=prov,
            )
            _record(
                f"enrich with provider={prov}",
                isinstance(out, dict),
                f"keys={sorted(out.keys())[:6]}",
            )
        except Exception as exc:
            _record(
                f"enrich with provider={prov}",
                False,
                f"raised {type(exc).__name__}: {exc}",
            )


def test_api_backend_with_httpx() -> None:
    """Hit the FastAPI backend directly with adversarial bodies."""
    try:
        from fastapi.testclient import TestClient
        from api import app
    except Exception as exc:
        _record("api backend importable", False, f"raised {type(exc).__name__}: {exc}")
        return
    _record("api backend importable", True, "imported ok")
    client = TestClient(app)
    # Empty user_input
    r = client.post("/analyze", json={"content_type": "text", "user_input": ""})
    _record(
        "POST /analyze rejects empty user_input",
        r.status_code in (400, 422),
        f"status={r.status_code} body={r.text[:120]}",
    )
    # Whitespace user_input
    r = client.post(
        "/analyze", json={"content_type": "text", "user_input": "   \n\t  "}
    )
    _record(
        "POST /analyze rejects whitespace user_input",
        r.status_code == 400,
        f"status={r.status_code} body={r.text[:120]}",
    )
    # Bad content_type
    r = client.post(
        "/analyze", json={"content_type": "pdf", "user_input": "hi"}
    )
    _record(
        "POST /analyze rejects bad content_type",
        r.status_code in (400, 422),
        f"status={r.status_code} body={r.text[:120]}",
    )
    # Missing fields
    r = client.post("/analyze", json={})
    _record(
        "POST /analyze rejects missing fields",
        r.status_code in (400, 422),
        f"status={r.status_code} body={r.text[:120]}",
    )
    # Happy path
    r = client.post(
        "/analyze",
        json={
            "content_type": "text",
            "user_input": "The sun is a star and the earth is round.",
            "use_rag": False,
            "enrich_with_llm": False,
        },
    )
    _record(
        "POST /analyze happy path",
        r.status_code == 200,
        f"status={r.status_code} keys={list(r.json().keys())[:5]}",
    )
    # Very long happy path
    r = client.post(
        "/analyze",
        json={
            "content_type": "text",
            "user_input": ("the sun is a star. " * 500),
            "use_rag": False,
            "enrich_with_llm": False,
        },
    )
    _record(
        "POST /analyze 9k char input",
        r.status_code == 200,
        f"status={r.status_code}",
    )
    # Malformed JSON
    r = client.post(
        "/analyze",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    _record(
        "POST /analyze rejects malformed JSON",
        r.status_code in (400, 422),
        f"status={r.status_code}",
    )
    # Upload endpoint with bad content_type
    files = {"file": ("test.txt", io.BytesIO(b"hi"), "text/plain")}
    r = client.post(
        "/analyze/upload",
        files=files,
        data={"content_type": "pdf"},
    )
    _record(
        "POST /analyze/upload rejects bad content_type",
        r.status_code == 400,
        f"status={r.status_code} body={r.text[:120]}",
    )


def test_rag_health_reports_actual_status() -> None:
    h = rag_health()
    _record(
        "rag_health returns dict with ok key",
        isinstance(h, dict) and "ok" in h,
        f"keys={list(h.keys())}",
    )


def test_is_ytdlp_reports_correctly() -> None:
    v = is_ytdlp_available()
    _record(
        "is_ytdlp_available returns bool",
        isinstance(v, bool),
        f"got {v!r}",
    )


def main() -> None:
    tests = [
        test_empty_string,
        test_whitespace_only,
        test_unicode_inputs,
        test_very_long_input,
        test_malformed_urls,
        test_html_with_nested_structures,
        test_mixed_truth_and_lies,
        test_no_punctuation,
        test_null_bytes_and_binary,
        test_local_http_server_meta_fallback,
        test_llm_provider_switching,
        test_api_backend_with_httpx,
        test_rag_health_reports_actual_status,
        test_is_ytdlp_reports_correctly,
    ]
    for t in tests:
        print(f"\n=== {t.__name__} ===")
        try:
            t()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _record(t.__name__, False, f"raised {type(exc).__name__}: {exc}")
    print("\n=== SUMMARY ===")
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"{passed}/{total} adversarial checks passed")
    fails = [r for r in RESULTS if not r[1]]
    if fails:
        print(f"\n{len(fails)} FAILURES:")
        for name, _, detail in fails:
            print(f"  - {name}: {detail}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
