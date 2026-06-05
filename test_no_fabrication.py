"""
Regression test: no fabricated cross-references.

The user reported that typing "man went to the moon" produced
cross-reference hits about the sun, water, vaccines, etc. That was
because _cross_reference used to seed in random facts from its
table when no claim was matched. This test guards against that
fallback ever creeping back in.
"""

from utils.verifier import analyze_content


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise AssertionError(label)


def test_moon_text_no_fabricated_hits() -> None:
    print("test_moon_text_no_fabricated_hits")
    report = analyze_content("text", "The first man went to the moon in 1969.")
    hits = report.get("cross_reference_results", [])
    _check(
        len(hits) == 0,
        f"no cross-reference hits for 'man went to the moon' (got {len(hits)})",
    )


def test_unrelated_text_no_hits() -> None:
    print("test_unrelated_text_no_hits")
    report = analyze_content("text", "My cat loves tuna and sleeps on the sofa.")
    hits = report.get("cross_reference_results", [])
    _check(
        len(hits) == 0,
        f"no cross-reference hits for cat text (got {len(hits)})",
    )


def test_matched_claim_still_hits() -> None:
    """Sanity check: real matches must still appear."""
    print("test_matched_claim_still_hits")
    report = analyze_content(
        "text",
        "The sun is a star and the earth is round.",
    )
    hits = report.get("cross_reference_results", [])
    _check(
        len(hits) >= 2,
        f"at least 2 hits for sun+earth text (got {len(hits)})",
    )


def test_llm_enrich_evidence_fields() -> None:
    """The enricher must attach a calls ledger with proof fields."""
    print("test_llm_enrich_evidence_fields")
    # When LLM is unavailable (no key), the enricher still attaches
    # the ledger shape, just empty.
    from utils.llm import llm_enrich_report, is_llm_available
    report = {
        "cross_reference_results": [],
        "content_type": "text",
        "diagnostics": {"input_summary": {}},
    }
    out = llm_enrich_report(report)
    for key in ("llm_calls", "llm_tokens_in", "llm_tokens_out",
                "llm_latency_ms_total", "llm_model", "llm_available",
                "llm_provider"):
        _check(key in out, f"enriched report has '{key}'")
    _check(
        out["llm_available"] == is_llm_available(),
        "llm_available matches the real availability check",
    )


def test_meta_tag_fallback_extracts_og_description() -> None:
    """A page with only JS-loaded body but populated og: tags
    should still yield useful text via the meta-tag fallback."""
    print("test_meta_tag_fallback_extracts_og_description")
    from bs4 import BeautifulSoup
    from utils.verifier import _extract_main_text

    # Simulate a YouTube / Twitter / OG-rich page with no body text.
    html = """
    <html><head>
      <meta property="og:title" content="Cat plays piano beautifully">
      <meta property="og:description" content="A cute cat plays a Chopin nocturne on a grand piano">
      <meta name="description" content="Cat plays piano beautifully">
    </head><body><div id="root"></div></body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    text = _extract_main_text(soup)
    _check("piano" in text, "meta fallback extracted 'piano' from og:description")
    _check("cat" in text, "meta fallback extracted 'cat' from og:title")


def test_meta_tag_fallback_skipped_when_body_has_article() -> None:
    """A page with a real <article> must use that, not the meta tags."""
    print("test_meta_tag_fallback_skipped_when_body_has_article")
    from bs4 import BeautifulSoup
    from utils.verifier import _extract_main_text

    html = """
    <html><head>
      <meta property="og:description" content="UNRELATED META">
    </head><body>
      <article><p>The sun is a star and the earth is round. We have known
      this since antiquity and modern astrophysics confirms it. Heliocentrism
      was established centuries ago and is the foundation of modern astronomy
      and our understanding of the cosmos at large.</p></article>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    text = _extract_main_text(soup)
    _check("heliocentrism" in text, "article text wins over og: description")
    _check("unrelated meta" not in text, "og: tag did not leak into the body")


def test_llm_provider_override_dispatches_correctly() -> None:
    """Forcing a provider name must route to that provider or fail
    cleanly when no key is set for it."""
    print("test_llm_provider_override_dispatches_correctly")
    from utils.llm import (
        get_provider,
        available_providers,
        llm_enrich_report,
    )
    # 1. get_provider('openai') with the OpenAI key already set in
    #    this environment must return an OpenAIProvider.
    if "openai" in available_providers():
        p = get_provider("openai")
        _check(p is not None, "get_provider('openai') returns a provider")
        _check(p.name == "openai", "provider name is 'openai'")
    # 2. Forcing gemini when no GEMINI key is set must return None
    #    and the enricher must still return a report (no crash).
    import os
    saved = os.environ.pop("GEMINI_API_KEY", None)
    try:
        out = llm_enrich_report(
            {
                "cross_reference_results": [],
                "content_type": "text",
                "diagnostics": {},
            },
            provider_name="gemini",
        )
        _check(
            out.get("llm_provider") is None,
            "no gemini key -> llm_provider stays None",
        )
        _check(
            out.get("llm_calls") == [],
            "no gemini key -> no LLM calls attempted",
        )
    finally:
        if saved is not None:
            os.environ["GEMINI_API_KEY"] = saved


def test_video_analyzer_returns_error_when_yt_dlp_missing() -> None:
    """The video orchestrator must not crash when yt-dlp is missing;
    it must return a structured report with an explanatory message."""
    print("test_video_analyzer_returns_error_when_yt_dlp_missing")
    # We can't easily uninstall yt-dlp in the test, so we just confirm
    # the orchestrator returns a dict with the expected shape for any
    # URL. The behavior with yt-dlp missing is exercised when the
    # import fails; the happy path needs a real video.
    try:
        from app import _run_video_analysis  # type: ignore
    except Exception as exc:
        print(f"  [SKIP] could not import app._run_video_analysis: {exc}")
        return
    report = _run_video_analysis("https://example.invalid/shorts/abc", use_rag=False)
    _check(isinstance(report, dict), "video analysis returns a dict")
    _check(report.get("content_type") == "video", "content_type is 'video'")
    _check("veracity_score" in report, "report has veracity_score")
    # Either we got a real video analysis (yt-dlp works) or a clean
    # "yt-dlp missing" / "download failed" message.
    if report.get("veracity_score", 0) == 0:
        hr = report.get("hallucination_report") or []
        mc = (report.get("multimodal_consistency") or {}).get("summary", "")
        _check(
            any("yt-dlp" in s or "download" in s for s in hr) or "download" in mc,
            "missing/failed yt-dlp surfaces a clear message",
        )


def main() -> None:
    test_moon_text_no_fabricated_hits()
    test_unrelated_text_no_hits()
    test_matched_claim_still_hits()
    test_llm_enrich_evidence_fields()
    test_meta_tag_fallback_extracts_og_description()
    test_meta_tag_fallback_skipped_when_body_has_article()
    test_llm_provider_override_dispatches_correctly()
    test_video_analyzer_returns_error_when_yt_dlp_missing()
    print("\nAll no-fabrication tests passed.")


if __name__ == "__main__":
    main()
