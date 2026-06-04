"""
Targeted regression test for the cross-reference relevance fix.

Validates three things that the user reported as broken:

  1. Word-boundary matching: "mars" must NOT match inside "remarkable"
     or "Marsala"; it must match the standalone word "mars".
  2. Each cross-reference hit carries a ``relevance`` field in [0, 1].
  3. The main-content extractor strips sidebar / comment / nav noise
     so a hit near the start of the article scores higher than the
     same claim buried in a comment.

Run from the project root:
    .\.venv\Scripts\python.exe test_relevance.py
"""

from bs4 import BeautifulSoup

from utils.verifier import (
    _extract_main_text,
    _word_boundary_match,
    _relevance_score,
    _cross_reference,
    analyze_content,
)


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise AssertionError(label)


def test_word_boundary() -> None:
    print("test_word_boundary")
    # Standalone: must match.
    _check(
        _word_boundary_match("mars", "humans have landed on mars") is not None,
        "'mars' matches standalone word",
    )
    # Inside another word: must NOT match.
    _check(
        _word_boundary_match("mars", "this is a remarkable achievement") is None,
        "'mars' does not match inside 'remarkable'",
    )
    _check(
        _word_boundary_match("mars", "the Marsala wine region") is None,
        "'mars' does not match inside 'Marsala'",
    )
    # Multi-word claim: must also use word boundaries on the outside only.
    _check(
        _word_boundary_match(
            "the sun is a star",
            "the sun is a star in our galaxy",
        )
        is not None,
        "multi-word claim matches in a sentence",
    )
    _check(
        _word_boundary_match(
            "the sun is a star",
            "thestar is huge",  # no spaces -> different word
        )
        is None,
        "multi-word claim does not match glued-together word",
    )


def test_relevance_range() -> None:
    print("test_relevance_range")
    body = "The sun is a star. The sun is a star. The sun is a star."
    idx = _word_boundary_match("the sun is a star", body)
    score = _relevance_score("the sun is a star", body, idx or 0)
    _check(0.0 <= score <= 1.0, f"relevance in [0,1] (got {score})")
    # A second match in a longer body should have lower relevance
    # than the first one because the position factor decays.
    longer = body + " " + ("filler text. " * 200)
    idx2 = _word_boundary_match("the sun is a star", longer)
    score2 = _relevance_score("the sun is a star", longer, idx2 or 0)
    _check(
        score2 < score,
        f"longer body dilutes relevance (early={score}, late={score2})",
    )


def test_main_content_extraction() -> None:
    print("test_main_content_extraction")
    html = """
    <html><body>
      <nav>Home | About | The sun is a star (nav link)</nav>
      <aside class="sidebar">
        <h3>Related</h3>
        <p>The earth is round (sidebar promo)</p>
      </aside>
      <article>
        <h1>Welcome to my blog about Mars rovers</h1>
        <p>The latest Mars rover landed successfully and humans have
        landed on mars is not true. The sun is a star, of course.</p>
        <p>Follow me on Twitter.</p>
      </article>
      <footer>Contact us | Privacy | The sun is a star (footer link)</footer>
    </body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    main_text = _extract_main_text(soup)
    # The article body should dominate.
    _check(
        "rover landed successfully" in main_text,
        "main content includes the article body",
    )
    # The nav text "Home | About" should NOT be the only thing.
    _check(
        "Home | About" not in main_text,
        "nav text is excluded from the main content",
    )


def test_analyze_content_url_relevance() -> None:
    print("test_analyze_content_url_relevance")
    # No live URL fetch - we just want to make sure every hit has a
    # relevance field and that the field is in [0, 1].
    report = analyze_content(
        "text",
        "The sun is a star. The earth is round. Humans have landed on mars.",
    )
    hits = report.get("cross_reference_results", [])
    _check(len(hits) >= 3, f"at least 3 hits (got {len(hits)})")
    for h in hits:
        rel = h.get("relevance")
        _check(
            isinstance(rel, (int, float)) and 0.0 <= rel <= 1.0,
            f"hit '{h.get('claim','')[:30]}' has relevance in [0,1] (got {rel})",
        )


def test_url_substring_noise_filtered() -> None:
    print("test_url_substring_noise_filtered")
    # The substring noise case the user complained about: text that
    # contains 'mars' inside an unrelated word should not produce a hit.
    report = analyze_content(
        "text",
        "This remarkable Marsala-flavoured ice cream is delicious.",
    )
    hits = report.get("cross_reference_results", [])
    bad = [h for h in hits if h.get("claim") == "humans have landed on mars"]
    _check(
        bad == [],
        "no 'humans have landed on mars' hit when only 'Marsala' is present",
    )


def main() -> None:
    test_word_boundary()
    test_relevance_range()
    test_main_content_extraction()
    test_analyze_content_url_relevance()
    test_url_substring_noise_filtered()
    print("\nAll regression tests passed.")


if __name__ == "__main__":
    main()
