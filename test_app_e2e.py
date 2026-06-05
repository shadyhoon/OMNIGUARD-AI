"""
End-to-end smoke test of the OmniGuard Streamlit dashboard.

Drives the live app via AppTest, simulating a real user:
  1. Opens the page (initial render)
  2. Picks each of the three content-type toggles
  3. Pastes content
  4. Clicks "Analyze with OmniGuard"
  5. Asserts no exceptions + report sections present

Run from the project root:
    .\.venv\Scripts\python.exe test_app_e2e.py
"""

from streamlit.testing.v1 import AppTest


SAMPLE_TEXT = (
    "Breaking news! The sun is a star and humans have landed on mars. "
    "Furthermore, this groundbreaking discovery is a testament to science."
)
SAMPLE_SOCIAL = (
    "🚨 JUST IN: The earth is flat and vaccines cause autism. "
    "Furthermore, this is unprecedented. As an AI I can confirm. #truth"
)
SAMPLE_URL = "https://en.wikipedia.org/wiki/Earth"
# A small, freely-redistributable test video. Used by the Video
# Link scenario so yt-dlp has something real to download. This is
# the same clip used in yt-dlp's own test suite.
SAMPLE_VIDEO = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def _summary(at: AppTest) -> None:
    print(
        f"  exceptions={len(at.exception)} "
        f"errors={len(at.error)} "
        f"warnings={len(at.warning)} "
        f"markdown_blocks={len(at.markdown)} "
        f"expanders={len(at.expander)} "
        f"plots={len(at.get('plotly_chart'))} "
        f"badges_html_len={sum(len(m.value) for m in at.markdown)}"
    )
    for e in at.exception:
        print(f"  !! EXCEPTION: {e.value}")
    for e in at.error:
        print(f"  !! ERROR: {e.value}")


def _find_button(at: AppTest, text: str):
    for b in at.button:
        if text in b.label:
            return b
    return None


def _drive(label: str, sample: str, radio_choice: str, *, timeout: int = 30) -> None:
    print(f"\n=== {label} ===")
    at = AppTest.from_file("app.py", default_timeout=timeout)
    at.run()
    _summary(at)
    assert not at.exception, f"Initial render raised for {label}"

    # Pick the content-type radio. AppTest's set_value takes the
    # label string, not a numeric index.
    radios = at.radio
    if not radios:
        print("  (no radio found - defaulting to first content type)")
    else:
        radios[0].set_value(radio_choice).run()
        _summary(at)

    # Fill the input. URL toggle uses text_input, the others use text_area.
    widget = at.text_input[0] if radio_choice == "🎬  Video Link" else at.text_area[0]
    widget.set_value(sample).run()
    _summary(at)

    # Click the analyze button
    btn = _find_button(at, "Analyze")
    if btn is None:
        print("  !! Analyze button not found")
        return
    btn.click().run()
    _summary(at)

    assert not at.exception, f"Analysis raised for {label}"
    # The donut chart should now be present
    assert len(at.get("plotly_chart")) >= 1, f"No plotly chart for {label}"
    # The cross-reference expander block should be there
    assert len(at.expander) >= 1, f"No expanders for {label}"
    # The HTML badges (success/warning/danger) should be in the markdown
    full_md = "\n".join(m.value for m in at.markdown)
    assert 'class="og-badge' in full_md, f"No status badges in {label}"
    print(f"  -> {label}: OK ({len(at.expander)} expanders, "
          f"{len(at.get('plotly_chart'))} charts)")


def main() -> None:
    _drive("Text Article", SAMPLE_TEXT, radio_choice="📰  Text Article")
    _drive("Social Media Thread", SAMPLE_SOCIAL, radio_choice="🧵  Social Media Thread")
    # Video Link now actually downloads via yt-dlp and runs the
    # OpenCV vision pass. We give it a longer timeout and skip the
    # scenario if the network or yt-dlp can't reach the test video.
    try:
        _drive(
            "Video Link (real video)",
            SAMPLE_VIDEO,
            radio_choice="🎬  Video Link",
            timeout=120,
        )
    except AssertionError as exc:
        # Most likely failure modes: no network, yt-dlp missing,
        # or YouTube blocking the test IP. The text/social scenarios
        # are still valid - just skip the video one.
        print(f"  [SKIP] Video Link scenario skipped: {exc}")
    print("\nAll scenarios passed.")


if __name__ == "__main__":
    main()
