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
                "llm_latency_ms_total", "llm_model", "llm_available"):
        _check(key in out, f"enriched report has '{key}'")
    _check(
        out["llm_available"] == is_llm_available(),
        "llm_available matches the real availability check",
    )


def main() -> None:
    test_moon_text_no_fabricated_hits()
    test_unrelated_text_no_hits()
    test_matched_claim_still_hits()
    test_llm_enrich_evidence_fields()
    print("\nAll no-fabrication tests passed.")


if __name__ == "__main__":
    main()
