"""
End-to-end test of the OmniGuard full stack.

Covers:
  * RAG cross-reference engine (Chroma + DuckDuckGo fallback)
  * OpenCV vision layer (synthetic image)
  * FastAPI backend (TestClient, no real server)
  * Pipeline integration with use_rag=True
  * Substring-noise regression guard for the URL/cross-ref path

Run from the project root:
    .\.venv\Scripts\python.exe test_full_stack.py
"""

import io
import sys
import time

import numpy as np


def _check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise AssertionError(label)


# ---------------------------------------------------------------------
# 1. RAG engine
# ---------------------------------------------------------------------
def test_rag_engine() -> None:
    print("test_rag_engine")
    from utils.rag import rag_cross_reference, rag_health

    health = rag_health()
    _check(health["chroma_ok"], "Chroma is loadable")
    _check(health["embedding_model_ok"], "Embedding model is loadable")
    _check(health["chroma_count"] >= 10, f"Chroma has at least 10 truths (got {health['chroma_count']})")

    hits = rag_cross_reference(
        "The sun is a star at the centre of the solar system.",
        max_claims=2,
    )
    _check(len(hits) >= 1, "RAG returned at least one hit")
    _check(
        any(h["status"] == "confirmed" for h in hits),
        "RAG hit is 'confirmed'",
    )
    _check(
        any(h.get("engine") == "chroma" for h in hits),
        "Hit came from Chroma (local vector store)",
    )


# ---------------------------------------------------------------------
# 2. OpenCV vision
# ---------------------------------------------------------------------
def test_vision_layer() -> None:
    print("test_vision_layer")
    try:
        import cv2
    except ImportError as exc:
        print(f"  [SKIP] OpenCV not installed: {exc}")
        return

    from utils.vision import analyse_image_bytes, vision_health

    _check(vision_health()["opencv_ok"], "OpenCV health probe OK")

    # Build a synthetic 256x256 BGR image: cool blue background with a
    # warm rectangle in the centre. The colour cast is intentionally
    # mismatched so the lighting check should flag it.
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :] = (180, 80, 60)  # cool background (BGR)
    img[64:192, 64:192] = (60, 180, 200)  # warm centre (BGR)
    ok, buf = cv2.imencode(".png", img)
    _check(ok, "encoded synthetic image to PNG")

    result = analyse_image_bytes(buf.tobytes())
    _check(result.get("ok"), "analyse_image_bytes reports ok=True")
    _check(
        result.get("lighting_anomaly_score", 0) >= 0.5,
        f"lighting_anomaly_score flags the colour mismatch (got {result.get('lighting_anomaly_score')})",
    )


# ---------------------------------------------------------------------
# 3. FastAPI backend
# ---------------------------------------------------------------------
def test_api_backend() -> None:
    print("test_api_backend")
    from fastapi.testclient import TestClient

    from api import app

    client = TestClient(app)

    r = client.get("/")
    _check(r.status_code == 200, f"GET / -> 200 (got {r.status_code})")
    _check(r.json()["service"] == "OmniGuard AI API", "service name correct")

    r = client.get("/health")
    _check(r.status_code == 200, "GET /health -> 200")
    h = r.json()
    _check(h["status"] == "ok", "health status is 'ok'")
    _check(h["heuristics"] is True, "heuristics flag is True")

    r = client.post(
        "/analyze",
        json={
            "content_type": "text",
            "user_input": "The sun is a star and vaccines do not cause autism.",
            "use_rag": False,
            "enrich_with_llm": False,
        },
    )
    _check(r.status_code == 200, f"POST /analyze -> 200 (got {r.status_code})")
    rep = r.json()
    _check("veracity_score" in rep, "response has veracity_score")
    _check(0 <= rep["veracity_score"] <= 100, "veracity_score in [0, 100]")

    # RAG path
    r = client.post(
        "/analyze",
        json={
            "content_type": "text",
            "user_input": "The first man walked on the Moon in 1969 on Apollo 11.",
            "use_rag": True,
            "enrich_with_llm": False,
        },
    )
    _check(r.status_code == 200, "POST /analyze (RAG) -> 200")
    rep = r.json()
    _check(
        len(rep.get("cross_reference_results", [])) >= 1,
        f"RAG path returned at least 1 hit (got {len(rep.get('cross_reference_results', []))})",
    )


# ---------------------------------------------------------------------
# 4. No fabricated cross-references
# ---------------------------------------------------------------------
def test_no_fabrication() -> None:
    print("test_no_fabrication")
    from utils.verifier import analyze_content

    rep = analyze_content("text", "The first man went to the moon in 1969.")
    _check(
        len(rep.get("cross_reference_results", [])) == 0,
        "unrelated text returns no fabricated cross-reference hits",
    )

    rep = analyze_content("text", "The sun is a star and the earth is round.")
    _check(
        len(rep.get("cross_reference_results", [])) >= 2,
        "real matches still produce hits",
    )


# ---------------------------------------------------------------------
# 5. End-to-end latency
# ---------------------------------------------------------------------
def test_latency_budget() -> None:
    print("test_latency_budget")
    from utils.verifier import analyze_content

    t0 = time.time()
    rep = analyze_content(
        "text",
        "The sun is a star and vaccines do not cause autism.",
    )
    elapsed = time.time() - t0
    _check(
        elapsed < 2.0,
        f"heuristic text path completes in <2s (took {elapsed:.2f}s)",
    )
    _check(
        "veracity_score" in rep,
        "report has veracity_score",
    )


def main() -> None:
    test_rag_engine()
    test_vision_layer()
    test_api_backend()
    test_no_fabrication()
    test_latency_budget()
    print("\nAll full-stack tests passed.")


if __name__ == "__main__":
    main()
