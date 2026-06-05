"""
End-to-end smoke test for the FastAPI backend.
Uses FastAPI's TestClient so we don't need a real server.
"""

from fastapi.testclient import TestClient

from api import app


def main() -> None:
    client = TestClient(app)

    # 1. Root
    r = client.get("/")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["service"] == "OmniGuard AI API"
    print(f"  [PASS] GET /  -> {body['service']} v{body['version']}")

    # 2. Health
    r = client.get("/health")
    assert r.status_code == 200, r.text
    h = r.json()
    print(f"  [PASS] GET /health -> heuristics={h['heuristics']}, "
          f"chroma_ok={h['rag'].get('chroma_ok')}, "
          f"vision_ok={h['vision'].get('opencv_ok')}, "
          f"llm_available={h['llm'].get('available')}")

    # 3. Analyze (text, no RAG)
    r = client.post(
        "/analyze",
        json={"content_type": "text",
              "user_input": "The sun is a star and vaccines do not cause autism.",
              "use_rag": False,
              "enrich_with_llm": False},
    )
    assert r.status_code == 200, r.text
    rep = r.json()
    print(f"  [PASS] POST /analyze (text) -> veracity={rep['veracity_score']}, "
          f"hits={len(rep['cross_reference_results'])}")

    # 4. Analyze (text, WITH RAG)
    r = client.post(
        "/analyze",
        json={"content_type": "text",
              "user_input": "The first man went to the Moon in 1969 on Apollo 11.",
              "use_rag": True,
              "enrich_with_llm": False},
    )
    assert r.status_code == 200, r.text
    rep = r.json()
    engines = {h.get("engine") for h in rep.get("cross_reference_results", [])}
    print(f"  [PASS] POST /analyze (text, RAG) -> veracity={rep['veracity_score']}, "
          f"engines={engines}")

    # 5. Analyze (URL, no RAG)
    r = client.post(
        "/analyze",
        json={"content_type": "url",
              "user_input": "https://en.wikipedia.org/wiki/Earth",
              "use_rag": False,
              "enrich_with_llm": False},
    )
    assert r.status_code == 200, r.text
    print(f"  [PASS] POST /analyze (url) -> "
          f"veracity={r.json()['veracity_score']}")

    # 6. Bad request (the /analyze endpoint only accepts text/url;
    # image/video/audio go through /analyze/upload).
    # pydantic's regex pre-validates the content_type, so the
    # response is 422 (validation error) before the route runs.
    r = client.post("/analyze", json={"content_type": "image", "user_input": "x"})
    assert r.status_code in (400, 422), r.text
    print(f"  [PASS] POST /analyze with bad type -> {r.status_code}")

    print("\nAll API smoke tests passed.")


if __name__ == "__main__":
    main()
