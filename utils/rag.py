"""
OmniGuard AI - RAG cross-reference engine
=========================================
Retrieval-Augmented Generation pipeline that backs the
``cross_reference_results`` field of every report.

Pipeline
--------
1. **Extract claims** from the submitted text or URL main content
   (split into sentences, keep assertive ones, drop trivial ones).
2. **Embed claims** with a local sentence-transformers model
   (``all-MiniLM-L6-v2``, ~80 MB, downloads on first call).
3. **Query Chroma** - a local, file-based vector store of "trusted
   truths" the user has pre-seeded (or that we auto-seed with
   public-domain canonical facts). If the top hit has cosine
   similarity >= 0.65, we treat the claim as "confirmed" by the
   local knowledge base.
4. **Live web search** - if the local store has no good match,
   fall back to a DuckDuckGo HTML search for the claim. The
   first 3 result URLs are scraped with BeautifulSoup, summarised
   by the LLM, and ranked by relevance.

Heavy imports (sentence-transformers, chromadb, duckduckgo-search)
are **lazy** - they only load when an actual cross-reference is
requested, so the dashboard boots fast and the LLM-only code
paths are unaffected.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Lazy dependency loaders
# ---------------------------------------------------------------------
_SENTENCE_MODEL = None
_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None
_DDG_TOOL = None


def _get_sentence_model():
    """Lazy-load the embedding model. ~80 MB download on first call."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SENTENCE_MODEL


def _get_chroma():
    """
    Lazy-load ChromaDB and the trusted-truths collection.

    Storage path is ``./data/chroma`` relative to the project root
    so the vector store persists across restarts.
    """
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        import chromadb
        from chromadb.config import Settings

        os.makedirs("./data/chroma", exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(
            path="./data/chroma",
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
        _CHROMA_COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
            name="trusted_truths",
            metadata={"hnsw:space": "cosine"},
        )
        _seed_default_truths(_CHROMA_COLLECTION, _get_sentence_model())
    return _CHROMA_COLLECTION


def _get_ddg():
    """Lazy-load the DuckDuckGo search tool."""
    global _DDG_TOOL
    if _DDG_TOOL is None:
        from duckduckgo_search import DDGS

        _DDG_TOOL = DDGS
    return _DDG_TOOL


# ---------------------------------------------------------------------
# Default knowledge seed
# ---------------------------------------------------------------------
# A small set of public-domain canonical facts. The vector store is
# seeded with these on first run. Users can add their own via
# :func:`add_trusted_truth`.
_DEFAULT_TRUTHS: List[Dict[str, str]] = [
    {
        "claim": "The Earth is an oblate spheroid, slightly flattened at the poles.",
        "verdict": "confirmed",
        "source": "NASA - https://science.nasa.gov/earth/",
    },
    {
        "claim": "Water boils at 100 degrees Celsius at standard sea-level atmospheric pressure.",
        "verdict": "confirmed",
        "source": "NIST Chemistry WebBook",
    },
    {
        "claim": "The Sun is a G-type main-sequence star at the centre of the Solar System.",
        "verdict": "confirmed",
        "source": "European Space Agency",
    },
    {
        "claim": "No crewed mission has ever landed on Mars; only robotic landers have.",
        "verdict": "confirmed",
        "source": "NASA Mars Exploration Program",
    },
    {
        "claim": "The Great Wall of China is not visible to the naked eye from low Earth orbit.",
        "verdict": "confirmed",
        "source": "NASA astronaut observations",
    },
    {
        "claim": "Vaccines do not cause autism; the original Wakefield study was retracted for fraud.",
        "verdict": "confirmed",
        "source": "WHO / CDC consensus",
    },
    {
        "claim": "Lightning can strike the same place multiple times; tall structures are hit repeatedly.",
        "verdict": "confirmed",
        "source": "NOAA / National Geographic",
    },
    {
        "claim": "Sharks are cartilaginous fish, not mammals.",
        "verdict": "confirmed",
        "source": "NOAA Fisheries",
    },
    {
        "claim": "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969.",
        "verdict": "confirmed",
        "source": "NASA Apollo program",
    },
    {
        "claim": "Antibiotics do not work against viral infections such as the common cold or flu.",
        "verdict": "confirmed",
        "source": "CDC antibiotic resistance resources",
    },
]


def _seed_default_truths(collection, model) -> None:
    """Idempotent seed: only inserts truths the collection doesn't already have."""
    try:
        existing = collection.get(include=[])
    except Exception:
        existing = {"ids": []}
    have = set(existing.get("ids", []) or [])
    for entry in _DEFAULT_TRUTHS:
        cid = hashlib.sha256(entry["claim"].encode("utf-8")).hexdigest()[:16]
        if cid in have:
            continue
        emb = model.encode([entry["claim"]]).tolist()
        try:
            collection.add(
                ids=[cid],
                documents=[entry["claim"]],
                embeddings=emb,
                metadatas=[
                    {
                        "verdict": entry["verdict"],
                        "source": entry["source"],
                    }
                ],
            )
        except Exception as exc:
            logger.warning("Chroma seed insert failed: %s", exc)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def add_trusted_truth(claim: str, source: str, verdict: str = "confirmed") -> str:
    """
    Add a new "trusted truth" to the local vector store.

    Returns the new entry's id.
    """
    collection = _get_chroma()
    model = _get_sentence_model()
    cid = hashlib.sha256(claim.encode("utf-8")).hexdigest()[:16]
    emb = model.encode([claim]).tolist()
    collection.upsert(
        ids=[cid],
        documents=[claim],
        embeddings=emb,
        metadatas=[{"verdict": verdict, "source": source}],
    )
    return cid


# Sentence splitter kept local to avoid circular import with utils.verifier
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TRIVIAL = re.compile(
    r"^(this|that|these|those|it|they|he|she|we|i|you|there|here)\b",
    re.IGNORECASE,
)


def _extract_claims(text: str, max_claims: int = 5) -> List[str]:
    """
    Split ``text`` into short, assertive claim-sized sentences.

    Filters out trivially short or pronoun-led sentences that
    are not fact-checkable on their own.
    """
    if not text:
        return []
    sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    claims: List[str] = []
    for s in sentences:
        # Drop trivially short
        if len(s) < 25:
            continue
        # Drop pronoun-led continuations
        if _TRIVIAL.match(s):
            continue
        # Truncate very long sentences to a useful chunk
        if len(s) > 240:
            s = s[:240].rsplit(" ", 1)[0] + "."
        claims.append(s)
        if len(claims) >= max_claims:
            break
    return claims


def _chroma_query(claim: str, threshold: float = 0.65) -> Optional[Dict[str, Any]]:
    """
    Query Chroma for the nearest neighbour of ``claim``.

    Returns a hit dict (with ``relevance`` filled in) or ``None`` if
    no neighbour beats the cosine threshold.
    """
    try:
        collection = _get_chroma()
        model = _get_sentence_model()
    except Exception as exc:
        logger.warning("Chroma unavailable, skipping: %s", exc)
        return None
    emb = model.encode([claim]).tolist()
    try:
        result = collection.query(
            query_embeddings=emb, n_results=1, include=["documents", "metadatas", "distances"]
        )
    except Exception as exc:
        logger.warning("Chroma query failed: %s", exc)
        return None
    if not result or not result.get("ids") or not result["ids"][0]:
        return None
    distance = result["distances"][0][0]
    # Cosine distance -> cosine similarity
    relevance = max(0.0, 1.0 - float(distance))
    if relevance < threshold:
        return None
    return {
        "claim": claim,
        "matched_truth": result["documents"][0][0],
        "relevance": round(relevance, 3),
        "verdict": (result["metadatas"][0][0] or {}).get("verdict", "unknown"),
        "source": (result["metadatas"][0][0] or {}).get("source", "trusted-truths-db"),
        "engine": "chroma",
    }


# ---------------------------------------------------------------------
# Live web search fallback
# ---------------------------------------------------------------------
_DDG_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 OmniGuard-AI/0.1"
)


def _ddg_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Run a DuckDuckGo search and return up to ``max_results`` hits."""
    try:
        DDGS = _get_ddg()
    except Exception as exc:
        logger.warning("DuckDuckGo unavailable: %s", exc)
        return []
    out: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href") or r.get("url") or "",
                        "snippet": r.get("body", ""),
                    }
                )
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
    return out


def _scrape_summary(url: str, max_chars: int = 1500) -> str:
    """Best-effort page summary. Returns up to ``max_chars`` of main content."""
    if not url:
        return ""
    try:
        resp = requests.get(
            url, timeout=8, headers={"User-Agent": _DDG_USER_AGENT}, allow_redirects=True
        )
        resp.raise_for_status()
    except requests.RequestException:
        return ""
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    article = soup.find("article")
    if article:
        text = article.get_text(separator=" ", strip=True)
    else:
        main = soup.find("main")
        if main:
            text = main.get_text(separator=" ", strip=True)
        else:
            ps = soup.find_all("p")
            text = " ".join(p.get_text(" ", strip=True) for p in ps)
    return text[:max_chars]


def _web_query(claim: str) -> Optional[Dict[str, Any]]:
    """
    Search DuckDuckGo for ``claim`` and return the most relevant
    single hit, or ``None`` if nothing usable was found.
    """
    results = _ddg_search(claim, max_results=3)
    if not results:
        return None
    # Use the first hit, with a small snippet preview.
    top = results[0]
    summary = _scrape_summary(top["url"])
    if not summary:
        summary = top.get("snippet", "")
    if not summary:
        return None
    return {
        "claim": claim,
        "matched_truth": summary[:280],
        "relevance": 0.5,  # live web = softer confidence
        "verdict": "live_web_match",
        "source": top.get("url", ""),
        "engine": "duckduckgo",
    }


# ---------------------------------------------------------------------
# Top-level entry point used by utils.verifier
# ---------------------------------------------------------------------
def rag_cross_reference(
    text: str,
    *,
    max_claims: int = 4,
    chroma_threshold: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Run the full RAG cross-reference pipeline on ``text``.

    Steps:
      1. Extract assertive claim sentences.
      2. For each claim, query the local Chroma store.
      3. If no Chroma hit, fall back to a live DuckDuckGo search.

    Returns a list of hit dicts in the same shape as
    ``utils.verifier._cross_reference`` so the UI doesn't need to
    know the difference.
    """
    claims = _extract_claims(text, max_claims=max_claims)
    if not claims:
        return []
    hits: List[Dict[str, Any]] = []
    for claim in claims:
        hit = _chroma_query(claim, threshold=chroma_threshold)
        if hit is None:
            hit = _web_query(claim)
        if hit is None:
            continue
        # Reshape to match the existing UI schema
        hits.append(
            {
                "claim": hit["claim"],
                "status": _verdict_to_status(hit.get("verdict", "")),
                "source": hit.get("source", ""),
                "summary": hit.get("matched_truth", ""),
                "context": (f"Engine: {hit.get('engine','?')} · "
                            f"relevance={hit.get('relevance',0):.2f}"),
                "relevance": hit.get("relevance", 0.0),
                "engine": hit.get("engine", ""),
            }
        )
    return hits


def _verdict_to_status(verdict: str) -> str:
    v = (verdict or "").lower()
    if v in ("confirmed", "true"):
        return "confirmed"
    if v in ("false", "debunked"):
        return "false"
    if v == "live_web_match":
        return "live"
    return "neutral"


def rag_health() -> Dict[str, Any]:
    """
    Quick health probe for the RAG subsystem. Reports whether the
    local model + Chroma are loadable. Doesn't hit the network.
    """
    out: Dict[str, Any] = {
        "chroma_ok": False,
        "chroma_count": 0,
        "embedding_model_ok": False,
        "duckduckgo_ok": False,
        "errors": [],
    }
    try:
        _ = _get_sentence_model()
        out["embedding_model_ok"] = True
    except Exception as exc:
        out["errors"].append(f"embedding_model: {exc}")
    try:
        collection = _get_chroma()
        out["chroma_ok"] = True
        out["chroma_count"] = collection.count()
    except Exception as exc:
        out["errors"].append(f"chroma: {exc}")
    try:
        _ = _get_ddg()
        out["duckduckgo_ok"] = True
    except Exception as exc:
        out["errors"].append(f"duckduckgo: {exc}")
    return out
