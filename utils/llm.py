"""
OmniGuard AI - Optional LLM augmentation
========================================
A thin wrapper around the OpenAI Chat Completions API that is used
only as an *optional enrichment* on top of the heuristic pipeline
in :mod:`utils.verifier`.

Design rules
------------
* The API key is **always** read at call time from the environment
  via :func:`os.getenv`. The key is never read from any file in this
  project, never written to disk by this module, never logged, and
  never included in error messages.
* If no key is set, every public function in this module returns
  ``None`` / an empty result, so callers can fall back to the
  heuristic output without special-casing.
* The module uses the ``openai`` Python SDK if it is installed. To
  keep the dependency footprint light, the SDK is imported lazily
  inside the function that needs it - the rest of the app runs
  without it.
* All network calls have a short timeout so a stuck request cannot
  freeze the Streamlit UI.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Public constant so the UI can show a friendly "LLM augmentation
# unavailable" hint without importing os itself.
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
LLM_TIMEOUT_SECONDS = 20


def is_llm_available() -> bool:
    """
    Return ``True`` only if an OpenAI API key is present in the
    environment *and* the ``openai`` package can be imported.

    This is a *cheap* local check: it does NOT contact OpenAI. Use
    :func:`probe_llm_health` for a real round-trip check that
    distinguishes "key present but quota exhausted" from "key works".

    This function never logs, prints, or returns the key value.
    """
    key = os.getenv(OPENAI_API_KEY_ENV)
    if not key:
        return False
    try:
        import openai  # noqa: F401  (imported for side-effect: availability check)
    except ImportError:
        return False
    return True


def probe_llm_health() -> Dict[str, Any]:
    """
    Do a cheap round-trip to OpenAI to confirm the configured key
    can actually serve requests. This is more expensive than
    :func:`is_llm_available` and should be cached in the UI.

    Returns
    -------
    dict with keys:
        * ``ok``       - True if the probe succeeded
        * ``reason``   - short human-readable status
        * ``checked``  - True if a real API call was made
    """
    if not is_llm_available():
        return {"ok": False, "reason": "no key or SDK missing", "checked": False}
    try:
        client = _get_client()
        # List models is the cheapest authenticated call. Even with
        # an exhausted quota it returns 200, so this primarily
        # detects auth/permission failures, not billing. The real
        # billing/quota check is implicit: the per-call error path
        # in ``llm_summarise_claim`` etc. returns None on 429.
        client.models.list()
        return {"ok": True, "reason": "ok", "checked": True}
    except Exception as exc:
        # Never include the key in the reason string.
        reason = (str(exc) or type(exc).__name__).split("\n")[0][:120]
        return {"ok": False, "reason": reason, "checked": True}


def _get_client():
    """
    Build a configured OpenAI client.

    Raises
    ------
    RuntimeError
        If no API key is configured or the SDK is not installed.
    """
    key = os.getenv(OPENAI_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{OPENAI_API_KEY_ENV} is not set in the environment."
        )
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is not installed. "
            "Run: pip install openai"
        ) from exc

    return openai.OpenAI(api_key=key, timeout=LLM_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------
def llm_summarise_claim(
    claim: str, model: str = "gpt-4o-mini"
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Ask the LLM to produce a short, neutral, evidence-oriented
    summary of a single claim.

    Returns a ``(summary, call_info)`` tuple. ``call_info`` always
    contains keys the UI can use as proof of the call:

        * ``attempted``  - True if we tried to make the call
        * ``succeeded``  - True if the API returned content
        * ``model``      - model name (echoed by the server)
        * ``latency_ms`` - round-trip time in milliseconds
        * ``tokens_in``  - prompt tokens (if returned)
        * ``tokens_out`` - completion tokens (if returned)
        * ``response_id``- OpenAI response id (if returned)
        * ``error``      - short error label on failure

    The summary is ``None`` if the LLM is unavailable, the request
    fails, or the response is empty.
    """
    info: Dict[str, Any] = {
        "attempted": False,
        "succeeded": False,
        "model": model,
        "latency_ms": 0,
        "tokens_in": None,
        "tokens_out": None,
        "response_id": None,
        "error": None,
    }
    if not is_llm_available():
        info["error"] = "llm_unavailable"
        return None, info

    system_prompt = (
        "You are a careful fact-checking assistant. Given a claim, "
        "respond in 2-3 sentences with a neutral, evidence-oriented "
        "summary. Flag obvious false claims and clearly state when "
        "you are uncertain. Do not invent sources."
    )
    user_prompt = f"Claim: {claim.strip()}"

    import time
    started = time.time()
    info["attempted"] = True
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
    except Exception as exc:  # broad: any SDK or network error
        info["latency_ms"] = int((time.time() - started) * 1000)
        # Capture only the error code (e.g. "429", "insufficient_quota")
        # so we never accidentally log the API key.
        err = str(exc)
        info["error"] = err.split("\n")[0][:120] or type(exc).__name__
        return None, info

    info["latency_ms"] = int((time.time() - started) * 1000)

    if not response.choices:
        info["error"] = "no_choices"
        return None, info
    text = (response.choices[0].message.content or "").strip()
    if not text:
        info["error"] = "empty_response"
        return None, info

    info["succeeded"] = True
    info["response_id"] = getattr(response, "id", None)
    usage = getattr(response, "usage", None)
    if usage is not None:
        info["tokens_in"] = getattr(usage, "prompt_tokens", None)
        info["tokens_out"] = getattr(usage, "completion_tokens", None)
    # Echo back the actual model the server used (may differ from request)
    if getattr(response, "model", None):
        info["model"] = response.model
    return text, info


def llm_flag_creative_leaps(
    text: str, model: str = "gpt-4o-mini"
) -> Tuple[Optional[List[str]], Dict[str, Any]]:
    """
    Ask the LLM to identify potential 'creative leaps' / unsupported
    inferences in a piece of text.

    Returns ``(observations, call_info)``. See :func:`llm_summarise_claim`
    for the ``call_info`` schema. Returns ``(None, info)`` when the
    LLM is unavailable, the text is too short, or the request fails.
    """
    info: Dict[str, Any] = {
        "attempted": False,
        "succeeded": False,
        "model": model,
        "latency_ms": 0,
        "tokens_in": None,
        "tokens_out": None,
        "response_id": None,
        "error": None,
    }
    if not is_llm_available():
        info["error"] = "llm_unavailable"
        return None, info
    if not text or len(text.strip()) < 20:
        info["error"] = "input_too_short"
        return None, info

    system_prompt = (
        "You are a forensic text analyst. Identify specific sentences "
        "or phrases in the user's text that look like unsupported "
        "inferences, creative leaps, or hallucinated facts. Reply as "
        "a short JSON array of strings only. Each string should be "
        "one observation, no preamble."
    )
    user_prompt = (
        "Analyse the following text and return a JSON array of "
        "creative-leap observations:\n\n"
        f"{text.strip()[:4000]}"
    )

    import time
    started = time.time()
    info["attempted"] = True
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        info["latency_ms"] = int((time.time() - started) * 1000)
        err = str(exc)
        info["error"] = err.split("\n")[0][:120] or type(exc).__name__
        return None, info

    info["latency_ms"] = int((time.time() - started) * 1000)

    if not response.choices:
        info["error"] = "no_choices"
        return None, info
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        info["error"] = "empty_response"
        return None, info

    # Parse the JSON object. The model is asked to return an object
    # containing a list, but we accept a bare list as a fallback.
    import json

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        info["error"] = "json_parse_error"
        return None, info

    if isinstance(parsed, list):
        observations = [str(x) for x in parsed if str(x).strip()]
    elif isinstance(parsed, dict):
        # Heuristic: pick the first list-valued field.
        for value in parsed.values():
            if isinstance(value, list):
                observations = [str(x) for x in value if str(x).strip()]
                break
        else:
            observations = []
    else:
        observations = []

    info["succeeded"] = True
    info["response_id"] = getattr(response, "id", None)
    usage = getattr(response, "usage", None)
    if usage is not None:
        info["tokens_in"] = getattr(usage, "prompt_tokens", None)
        info["tokens_out"] = getattr(usage, "completion_tokens", None)
    if getattr(response, "model", None):
        info["model"] = response.model
    return (observations or None), info


def llm_enrich_report(
    report: Dict[str, Any],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Augment a heuristic report from :func:`utils.verifier.analyze_content`
    with two optional LLM-derived fields:

    * ``llm_claim_summaries`` - per-claim neutral summaries
    * ``llm_creative_leaps``  - extra hallucination observations

    The original ``report`` is **not** mutated; a shallow copy with
    new keys is returned. If the LLM is unavailable, the new keys
    are set to ``None`` / ``[]`` so the UI can render the same shape.
    """
    enriched: Dict[str, Any] = dict(report)
    enriched["llm_claim_summaries"] = []
    enriched["llm_creative_leaps"] = []
    enriched["llm_available"] = is_llm_available()
    # Evidence-of-call ledger. Each entry proves one real OpenAI
    # request was attempted and what the server returned.
    enriched["llm_calls"] = []
    enriched["llm_tokens_in"] = 0
    enriched["llm_tokens_out"] = 0
    enriched["llm_latency_ms_total"] = 0
    enriched["llm_model"] = model

    if not is_llm_available():
        return enriched

    # 1. Per-claim summaries from cross-reference hits.
    for hit in report.get("cross_reference_results", []) or []:
        claim = hit.get("claim")
        if not claim or claim.startswith("("):
            continue
        summary, info = llm_summarise_claim(claim, model=model)
        info_with_kind = dict(info)
        info_with_kind["kind"] = "claim_summary"
        info_with_kind["target"] = claim
        enriched["llm_calls"].append(info_with_kind)
        if info.get("succeeded"):
            enriched["llm_tokens_in"] += info.get("tokens_in") or 0
            enriched["llm_tokens_out"] += info.get("tokens_out") or 0
            enriched["llm_latency_ms_total"] += info.get("latency_ms") or 0
        if summary:
            enriched["llm_claim_summaries"].append(
                {"claim": claim, "summary": summary}
            )

    # 2. Creative-leap detection for text content only.
    if report.get("content_type") == "text":
        diagnostics = report.get("diagnostics", {}) or {}
        text_to_scan = enriched.get("_raw_text") or ""
        if not text_to_scan:
            text_to_scan = (diagnostics.get("input_summary", {}) or {}).get(
                "url", ""
            ) or ""
        if text_to_scan:
            leaps, info = llm_flag_creative_leaps(text_to_scan, model=model)
            info_with_kind = dict(info)
            info_with_kind["kind"] = "creative_leaps"
            info_with_kind["target"] = "<submitted text>"
            enriched["llm_calls"].append(info_with_kind)
            if info.get("succeeded"):
                enriched["llm_tokens_in"] += info.get("tokens_in") or 0
                enriched["llm_tokens_out"] += info.get("tokens_out") or 0
                enriched["llm_latency_ms_total"] += info.get("latency_ms") or 0
            if leaps:
                enriched["llm_creative_leaps"] = leaps

    return enriched
