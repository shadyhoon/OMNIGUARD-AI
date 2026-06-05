"""
OmniGuard AI - Optional LLM augmentation
========================================
A thin, provider-agnostic wrapper around chat-completion APIs used
only as *optional enrichment* on top of the heuristic pipeline in
:mod:`utils.verifier`.

Design rules
------------
* The API key is **always** read at call time from the environment
  via :func:`os.getenv`. The key is never read from any file in this
  project, never written to disk by this module, never logged, and
  never included in error messages.
* If no key is set, every public function in this module returns
  ``None`` / an empty result, so callers can fall back to the
  heuristic output without special-casing.
* SDKs are imported lazily inside the functions that need them -
  the rest of the app runs without them.
* All network calls have a short timeout so a stuck request cannot
  freeze the Streamlit UI.
* Two providers are supported out of the box: **OpenAI** (paid, but
  most accurate) and **Google Gemini 1.5 Flash** (free tier, fast).
  The provider is auto-detected from the environment, but can be
  overridden explicitly via :func:`get_provider`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
LLM_TIMEOUT_SECONDS = 20

# Default models per provider. The OpenAI default is unchanged from
# the original code path so existing users see no regression.
DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}

# Provider preference order for auto-detect when no override is given.
# OpenAI wins if both keys happen to be set, preserving the old
# behaviour of users who already had OPENAI_API_KEY configured.
_AUTO_DETECT_ORDER: List[str] = ["openai", "gemini"]


# ---------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------
class LLMProvider(Protocol):
    """
    Minimal interface every chat-completion backend must implement.

    Returning a uniform ``(text, info)`` tuple lets the rest of the
    module stay provider-agnostic. ``info`` carries the same fields
    the UI expects to render as evidence of a real call.
    """

    name: str
    model: str

    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        ...


# ---------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------
class OpenAIProvider:
    """Chat Completions backend (gpt-4o-mini by default)."""

    def __init__(self, model: str = DEFAULT_MODELS["openai"]):
        self.name = "openai"
        self.model = model

    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        info: Dict[str, Any] = {
            "attempted": False,
            "succeeded": False,
            "model": self.model,
            "latency_ms": 0,
            "tokens_in": None,
            "tokens_out": None,
            "response_id": None,
            "error": None,
        }
        key = os.getenv(OPENAI_API_KEY_ENV)
        if not key:
            info["error"] = "no_api_key"
            return None, info
        try:
            import openai
        except ImportError as exc:
            info["error"] = f"openai_sdk_missing: {exc}"[:120]
            return None, info

        import time as _time
        started = _time.time()
        info["attempted"] = True
        try:
            client = openai.OpenAI(api_key=key, timeout=LLM_TIMEOUT_SECONDS)
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            info["latency_ms"] = int((_time.time() - started) * 1000)
            err = str(exc)
            info["error"] = err.split("\n")[0][:120] or type(exc).__name__
            return None, info

        info["latency_ms"] = int((_time.time() - started) * 1000)
        if not getattr(response, "choices", None):
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
        if getattr(response, "model", None):
            info["model"] = response.model
        return text, info


# ---------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------
class GeminiProvider:
    """
    Google Gemini chat backend.

    Uses the new ``google-genai`` SDK (the older
    ``google-generativeai`` package is deprecated). The free tier of
    ``gemini-1.5-flash`` allows 15 RPM / 1M tokens per day, which is
    more than enough for OmniGuard's per-claim summarisation.
    """

    def __init__(self, model: str = DEFAULT_MODELS["gemini"]):
        self.name = "gemini"
        self.model = model

    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        info: Dict[str, Any] = {
            "attempted": False,
            "succeeded": False,
            "model": self.model,
            "latency_ms": 0,
            "tokens_in": None,
            "tokens_out": None,
            "response_id": None,
            "error": None,
        }
        key = os.getenv(GEMINI_API_KEY_ENV)
        if not key:
            info["error"] = "no_api_key"
            return None, info
        try:
            from google import genai
        except ImportError as exc:
            info["error"] = f"google_genai_sdk_missing: {exc}"[:120]
            return None, info

        import time as _time
        started = _time.time()
        info["attempted"] = True
        try:
            client = genai.Client(api_key=key)
            gen_cfg: Dict[str, Any] = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                # Gemini uses a "response_mime_type" hint for JSON output.
                gen_cfg["response_mime_type"] = "application/json"
            response = client.models.generate_content(
                model=self.model,
                contents=user,
                config={
                    "system_instruction": system,
                    **gen_cfg,
                },
            )
        except Exception as exc:
            info["latency_ms"] = int((_time.time() - started) * 1000)
            err = str(exc)
            info["error"] = err.split("\n")[0][:120] or type(exc).__name__
            return None, info

        info["latency_ms"] = int((_time.time() - started) * 1000)
        text = getattr(response, "text", None)
        if not text or not text.strip():
            info["error"] = "empty_response"
            return None, info
        info["succeeded"] = True
        info["response_id"] = None  # Gemini doesn't echo a stable id we can rely on
        # Token usage - best-effort, may be missing on free tier.
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            info["tokens_in"] = getattr(um, "prompt_token_count", None)
            info["tokens_out"] = getattr(um, "candidates_token_count", None)
        return text.strip(), info


# ---------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------
_PROVIDER_BUILDERS: Dict[str, Callable[[], LLMProvider]] = {
    "openai": lambda: OpenAIProvider(model=os.getenv("OMNIGUARD_LLM_MODEL", DEFAULT_MODELS["openai"])),
    "gemini": lambda: GeminiProvider(model=os.getenv("OMNIGUARD_LLM_MODEL", DEFAULT_MODELS["gemini"])),
}


def _provider_available(name: str) -> bool:
    """True if the named provider's env key is set and SDK import would succeed."""
    if name == "openai":
        if not os.getenv(OPENAI_API_KEY_ENV):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True
    if name == "gemini":
        if not os.getenv(GEMINI_API_KEY_ENV):
            return False
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False
        return True
    return False


def get_provider(name: Optional[str] = None) -> Optional[LLMProvider]:
    """
    Return a provider instance, or ``None`` if no usable one is found.

    Parameters
    ----------
    name:
        Explicit provider name (``"openai"`` or ``"gemini"``).
        If ``None``, auto-detect from environment variables in
        ``_AUTO_DETECT_ORDER`` and return the first one whose
        required key and SDK are present.
    """
    if name:
        name = name.lower().strip()
        if name not in _PROVIDER_BUILDERS:
            return None
        if not _provider_available(name):
            return None
        return _PROVIDER_BUILDERS[name]()

    for candidate in _AUTO_DETECT_ORDER:
        if _provider_available(candidate):
            return _PROVIDER_BUILDERS[candidate]()
    return None


def available_providers() -> List[str]:
    """List of provider names that are usable right now."""
    return [n for n in _PROVIDER_BUILDERS if _provider_available(n)]


# ---------------------------------------------------------------------
# Public functions (preserved signatures)
# ---------------------------------------------------------------------
def is_llm_available() -> bool:
    """
    Return ``True`` if any provider has a key set and its SDK is
    importable. Cheap local check - does NOT contact the network.
    """
    return get_provider() is not None


def probe_llm_health(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Do a cheap round-trip to confirm the configured provider can
    actually serve requests. Returns ``{"ok", "reason", "checked",
    "provider", "model"}``. ``checked`` is True only if a real call
    was made.
    """
    provider = get_provider(name)
    if provider is None:
        return {
            "ok": False,
            "reason": "no key or SDK missing",
            "checked": False,
            "provider": None,
            "model": None,
        }

    # Use a tiny ping: a 5-token summarise on a benign claim.
    try:
        text, info = provider.chat(
            system="Reply with the single word: ok",
            user="ping",
            max_tokens=5,
            temperature=0.0,
        )
    except Exception as exc:
        return {
            "ok": False,
            "reason": (str(exc) or type(exc).__name__).split("\n")[0][:120],
            "checked": True,
            "provider": provider.name,
            "model": provider.model,
        }

    if info.get("succeeded") and text:
        return {
            "ok": True,
            "reason": "ok",
            "checked": True,
            "provider": provider.name,
            "model": provider.model,
        }
    return {
        "ok": False,
        "reason": info.get("error") or "ping_failed",
        "checked": info.get("attempted", False),
        "provider": provider.name,
        "model": provider.model,
    }


def llm_summarise_claim(
    claim: str,
    model: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Provider-agnostic version of the original helper."""
    provider = get_provider(provider_name)
    if provider is None:
        info: Dict[str, Any] = {
            "attempted": False,
            "succeeded": False,
            "model": None,
            "latency_ms": 0,
            "tokens_in": None,
            "tokens_out": None,
            "response_id": None,
            "error": "llm_unavailable",
        }
        return None, info

    # If the caller asked for a specific model, build a one-off provider.
    if model and model != provider.model:
        provider = _PROVIDER_BUILDERS[provider.name](model=model)  # type: ignore[assignment]

    system = (
        "You are a careful fact-checking assistant. Given a claim, "
        "respond in 2-3 sentences with a neutral, evidence-oriented "
        "summary. Flag obvious false claims and clearly state when "
        "you are uncertain. Do not invent sources."
    )
    return provider.chat(
        system=system,
        user=f"Claim: {claim.strip()}",
        max_tokens=200,
        temperature=0.2,
    )


def llm_flag_creative_leaps(
    text: str,
    model: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> Tuple[Optional[List[str]], Dict[str, Any]]:
    """Provider-agnostic creative-leap detector."""
    provider = get_provider(provider_name)
    if provider is None:
        info = {
            "attempted": False,
            "succeeded": False,
            "model": None,
            "latency_ms": 0,
            "tokens_in": None,
            "tokens_out": None,
            "response_id": None,
            "error": "llm_unavailable",
        }
        return None, info
    if model and model != provider.model:
        provider = _PROVIDER_BUILDERS[provider.name](model=model)  # type: ignore[assignment]

    if not text or len(text.strip()) < 20:
        info = {
            "attempted": False,
            "succeeded": False,
            "model": provider.model,
            "latency_ms": 0,
            "tokens_in": None,
            "tokens_out": None,
            "response_id": None,
            "error": "input_too_short",
        }
        return None, info

    system = (
        "You are a forensic text analyst. Identify specific sentences "
        "or phrases in the user's text that look like unsupported "
        "inferences, creative leaps, or hallucinated facts. Reply as "
        "a short JSON object with a single key 'leaps' whose value is "
        "a JSON array of strings (one observation each). No preamble."
    )
    user = (
        "Analyse the following text and return JSON "
        "{\"leaps\": [\"...\", \"...\"]}:\n\n"
        f"{text.strip()[:4000]}"
    )
    raw, info = provider.chat(
        system=system,
        user=user,
        max_tokens=400,
        temperature=0.0,
        json_mode=True,
    )
    if not info.get("succeeded") or not raw:
        return None, info

    import json
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        info["error"] = "json_parse_error"
        return None, info

    if isinstance(parsed, list):
        observations = [str(x) for x in parsed if str(x).strip()]
    elif isinstance(parsed, dict):
        # Prefer a 'leaps' key when present, then first list value.
        if isinstance(parsed.get("leaps"), list):
            observations = [str(x) for x in parsed["leaps"] if str(x).strip()]
        else:
            for value in parsed.values():
                if isinstance(value, list):
                    observations = [str(x) for x in value if str(x).strip()]
                    break
            else:
                observations = []
    else:
        observations = []
    return (observations or None), info


def llm_enrich_report(
    report: Dict[str, Any],
    model: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Augment a heuristic report from :func:`utils.verifier.analyze_content`
    with two optional LLM-derived fields:

    * ``llm_claim_summaries`` - per-claim neutral summaries
    * ``llm_creative_leaps``  - extra hallucination observations

    The original ``report`` is **not** mutated; a shallow copy with
    new keys is returned. If the LLM is unavailable, the new keys
    are set to ``None`` / ``[]`` so the UI can render the same shape.

    Parameters
    ----------
    model:
        Override the model name for this call (default: provider's
        default).
    provider_name:
        Force a specific provider (``"openai"`` or ``"gemini"``).
        ``None`` means auto-detect.
    """
    enriched: Dict[str, Any] = dict(report)
    enriched["llm_claim_summaries"] = []
    enriched["llm_creative_leaps"] = []
    enriched["llm_available"] = is_llm_available() if provider_name is None else (get_provider(provider_name) is not None)
    enriched["llm_provider"] = None
    enriched["llm_calls"] = []
    enriched["llm_tokens_in"] = 0
    enriched["llm_tokens_out"] = 0
    enriched["llm_latency_ms_total"] = 0
    enriched["llm_model"] = None

    provider = get_provider(provider_name)
    if provider is None:
        return enriched
    enriched["llm_provider"] = provider.name
    enriched["llm_model"] = provider.model if model is None else model

    # 1. Per-claim summaries from cross-reference hits.
    for hit in report.get("cross_reference_results", []) or []:
        claim = hit.get("claim")
        if not claim or claim.startswith("("):
            continue
        summary, info = llm_summarise_claim(claim, model=model, provider_name=provider_name)
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
            leaps, info = llm_flag_creative_leaps(text_to_scan, model=model, provider_name=provider_name)
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
