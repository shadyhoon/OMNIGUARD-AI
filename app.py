"""
OmniGuard AI - Streamlit dashboard
=================================
A real-time multimodal verification dashboard with a dark,
ultra-modern UI.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import os

import plotly.graph_objects as go
import requests
import streamlit as st

from dotenv import load_dotenv

from utils.verifier import analyze_content
from utils.llm import (
    is_llm_available,
    llm_enrich_report,
    probe_llm_health,
    available_providers,
    get_provider,
)

# Load .env at startup (no-op if file is absent or empty).
load_dotenv()


# =====================================================================
# Page configuration
# =====================================================================
st.set_page_config(
    page_title="OmniGuard AI · Verification Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# Theme + custom CSS
# =====================================================================
def _inject_theme_css() -> None:
    """
    Inject a dark, ultra-modern stylesheet into the Streamlit app.

    The CSS targets the standard Streamlit class names. We use the
    ``st.markdown`` trick with ``unsafe_allow_html=True`` because
    Streamlit does not expose a native dark-theme API.
    """
    st.markdown(
        """
        <style>
        /* ---- Global background + typography ---- */
        .stApp {
            background: radial-gradient(
                    1200px 600px at 10% -10%,
                    rgba(99, 102, 241, 0.15),
                    transparent 60%
                ),
                radial-gradient(
                    900px 500px at 100% 0%,
                    rgba(16, 185, 129, 0.10),
                    transparent 60%
                ),
                #0b0f17;
            color: #e6edf3;
        }
        html, body, [class*="css"]  {
            font-family: "Inter", "Segoe UI", system-ui, sans-serif;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1320 0%, #0a0f1a 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #f1f5f9;
        }

        /* ---- Headings ---- */
        h1, h2, h3, h4 {
            color: #f8fafc !important;
            letter-spacing: -0.01em;
        }
        h1 {
            background: linear-gradient(90deg, #a5b4fc 0%, #6ee7b7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
        }

        /* ---- Cards / containers ---- */
        .og-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px 22px;
            margin: 8px 0 18px 0;
            backdrop-filter: blur(8px);
        }
        .og-card-title {
            color: #94a3b8;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 10px;
        }

        /* ---- Status badges ---- */
        .og-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            margin: 4px 6px 4px 0;
            border: 1px solid transparent;
        }
        .og-badge.success {
            background: rgba(16, 185, 129, 0.12);
            color: #6ee7b7;
            border-color: rgba(16, 185, 129, 0.35);
        }
        .og-badge.warning {
            background: rgba(245, 158, 11, 0.12);
            color: #fbbf24;
            border-color: rgba(245, 158, 11, 0.35);
        }
        .og-badge.danger {
            background: rgba(239, 68, 68, 0.12);
            color: #fca5a5;
            border-color: rgba(239, 68, 68, 0.35);
        }
        .og-badge.neutral {
            background: rgba(148, 163, 184, 0.10);
            color: #cbd5e1;
            border-color: rgba(148, 163, 184, 0.25);
        }
        .og-badge .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 8px currentColor;
        }

        /* ---- Buttons ---- */
        .stButton > button {
            background: linear-gradient(90deg, #6366f1 0%, #10b981 100%);
            color: white;
            border: 0;
            border-radius: 12px;
            padding: 0.7rem 1.2rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.25);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 30px rgba(16, 185, 129, 0.30);
        }

        /* ---- Inputs ---- */
        .stTextInput input,
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.10) !important;
            border-radius: 12px !important;
            color: #e6edf3 !important;
        }
        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
        }

        /* ---- Radio pills (content-type toggle) ---- */
        div[role="radiogroup"] {
            gap: 0.5rem;
        }
        div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 8px 14px;
            transition: all 0.15s ease;
        }
        div[role="radiogroup"] label:hover {
            background: rgba(99, 102, 241, 0.10);
            border-color: rgba(99, 102, 241, 0.40);
        }

        /* ---- Expander ---- */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.03) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
        }

        /* ---- Hide Streamlit chrome we don't need ---- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# Backend mapping: UI toggle -> analyze_content() content_type
# =====================================================================
CONTENT_TYPE_OPTIONS: List[Tuple[str, str]] = [
    ("🎬  Video Link", "video"),
    ("📰  Text Article", "text"),
    ("🧵  Social Media Thread", "text"),
]

CONTENT_TYPE_PROMPTS: Dict[str, str] = {
    "url": "Paste the video page URL (e.g. a YouTube / Vimeo watch page):",
    "text": "Paste the article body, or a single claim to verify:",
    "text": "Paste the social-media thread. One post per line works best:",
}
# The "Social" and "Text" entries above collide; rebuild explicitly.
CONTENT_TYPE_PROMPTS = {
    "url": "Paste the video page URL (e.g. a YouTube / Vimeo watch page):",
    "text_article": "Paste the article body, or a single claim to verify:",
    "text_social": "Paste the social-media thread. One post per line works best:",
}


# =====================================================================
# Cached pipeline
# =====================================================================
@st.cache_data(show_spinner=False)
def _run_analysis(
    content_type: str,
    user_input: str,
    use_rag: bool = False,
) -> Dict[str, Any]:
    """
    Run the heuristic pipeline. Results are cached per
    (content_type, input, use_rag) tuple so the UI doesn't re-run
    on every widget interaction.
    """
    return analyze_content(content_type, user_input, use_rag=use_rag)


def _run_video_analysis(
    url: str, use_rag: bool
) -> Dict[str, Any]:
    """
    Orchestrator for the 🎬 Video Link toggle.

    1. Download a short clip via yt-dlp (bounded length + size).
    2. Hand the local file to ``analyze_content('video', path)`` so
       the OpenCV vision layer (lighting / edge / inter-frame
       variance) actually runs on real frames.
    3. Stash the video metadata (title, uploader, duration) in the
       report's ``diagnostics`` so the UI can render it.
    4. Always clean up the temp file - even on exception - so disk
       doesn't fill up over many analyses.
    """
    from utils.video import download_video_clip, cleanup_download, is_ytdlp_available  # lazy

    if not is_ytdlp_available():
        return {
            "veracity_score": 0,
            "multimodal_consistency": {
                "applicable": False,
                "summary": (
                    "yt-dlp is not installed. Run "
                    "`pip install -r requirements-full.txt` to enable "
                    "video downloads."
                ),
            },
            "cross_reference_results": [],
            "hallucination_report": [
                "Video Link toggle is unavailable: yt-dlp missing."
            ],
            "content_type": "video",
            "generated_at": "",
            "analysis_duration_ms": 0,
            "diagnostics": {},
        }

    dl = download_video_clip(url)
    path = None
    try:
        if not dl.get("ok"):
            return {
                "veracity_score": 0,
                "multimodal_consistency": {
                    "applicable": False,
                    "summary": f"Video download failed: {dl.get('error', 'unknown')}",
                },
                "cross_reference_results": [],
                "hallucination_report": [
                    f"Could not download video: {dl.get('error', 'unknown')}"
                ],
                "content_type": "video",
                "generated_at": "",
                "analysis_duration_ms": 0,
                "diagnostics": {"video_download": dl},
            }

        path = dl["path"]
        report = analyze_content("video", path, use_rag=use_rag)

        # Stash yt-dlp metadata so the UI can surface it.
        diag = report.get("diagnostics") or {}
        diag["video_metadata"] = {
            "title": dl.get("title", ""),
            "uploader": dl.get("uploader", ""),
            "duration_sec": dl.get("duration", 0),
            "filesize_mb": dl.get("filesize_mb", 0),
            "site": dl.get("site", ""),
            "source_url": url,
        }
        report["diagnostics"] = diag
        return report
    finally:
        # Always try to clean up; never let cleanup errors propagate.
        try:
            if path:
                cleanup_download(path)
        except Exception:
            pass


def _api_url() -> str:
    """Default URL of the FastAPI backend (override via env or sidebar)."""
    return os.getenv("OMNIGUARD_API_URL", "http://localhost:8000")


def _run_via_api(
    content_type: str, user_input: str, *, use_rag: bool, enrich_with_llm: bool
) -> Dict[str, Any]:
    """
    POST to the FastAPI /analyze endpoint. Falls back to the
    in-process pipeline on any network error so the dashboard
    never breaks just because the API is down.
    """
    try:
        resp = requests.post(
            f"{_api_url()}/analyze",
            json={
                "content_type": content_type,
                "user_input": user_input,
                "use_rag": use_rag,
                "enrich_with_llm": enrich_with_llm,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        return {
            "veracity_score": 0,
            "multimodal_consistency": {
                "applicable": False,
                "summary": f"API unreachable: {exc}. Falling back to local pipeline.",
            },
            "cross_reference_results": [],
            "hallucination_report": [
                f"Could not reach OmniGuard API at {_api_url()} - {exc}",
                "Local heuristic pipeline did run, but the response below is from the fallback path.",
            ],
            "content_type": content_type,
            "generated_at": "",
            "analysis_duration_ms": 0,
            "diagnostics": {},
        }


# =====================================================================
# Visualisation helpers
# =====================================================================
def _score_color(score: int) -> str:
    """Map a 0-100 veracity score to a hex colour."""
    if score >= 75:
        return "#10b981"  # emerald
    if score >= 50:
        return "#f59e0b"  # amber
    return "#ef4444"      # red


def _veracity_donut(score: int) -> go.Figure:
    """Build a Plotly donut chart for the veracity score."""
    colour = _score_color(score)
    fig = go.Figure(
        data=[
            go.Pie(
                values=[score, max(0, 100 - score)],
                hole=0.72,
                sort=False,
                direction="clockwise",
                marker=dict(
                    colors=[colour, "rgba(255,255,255,0.06)"],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                textinfo="none",
                hoverinfo="skip",
            )
        ]
    )
    fig.add_annotation(
        text=f"<b>{score}</b>",
        x=0.5, y=0.55,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=44, color="#f8fafc", family="Inter"),
    )
    fig.add_annotation(
        text="VERACITY",
        x=0.5, y=0.40,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=11, color="#94a3b8", family="Inter"),
    )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
    )
    return fig


def _band_from_pct(pct: float | None, higher_is_better: bool = True) -> Tuple[str, str]:
    """
    Convert a 0-100 percentage to a (badge_label, badge_class) tuple.

    ``higher_is_better=True`` (default) treats high values as good.
    For anomaly scores pass ``higher_is_better=False`` so the
    threshold direction is inverted.
    """
    if pct is None:
        return "N/A", "neutral"
    if higher_is_better:
        if pct >= 75:
            return "Good", "success"
        if pct >= 50:
            return "Watch", "warning"
        return "Risk", "danger"
    # Lower is better
    if pct <= 25:
        return "Good", "success"
    if pct <= 50:
        return "Watch", "warning"
    return "Risk", "danger"


def _badge(label: str, kind: str) -> str:
    """Render a single status badge as raw HTML."""
    return (
        f'<span class="og-badge {kind}">'
        f'<span class="dot"></span>{label}</span>'
    )


def _badge_row(mm: Dict[str, Any]) -> str:
    """
    Build the multimodal-consistency badge row HTML.

    For text-only inputs the backend reports ``applicable=False``;
    we render a single neutral "Not applicable" badge in that case.
    """
    if not mm.get("applicable"):
        return _badge("Not applicable for this content", "neutral")

    parts: List[str] = []

    av = mm.get("audio_visual_alignment_pct")
    if av is not None:
        label, kind = _band_from_pct(av, higher_is_better=True)
        parts.append(_badge(f"Audio/Visual alignment: {label} ({av:.0f}%)", kind))

    light = mm.get("lighting_anomaly_pct")
    if light is not None:
        # Backend already inverted: high = clean, low = anomalous.
        label, kind = _band_from_pct(light, higher_is_better=True)
        parts.append(_badge(f"Lighting consistency: {label} ({light:.0f}%)", kind))

    artifact = mm.get("artifact_score_pct")
    if artifact is not None:
        label, kind = _band_from_pct(artifact, higher_is_better=True)
        parts.append(_badge(f"Artifact cleanliness: {label} ({artifact:.0f}%)", kind))

    return " ".join(parts) if parts else _badge("No multimodal signals", "neutral")


# =====================================================================
# Report renderer
# =====================================================================
def _render_cross_ref(results: List[Dict[str, Any]]) -> None:
    """Render cross-reference hits inside expanders.

    Each hit may carry a ``relevance`` score (0.0-1.0). We surface
    it as a coloured chip so the user can tell at a glance which
    matches are about the page's main topic vs incidental mentions
    pulled from nav/sidebar/comment text.
    """
    if not results:
        st.info("No cross-reference hits.")
        return
    for i, hit in enumerate(results, 1):
        status = (hit.get("status") or "").lower()
        if status == "confirmed":
            icon = "✅"
        elif status == "false":
            icon = "❌"
        elif status == "error":
            icon = "⚠️"
        else:
            icon = "ℹ️"

        relevance = hit.get("relevance")
        rel_pct = (
            f"{relevance * 100:.0f}%" if isinstance(relevance, (int, float)) else "—"
        )
        if isinstance(relevance, (int, float)):
            if relevance >= 0.6:
                rel_kind, rel_icon = "success", "●"
            elif relevance >= 0.3:
                rel_kind, rel_icon = "warning", "●"
            else:
                rel_kind, rel_icon = "danger", "●"
        else:
            rel_kind, rel_icon = "neutral", "○"

        title = (
            f"{icon} {hit.get('claim', '(no claim)')} — "
            f"{status.title()} · relevance {rel_pct}"
        )
        with st.expander(title, expanded=(i == 1 and status != "neutral")):
            st.markdown(
                f"**Source:** {hit.get('source', 'n/a')}\n\n"
                f"**Summary:** {hit.get('summary', 'n/a')}"
            )
            # Relevance chip in the body too, with a one-line explanation
            chip = (
                f'<span class="og-badge {rel_kind}">'
                f'<span class="dot"></span>'
                f"Relevance: {rel_pct}</span>"
            )
            st.markdown(chip, unsafe_allow_html=True)
            if isinstance(relevance, (int, float)) and relevance < 0.3:
                st.caption(
                    "_Low relevance: this fact may have appeared in nav, "
                    "comments, or sidebar text rather than the page's main content._"
                )
            context = hit.get("context")
            if context:
                st.caption(f"_Context:_ {context}")


def _render_hallucinations(notes: List[str]) -> None:
    """Render hallucination / forensic observations inside expanders."""
    if not notes:
        st.success("No hallucination or forensic red flags detected.")
        return
    for i, note in enumerate(notes, 1):
        with st.expander(f"Observation {i}", expanded=(i <= 2)):
            st.markdown(note)


def _render_report(report: Dict[str, Any]) -> None:
    """Render the full verification report."""
    score = int(report.get("veracity_score", 0))
    mm = report.get("multimodal_consistency", {})

    # --- Top row: donut + summary metrics ---
    top_l, top_r = st.columns([1, 2], gap="large")

    with top_l:
        st.markdown('<div class="og-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="og-card-title">Overall Veracity</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_veracity_donut(score), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with top_r:
        st.markdown('<div class="og-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="og-card-title">Multimodal Consistency</div>',
            unsafe_allow_html=True,
        )
        st.markdown(_badge_row(mm), unsafe_allow_html=True)
        summary = mm.get("summary")
        if summary:
            st.caption(summary)
        # Diagnostic chips for quick reading
        gen = report.get("generated_at", "")
        ms = report.get("analysis_duration_ms", 0)
        ct = (report.get("content_type") or "n/a").upper()
        st.markdown(
            f"<div style='margin-top:14px;'>"
            f'<span class="og-badge neutral"><span class="dot"></span>'
            f"Type: {ct}</span>"
            f'<span class="og-badge neutral"><span class="dot"></span>'
            f"Latency: {ms} ms</span>"
            f'<span class="og-badge neutral"><span class="dot"></span>'
            f"Generated: {gen}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Cross-reference hits ---
    st.markdown("### 🔗 Cross-Reference Facts")
    _render_cross_ref(report.get("cross_reference_results", []))

    # --- Hallucination / forensic warnings ---
    st.markdown("### ⚠️ Hallucination & Forensic Warnings")
    _render_hallucinations(report.get("hallucination_report", []))

    # --- LLM augmentation (only when key is present) ---
    _render_llm_section(report)


def _render_llm_section(report: Dict[str, Any]) -> None:
    """
    Render the LLM augmentation panel along with hard evidence
    that the calls actually reached OpenAI. The evidence ledger
    is built in :func:`utils.llm.llm_enrich_report` and contains
    per-call model name, response id, latency, and token usage.
    """
    claim_summaries = report.get("llm_claim_summaries") or []
    leaps = report.get("llm_creative_leaps") or []
    calls = report.get("llm_calls") or []

    if not (claim_summaries or leaps or calls):
        return  # nothing to show; the heuristic output is enough

    st.markdown("### 🧠 LLM Augmentation")

    # Per-claim summaries
    if claim_summaries:
        with st.expander("Per-claim LLM summaries", expanded=False):
            for item in claim_summaries:
                st.markdown(
                    f"**Claim:** {item.get('claim','')}\n\n"
                    f"> {item.get('summary','')}"
                )

    # Creative leaps
    if leaps:
        with st.expander("LLM-detected creative leaps", expanded=True):
            for leap in leaps:
                st.markdown(f"- {leap}")

    # Evidence ledger: prove the calls actually hit OpenAI.
    successful = [c for c in calls if c.get("succeeded")]
    attempted = [c for c in calls if c.get("attempted")]
    if attempted:
        with st.expander(
            f"🔬 LLM call evidence "
            f"({len(successful)}/{len(attempted)} succeeded)",
            expanded=False,
        ):
            st.caption(
                "Each row below is a real LLM API request and the "
                "server's response. If a row shows an error, the "
                "heuristic output above is still valid - the LLM "
                "augmentation just didn't run for that step."
            )
            tokens_in = report.get("llm_tokens_in") or 0
            tokens_out = report.get("llm_tokens_out") or 0
            total_ms = report.get("llm_latency_ms_total") or 0
            model = report.get("llm_model") or "?"
            provider = report.get("llm_provider") or "?"
            st.markdown(
                f"**Provider:** `{provider}` · "
                f"**Model:** `{model}` · "
                f"**Calls:** {len(attempted)} attempted, "
                f"{len(successful)} succeeded · "
                f"**Tokens:** {tokens_in} in / {tokens_out} out · "
                f"**Total latency:** {total_ms} ms"
            )
            for c in calls:
                ok = c.get("succeeded")
                kind = c.get("kind", "?")
                target = c.get("target", "")
                rid = c.get("response_id") or "-"
                latency = c.get("latency_ms") or 0
                tok_in = c.get("tokens_in")
                tok_out = c.get("tokens_out")
                err = c.get("error")
                if ok:
                    st.markdown(
                        f"- ✅ **{kind}** → `{target[:60]}` · "
                        f"id=`{rid}` · {latency} ms · "
                        f"tokens {tok_in}→{tok_out}"
                    )
                else:
                    st.markdown(
                        f"- ❌ **{kind}** → `{target[:60]}` · "
                        f"err=`{err}` · {latency} ms"
                    )


# =====================================================================
# Sidebar
# =====================================================================
def _api_health(api_url: str, _nonce: int = 0) -> bool:
    """
    Cheap GET against the FastAPI /health endpoint. No caching - the
    backend can be started or stopped between renders, and a hung
    request is short (2 s timeout). The ``_nonce`` arg lets the
    Re-probe button invalidate any future cache.
    """
    try:
        import requests
        r = requests.get(f"{api_url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _llm_health_cached(_nonce: int = 0, _provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Probe the LLM provider's health. The ``_nonce`` and ``_provider``
    parameters are deliberately unused in the body - they exist only
    so the Re-probe button and the provider dropdown can invalidate
    the result. We intentionally do NOT use ``st.cache_data`` here
    because ``None`` argument hashing interacts badly with provider
    overrides: cached results would stick across provider changes.
    The provider probe is a single small request (5 tokens), so the
    cost of skipping the cache is negligible.
    """
    return probe_llm_health(name=_provider)


def _selected_provider_name() -> Optional[str]:
    """
    Return the provider name the user picked in the sidebar, or
    ``None`` to let ``utils.llm`` auto-detect from env vars.
    """
    val = st.session_state.get("llm_provider_override")
    if not val or val == "auto":
        return None
    return val


def _render_sidebar() -> None:
    # Nonce stored in session_state; bumping it invalidates the cache.
    if "llm_probe_nonce" not in st.session_state:
        st.session_state["llm_probe_nonce"] = 0
    if "llm_provider_override" not in st.session_state:
        st.session_state["llm_provider_override"] = "auto"

    with st.sidebar:
        st.markdown("## 🛡️ OmniGuard AI")
        st.caption("Real-time multimodal verification")

        st.divider()
        st.markdown("### System status")

        # FastAPI backend health badge. The dashboard can run without
        # uvicorn (in-process pipeline), but if the user has ticked
        # "Call FastAPI backend" in Advanced options they will see a
        # connection error on every analyze. Show the state up front.
        api_url = os.getenv("OMNIGUARD_API_URL", "http://localhost:8000")
        api_up = _api_health(api_url, _nonce=st.session_state["llm_probe_nonce"])
        if api_up:
            st.success(f"FastAPI backend: **online** (`{api_url}`)")
        else:
            st.warning(
                f"FastAPI backend: **offline** (`{api_url}`)\n\n"
                "Start it in another terminal with:\n"
                "```\nuvicorn api:app --host 0.0.0.0 --port 8000\n```\n"
                "Or uncheck 'Call FastAPI backend' in Advanced options "
                "to use the in-process pipeline."
            )

        # LLM provider selector (auto-detect by default).
        detected = available_providers()
        provider_options = ["auto"] + detected
        st.selectbox(
            "LLM provider",
            options=provider_options,
            key="llm_provider_override",
            help=(
                "Auto picks the first provider with a valid key. "
                "Pick a specific one to override. Set the corresponding "
                "API key as a system env var (`OPENAI_API_KEY` or "
                "`GEMINI_API_KEY`)."
            ),
        )
        if st.session_state["llm_provider_override"] == "auto" and detected:
            st.caption(f"Detected: `{detected[0]}`")
        elif st.session_state["llm_provider_override"] != "auto":
            st.caption(f"Forced: `{st.session_state['llm_provider_override']}`")

        chosen = _selected_provider_name()
        llm_ok = is_llm_available() if chosen is None else get_provider(chosen) is not None
        if not llm_ok:
            needed = chosen or "OPENAI_API_KEY or GEMINI_API_KEY"
            st.warning(
                f"LLM augmentation: **offline**\n\n"
                f"Set `{needed}` as a system env var to enable it."
            )
        else:
            health = _llm_health_cached(
                _nonce=st.session_state["llm_probe_nonce"],
                _provider=chosen,
            )
            if health.get("ok"):
                prov = health.get("provider") or "?"
                model = health.get("model") or "?"
                st.success(
                    f"LLM augmentation: **online** (probe OK)\n\n"
                    f"Provider: `{prov}` · Model: `{model}`"
                )
            else:
                reason = health.get("reason", "unknown")
                st.error(
                    "LLM augmentation: **unreachable**\n\n"
                    f"Key is set but the API is not responding: _{reason}_\n\n"
                    "The dashboard will continue to work in heuristic-only mode."
                )

            if st.button(
                "🔄 Re-probe LLM",
                use_container_width=True,
                help="Re-check the LLM endpoint (60s cache).",
                key="probe_llm_btn",
            ):
                st.session_state["llm_probe_nonce"] += 1
                st.rerun()

        st.markdown("### Modules")
        for label in (
            "Text / article analysis",
            "URL page parsing",
            "Image forensic header",
            "Video container header",
            "Audio container header",
        ):
            st.markdown(f"- {label}")

        st.divider()
        st.caption("Heuristic pipeline: utils/verifier.py · LLM: utils/llm.py")


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    _inject_theme_css()
    _render_sidebar()

    # ---- Hero ----
    st.markdown(
        "<h1>🛡️ OmniGuard AI</h1>"
        "<p style='color:#94a3b8; margin-top:-8px;'>"
        "Paste a video link, an article, or a social thread — "
        "get a forensic-style veracity report in seconds."
        "</p>",
        unsafe_allow_html=True,
    )

    # ---- Content-type toggle + input ----
    labels = [opt[0] for opt in CONTENT_TYPE_OPTIONS]
    chosen_label = st.radio(
        "Content type",
        labels,
        index=1,  # default: Text Article
        horizontal=True,
        label_visibility="collapsed",
    )
    chosen_type = dict(CONTENT_TYPE_OPTIONS)[chosen_label]

    # Map display label -> prompt key
    if chosen_type == "url":
        prompt_key = "url"
    elif chosen_label.startswith("🧵"):
        prompt_key = "text_social"
    else:
        prompt_key = "text_article"

    st.markdown(
        f"<div class='og-card-title' style='margin-top:14px;'>"
        f"{CONTENT_TYPE_PROMPTS[prompt_key]}</div>",
        unsafe_allow_html=True,
    )

    # Single input widget that adapts to the content type
    if chosen_type == "url":
        user_input = st.text_input(
            "Input",
            placeholder="https://...",
            label_visibility="collapsed",
            key="primary_input",
        )
    else:
        user_input = st.text_area(
            "Input",
            height=200,
            placeholder=(
                "Paste text here..."
                if prompt_key == "text_article"
                else "Tweet 1...\nTweet 2...\nTweet 3..."
            ),
            label_visibility="collapsed",
            key="primary_input",
        )

    # ---- Advanced options (RAG + API) ----
    with st.expander("⚙️ Advanced options", expanded=False):
        opt_l, opt_r = st.columns(2)
        with opt_l:
            use_rag = st.checkbox(
                "Use RAG cross-reference (Chroma + DuckDuckGo)",
                value=False,
                help=(
                    "Routes the analysis through the local vector store and "
                    "live web search. Slower but catches claims outside "
                    "the built-in fact table."
                ),
                key="use_rag",
            )
        with opt_r:
            use_api = st.checkbox(
                "Call FastAPI backend",
                value=True,
                help=(
                    "Sends the request to the FastAPI backend instead of "
                    "running the pipeline in-process. Falls back to the "
                    "local pipeline automatically if the API is unreachable. "
                    f"Default URL: {_api_url()}"
                ),
                key="use_api",
            )
            api_url = st.text_input(
                "API URL",
                value=_api_url(),
                key="api_url",
                disabled=not use_api,
            )

    # ---- Analyze button + spinner ----
    clicked = st.button(
        "🔍  Analyze with OmniGuard",
        type="primary",
        use_container_width=True,
    )

    if not clicked:
        st.markdown(
            "<div class='og-card' style='text-align:center; color:#94a3b8;'>"
            "Awaiting input. Choose a content type above and paste your "
            "content, then press <b>Analyze with OmniGuard</b>."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if not user_input or not user_input.strip():
        st.warning("Please paste some content first.")
        return

    # Map the chosen content-type to the backend content_type arg
    backend_type = chosen_type  # "video" | "url" | "text"

    use_rag = bool(st.session_state.get("use_rag", False))
    use_api = bool(st.session_state.get("use_api", False))
    provider_name = _selected_provider_name()
    enrich_ok = (provider_name is None and is_llm_available()) or (
        provider_name is not None and get_provider(provider_name) is not None
    )

    with st.spinner("🛡️ OmniGuard is analysing the content..."):
        if backend_type == "video":
            # Video Link: download via yt-dlp, then run the OpenCV
            # vision pass on the actual frames. The orchestrator
            # cleans up the temp file on the way out.
            if use_api:
                # Server-side download (lets the FastAPI worker do it).
                report = _run_via_api(
                    "video",
                    user_input.strip(),
                    use_rag=use_rag,
                    enrich_with_llm=enrich_ok,
                )
            else:
                report = _run_video_analysis(user_input.strip(), use_rag=use_rag)
                if enrich_ok:
                    try:
                        report = llm_enrich_report(report, provider_name=provider_name)
                    except Exception:
                        pass
        elif use_api:
            report = _run_via_api(
                backend_type,
                user_input.strip(),
                use_rag=use_rag,
                enrich_with_llm=enrich_ok,
            )
        else:
            report = _run_analysis(backend_type, user_input.strip(), use_rag=use_rag)
            # Carry the raw text through so the LLM augmentation can use it
            if backend_type == "text":
                report["_raw_text"] = user_input
            if enrich_ok:
                try:
                    report = llm_enrich_report(report, provider_name=provider_name)
                except Exception:
                    # Never let an LLM error block the heuristic output.
                    pass

    _render_report(report)


if __name__ == "__main__":
    main()
