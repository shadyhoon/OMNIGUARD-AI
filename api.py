"""
OmniGuard AI - FastAPI backend
==============================
Exposes a small HTTP surface so the Streamlit dashboard (or any
other client) can hit the same verification pipeline over the
network instead of running it in-process.

Endpoints
---------
GET  /                 - tiny health page
GET  /health           - JSON health probe (heuristics + RAG + vision + LLM)
POST /analyze          - run the verification pipeline
                         body: { "content_type": "text"|"url"|"video",
                                 "user_input": "...",
                                 "use_rag": false,
                                 "enrich_with_llm": true }
POST /analyze/upload   - multipart upload for image/video/audio
GET  /docs             - Swagger UI (auto-generated)
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from utils.verifier import analyze_content
from utils.llm import is_llm_available, llm_enrich_report, probe_llm_health
from utils.rag import rag_health
from utils.vision import vision_health


app = FastAPI(
    title="OmniGuard AI API",
    description=(
        "Real-time multimodal verification. POST text, URLs, or "
        "uploaded media and receive a forensic-style report."
    ),
    version="0.2.0",
)


# ---------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    content_type: str = Field(..., pattern="^(text|url)$")
    user_input: str = Field(..., min_length=1)
    use_rag: bool = False
    enrich_with_llm: bool = True


class HealthResponse(BaseModel):
    status: str
    heuristics: bool
    rag: Dict[str, Any]
    vision: Dict[str, Any]
    llm: Dict[str, Any]


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "OmniGuard AI API",
        "version": app.version,
        "endpoints": ["/health", "/analyze", "/analyze/upload", "/docs"],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        heuristics=True,
        rag=rag_health(),
        vision=vision_health(),
        llm={
            "available": is_llm_available(),
            "probe": probe_llm_health() if is_llm_available() else None,
        },
    )


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> JSONResponse:
    """
    Run the verification pipeline on text or URL input.

    Returns the full report dict from
    :func:`utils.verifier.analyze_content`. If ``enrich_with_llm``
    is true and the OpenAI key is set, the report is augmented
    with LLM summaries and the per-call evidence ledger.
    """
    if req.content_type not in ("text", "url"):
        raise HTTPException(status_code=400, detail="content_type must be 'text' or 'url'")
    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input must not be empty")

    report = analyze_content(
        req.content_type,
        req.user_input.strip(),
        use_rag=req.use_rag,
    )
    if req.enrich_with_llm and is_llm_available():
        try:
            if req.content_type == "text":
                report["_raw_text"] = req.user_input
            report = llm_enrich_report(report)
        except Exception:
            pass
    return JSONResponse(report)


@app.post("/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    content_type: str = Form("image"),
    enrich_with_llm: bool = Form(True),
) -> JSONResponse:
    """
    Multipart upload for image / video / audio.

    The file is written to a temporary path, the existing
    :func:`analyze_content` pipeline runs on it, and the temp file
    is cleaned up at the end.
    """
    if content_type not in ("image", "video", "audio"):
        raise HTTPException(
            status_code=400, detail="content_type must be 'image', 'video' or 'audio'"
        )
    suffix = os.path.splitext(file.filename or "")[-1] or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = tmp.name
    try:
        report = analyze_content(content_type, tmp_path)
        if enrich_with_llm and is_llm_available():
            try:
                report = llm_enrich_report(report)
            except Exception:
                pass
        return JSONResponse(report)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
