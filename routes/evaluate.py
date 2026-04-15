import io
import os

import pdfplumber
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from models.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    ExtractResumeResponse,
    RefineResumeRequest,
    RefineResumeResponse,
)
from services.gemini_service import evaluate_resume_against_job, refine_resume_with_feedback
from services.rate_limiter import InMemoryDailyRateLimiter

router = APIRouter(tags=["evaluation"])
MAX_PDF_SIZE_BYTES = 8 * 1024 * 1024
DAILY_EVALUATION_LIMIT = int(os.getenv("DAILY_EVALUATION_LIMIT", "2"))
RATE_LIMIT_BYPASS_IPS = {
    ip.strip()
    for ip in os.getenv("RATE_LIMIT_BYPASS_IPS", "127.0.0.1,::1,localhost").split(",")
    if ip.strip()
}

rate_limiter = InMemoryDailyRateLimiter(limit_per_day=DAILY_EVALUATION_LIMIT)


def _get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        first_ip = forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


@router.post("/extract-resume-text", response_model=ExtractResumeResponse)
async def extract_resume_text(file: UploadFile = File(...)) -> ExtractResumeResponse:
    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")
    if len(content) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="PDF is too large. Limit is 8 MB.")

    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text_chunks = [(page.extract_text() or "").strip() for page in pdf.pages]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Unable to read PDF content.") from exc

    resume_text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
    if len(resume_text) < 50:
        raise HTTPException(status_code=400, detail="Could not extract enough text from the PDF.")

    return ExtractResumeResponse(resume_text=resume_text)


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_resume(request: EvaluationRequest, raw_request: Request) -> EvaluationResponse:
    client_ip = _get_client_ip(raw_request)
    if client_ip not in RATE_LIMIT_BYPASS_IPS:
        rate_limiter.check_and_consume(client_ip)

    return await evaluate_resume_against_job(
        resume_text=request.resume_text,
        job_description=request.job_description,
    )


@router.post("/refine-resume", response_model=RefineResumeResponse)
async def refine_resume(request: RefineResumeRequest, raw_request: Request) -> RefineResumeResponse:
    client_ip = _get_client_ip(raw_request)
    if client_ip not in RATE_LIMIT_BYPASS_IPS:
        rate_limiter.check_and_consume(client_ip)

    ats_resume = await refine_resume_with_feedback(
        current_resume_text=request.current_resume_text,
        job_description=request.job_description,
        suggestions=request.suggestions,
        skill_gaps=request.skill_gaps,
        missing_keywords=request.missing_keywords,
        phrasing_improvements=request.phrasing_improvements,
    )
    return RefineResumeResponse(ats_resume=ats_resume)
