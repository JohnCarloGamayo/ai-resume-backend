from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EvaluationRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, max_length=120000)
    job_description: str = Field(..., min_length=50, max_length=120000)

    @field_validator("resume_text", "job_description")
    @classmethod
    def validate_non_empty_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Content cannot be empty.")
        return normalized


class EvaluationResponse(BaseModel):
    match_score: int = Field(..., ge=0, le=100)
    match_verdict: Literal["STRONG MATCH", "PARTIAL MATCH", "WEAK MATCH"]
    is_qualified: bool
    strengths: list[str]
    skill_gaps: list[str]
    suggestions: list[str]
    executive_summary: str
    score_breakdown: list[dict]
    normalized_scoring: str
    critical_gaps: list[str]
    moderate_gaps: list[str]
    optional_gaps: list[str]
    missing_keywords: list[str]
    phrasing_improvements: list[str]
    ats_resume: str | None


class ExtractResumeResponse(BaseModel):
    resume_text: str


class RefineResumeRequest(BaseModel):
    current_resume_text: str = Field(..., min_length=50, max_length=120000)
    job_description: str = Field(..., min_length=50, max_length=120000)
    suggestions: list[str] = Field(default_factory=list)
    skill_gaps: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    phrasing_improvements: list[str] = Field(default_factory=list)

    @field_validator("current_resume_text", "job_description")
    @classmethod
    def validate_refine_non_empty_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Content cannot be empty.")
        return normalized


class RefineResumeResponse(BaseModel):
    ats_resume: str
