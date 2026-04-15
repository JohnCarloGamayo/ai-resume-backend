import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

CATEGORY_WEIGHTS = {
    "Skills Match": 0.30,
    "Experience Match": 0.25,
    "Tooling/Tech Stack Match": 0.20,
    "Domain/AI Relevance": 0.25,
}

ACTION_VERBS = {
    "built", "designed", "implemented", "developed", "optimized", "deployed", "integrated",
    "engineered", "improved", "scaled", "automated", "led", "managed", "delivered",
}

METRIC_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent|x|ms|s|sec|seconds|minutes|hours|k|m|b)\b", re.IGNORECASE)

SKILL_CANONICAL = {
    "python": {"python", "py"},
    "typescript": {"typescript", "ts"},
    "javascript": {"javascript", "js", "node", "nodejs"},
    "react": {"react", "reactjs"},
    "fastapi": {"fastapi"},
    "django": {"django"},
    "sql": {"sql", "postgresql", "mysql", "sqlite", "database"},
    "docker": {"docker", "container", "containerization"},
    "kubernetes": {"kubernetes", "k8s"},
    "cicd": {"cicd", "ci/cd", "pipeline", "github actions", "gitlab ci"},
    "aws": {"aws", "ec2", "s3", "lambda", "cloudwatch", "bedrock"},
    "gcp": {"gcp", "google cloud", "vertex ai"},
    "azure": {"azure", "azure openai"},
    "llm": {"llm", "large language model", "gpt", "transformer", "foundation model"},
    "rag": {"rag", "retrieval augmented generation", "retrieval-augmented generation"},
    "embedding": {"embedding", "vector search", "vector database", "semantic search"},
    "prompt engineering": {"prompt engineering", "prompt design", "prompt tuning"},
    "mlops": {"mlops", "model deployment", "model serving", "model monitoring"},
    "ocr": {"ocr", "optical character recognition", "document extraction"},
    "api": {"api", "rest", "graphql", "webhook"},
}

RELATED_SKILLS = {
    "llm": {"prompt engineering", "rag", "embedding"},
    "docker": {"kubernetes", "cicd"},
    "aws": {"gcp", "azure", "docker", "kubernetes"},
    "mlops": {"docker", "kubernetes", "cicd", "llm"},
    "rag": {"embedding", "llm"},
    "api": {"fastapi", "django", "react"},
}

DOMAIN_TERMS = {
    "ai", "ml", "machine learning", "artificial intelligence", "llm", "rag", "embedding",
    "prompt engineering", "model", "inference", "training", "computer vision", "nlp", "mlops",
}

HEADING_ALIASES = {
    "professional summary": "summary",
    "summary": "summary",
    "core competencies": "skills",
    "skills": "skills",
    "technical expertise": "skills",
    "professional experience": "experience",
    "experience": "experience",
    "projects": "projects",
    "education": "education",
}


@dataclass
class SectionedResume:
    summary: str
    skills: str
    experience: str
    projects: str
    education: str
    full_text: str



def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())



def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+/#.-]*", (text or "").lower())



def _split_resume_sections(resume_text: str) -> SectionedResume:
    lines = [line.strip() for line in (resume_text or "").replace("\r", "").split("\n")]
    sections: dict[str, list[str]] = {
        "summary": [],
        "skills": [],
        "experience": [],
        "projects": [],
        "education": [],
        "header": [],
    }

    current = "header"
    for raw in lines:
        if not raw:
            continue
        normalized = raw.lower().rstrip(":")
        if normalized in HEADING_ALIASES:
            current = HEADING_ALIASES[normalized]
            continue
        sections[current].append(raw)

    if not any(sections[key] for key in ("summary", "skills", "experience", "projects", "education")):
        # Fallback: use full text when no headings exist.
        payload = "\n".join(lines)
        return SectionedResume("", "", payload, "", "", payload)

    return SectionedResume(
        summary="\n".join(sections["summary"]),
        skills="\n".join(sections["skills"]),
        experience="\n".join(sections["experience"]),
        projects="\n".join(sections["projects"]),
        education="\n".join(sections["education"]),
        full_text="\n".join(lines),
    )



def _extract_required_canonicals(job_description: str) -> set[str]:
    jd = _clean_text(job_description)
    required: set[str] = set()
    for canonical, variants in SKILL_CANONICAL.items():
        if any(v in jd for v in variants):
            required.add(canonical)
    return required



def _evidence_strength(canonical: str, resume: SectionedResume) -> float:
    variants = SKILL_CANONICAL.get(canonical, {canonical})
    summary = _clean_text(resume.summary)
    skills = _clean_text(resume.skills)
    experience = _clean_text(resume.experience)
    projects = _clean_text(resume.projects)
    full_text = _clean_text(resume.full_text)

    in_experience = any(v in experience for v in variants)
    in_projects = any(v in projects for v in variants)
    in_skills = any(v in skills for v in variants)
    in_summary = any(v in summary for v in variants)

    if in_experience or in_projects:
        return 1.0
    if in_summary:
        return 0.85
    if in_skills:
        return 0.7

    related = RELATED_SKILLS.get(canonical, set())
    if related:
        for rel in related:
            rel_variants = SKILL_CANONICAL.get(rel, {rel})
            if any(v in full_text for v in rel_variants):
                return 0.45

    return 0.0



def _jaccard_similarity(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)



def _experience_complexity_score(resume: SectionedResume, job_description: str, required: set[str]) -> float:
    exp_text = f"{resume.experience}\n{resume.projects}".strip()
    if not exp_text:
        return 15.0

    lines = [line.strip().lower() for line in exp_text.split("\n") if line.strip()]
    action_hits = sum(1 for line in lines if any(verb in line for verb in ACTION_VERBS))
    action_component = min(1.0, action_hits / 8.0)

    complexity_markers = {
        "architecture", "scalable", "distributed", "pipeline", "optimization", "security", "reliability",
        "latency", "deployment", "integration", "monitoring", "migration", "automation",
    }
    exp_lower = _clean_text(exp_text)
    complexity_hits = sum(1 for marker in complexity_markers if marker in exp_lower)
    complexity_component = min(1.0, complexity_hits / 7.0)

    semantic_component = _jaccard_similarity(exp_text, job_description)

    evidence_values = [_evidence_strength(item, resume) for item in required] if required else []
    demonstrated = [value for value in evidence_values if value >= 0.45]
    demonstrated_breadth = min(1.0, len(demonstrated) / 6.0)
    demonstrated_strength = (sum(demonstrated) / len(demonstrated)) if demonstrated else 0.0
    responsibility_match = 0.55 * demonstrated_breadth + 0.45 * demonstrated_strength

    # Normalize intern-title penalty: penalize title only when responsibility evidence is weak.
    has_intern = "intern" in _clean_text(exp_text)
    responsibility_strength = 0.5 * complexity_component + 0.5 * responsibility_match
    internship_penalty = 0.0
    if has_intern and responsibility_strength < 0.35:
        internship_penalty = 0.06

    score = (
        0.35 * responsibility_match
        + 0.25 * complexity_component
        + 0.20 * action_component
        + 0.20 * semantic_component
    ) * 100

    # Prevent harsh under-scoring when substantial experience evidence exists.
    if action_component >= 0.45 and complexity_component >= 0.45:
        score = max(score, 74.0)
    elif action_component >= 0.30 and complexity_component >= 0.35:
        score = max(score, 65.0)
    elif responsibility_match >= 0.55:
        score = max(score, 60.0)

    score -= internship_penalty * 100
    return max(0.0, min(100.0, score))



def _tooling_score(required: set[str], resume: SectionedResume) -> float:
    tool_candidates = {"docker", "kubernetes", "cicd", "aws", "azure", "gcp", "api", "sql", "fastapi", "react", "django"}
    required_tools = [item for item in required if item in tool_candidates]

    if not required_tools:
        return 72.0

    total = 0.0
    for tool in required_tools:
        total += _evidence_strength(tool, resume)

    raw = (total / len(required_tools)) * 100

    # Do not overweight tooling.
    return max(0.0, min(100.0, raw))



def _domain_score(required: set[str], resume: SectionedResume, job_description: str) -> float:
    domain_required = [k for k in required if k in {"llm", "rag", "embedding", "prompt engineering", "mlops", "ocr"}]
    domain_text = f"{resume.summary}\n{resume.experience}\n{resume.projects}\n{resume.skills}"
    domain_lower = _clean_text(domain_text)
    jd_lower = _clean_text(job_description)

    resume_domain_hits = sum(1 for term in DOMAIN_TERMS if term in domain_lower)
    jd_domain_hits = sum(1 for term in DOMAIN_TERMS if term in jd_lower)
    baseline = 35.0

    if resume_domain_hits >= 2 and jd_domain_hits >= 2:
        baseline = 58.0
    elif resume_domain_hits >= 1 and jd_domain_hits >= 1:
        baseline = 50.0

    if not domain_required:
        semantic = _jaccard_similarity(domain_text, job_description)
        return max(0.0, min(100.0, baseline + semantic * 25.0))

    evidences = [_evidence_strength(item, resume) for item in domain_required]
    evidence_component = (sum(evidences) / len(evidences)) * 100
    semantic = _jaccard_similarity(domain_text, job_description) * 100

    score = 0.45 * evidence_component + 0.25 * semantic + 0.30 * baseline
    return max(0.0, min(100.0, score))



def _skills_score(required: set[str], resume: SectionedResume) -> tuple[float, list[str]]:
    if not required:
        return 75.0, []

    missing: list[str] = []
    total = 0.0

    for skill in sorted(required):
        evidence = _evidence_strength(skill, resume)
        total += evidence
        if evidence < 0.35:
            missing.append(skill)

    raw = (total / len(required)) * 100
    return max(0.0, min(100.0, raw)), missing



def _metrics_boost(resume: SectionedResume) -> float:
    text = f"{resume.experience}\n{resume.projects}"
    metric_hits = len(METRIC_PATTERN.findall(text))
    # Metrics are bonus only, never a hard penalty.
    return min(6.0, metric_hits * 1.5)



def _keyword_stuffing_penalty(resume_text: str, required: set[str]) -> float:
    tokens = _tokenize(resume_text)
    if len(tokens) < 50 or not required:
        return 0.0

    counts = Counter(tokens)
    repeated_required = 0
    for req in required:
        req_tokens = list(SKILL_CANONICAL.get(req, {req}))
        repeated_required += max(counts.get(token.lower(), 0) for token in req_tokens)

    unique_required_covered = sum(1 for req in required if any(v in _clean_text(resume_text) for v in SKILL_CANONICAL.get(req, {req})))
    coverage_ratio = unique_required_covered / max(len(required), 1)
    repetition_ratio = repeated_required / max(len(tokens), 1)

    if repetition_ratio > 0.08 and coverage_ratio < 0.65:
        return min(8.0, (repetition_ratio - 0.08) * 120)

    return 0.0



def _verdict(score: int) -> str:
    if score >= 80:
        return "STRONG MATCH"
    if score >= 60:
        return "PARTIAL MATCH"
    return "WEAK MATCH"



def _reason(category: str, score: float, missing: list[str]) -> str:
    if category == "Skills Match":
        if missing:
            return f"Strong partial alignment with semantic credit; lower confidence for: {', '.join(missing[:4])}."
        return "High alignment with direct and project-level evidence for required skills."

    if category == "Experience Match":
        return "Scored from responsibility complexity, action-oriented delivery, and semantic alignment with role expectations."

    if category == "Tooling/Tech Stack Match":
        return "Tooling scored with partial credit for related platforms; tooling weight is moderated to avoid over-penalization."

    return "Domain score reflects AI/ML signal coverage with semantic partial matches for related concepts."



def compute_evidence_based_score(resume_text: str, job_description: str) -> dict[str, Any]:
    resume = _split_resume_sections(resume_text)
    required = _extract_required_canonicals(job_description)

    skills, missing_keywords = _skills_score(required, resume)
    experience = _experience_complexity_score(resume, job_description, required)
    tooling = _tooling_score(required, resume)
    domain = _domain_score(required, resume, job_description)

    metric_bonus = _metrics_boost(resume)
    stuffing_penalty = _keyword_stuffing_penalty(resume.full_text, required)

    coverage_ratio = 0.0
    if required:
        coverage_ratio = (len(required) - len(missing_keywords)) / len(required)

    high_alignment_boost = 0.0
    if stuffing_penalty <= 0.1:
        if skills >= 80 and experience >= 70 and domain >= 55:
            high_alignment_boost += 4.0
        if coverage_ratio >= 0.78 and tooling >= 75:
            high_alignment_boost += 2.0

    high_alignment_boost = min(6.0, high_alignment_boost)

    weighted = (
        skills * CATEGORY_WEIGHTS["Skills Match"]
        + experience * CATEGORY_WEIGHTS["Experience Match"]
        + tooling * CATEGORY_WEIGHTS["Tooling/Tech Stack Match"]
        + domain * CATEGORY_WEIGHTS["Domain/AI Relevance"]
    )

    weighted = weighted + metric_bonus + high_alignment_boost - stuffing_penalty
    weighted = max(0.0, min(100.0, weighted))

    score_int = int(round(weighted))

    breakdown = [
        {
            "category": "Skills Match",
            "score": int(round(skills)),
            "weight": CATEGORY_WEIGHTS["Skills Match"],
            "reasoning": _reason("Skills Match", skills, missing_keywords),
        },
        {
            "category": "Experience Match",
            "score": int(round(experience)),
            "weight": CATEGORY_WEIGHTS["Experience Match"],
            "reasoning": _reason("Experience Match", experience, missing_keywords),
        },
        {
            "category": "Tooling/Tech Stack Match",
            "score": int(round(tooling)),
            "weight": CATEGORY_WEIGHTS["Tooling/Tech Stack Match"],
            "reasoning": _reason("Tooling/Tech Stack Match", tooling, missing_keywords),
        },
        {
            "category": "Domain/AI Relevance",
            "score": int(round(domain)),
            "weight": CATEGORY_WEIGHTS["Domain/AI Relevance"],
            "reasoning": _reason("Domain/AI Relevance", domain, missing_keywords),
        },
    ]

    formula = (
        f"Weighted score = ({int(round(skills))} x 0.30) + ({int(round(experience))} x 0.25) + "
        f"({int(round(tooling))} x 0.20) + ({int(round(domain))} x 0.25) + bonus({metric_bonus:.1f}) + align_boost({high_alignment_boost:.1f}) - penalty({stuffing_penalty:.1f}) = {weighted:.2f}"
    )

    return {
        "match_score": score_int,
        "match_verdict": _verdict(score_int),
        "is_qualified": score_int >= 70,
        "score_breakdown": breakdown,
        "normalized_scoring": formula,
        "missing_keywords": [item for item in missing_keywords if item][:15],
        "metric_bonus": metric_bonus,
        "high_alignment_boost": high_alignment_boost,
        "keyword_stuffing_penalty": stuffing_penalty,
    }
