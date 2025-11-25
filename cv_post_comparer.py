import os
import json
from typing import Optional, List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ---------- 1) Soft, robust schema ----------

class Requirement(BaseModel):
    name: str
    # Soft constraints: we *suggest* values but don't enforce them
    priority: str = Field(
        description=(
            "How critical this requirement is for the role. "
            "Recommended values: 'must_have', 'nice_to_have', 'optional'."
        )
    )
    category: str = Field(
        default="other",
        description=(
            "Category of the requirement. Recommended values: "
            "'skill', 'soft_skill', 'experience', 'domain_knowledge', "
            "'tool', 'other', etc."
        )
    )
    details: Optional[str] = Field(
        default=None,
        description="Free-text explanation of what this requirement means in context."
    )


class ExplicitMatch(BaseModel):
    """Requirements clearly and explicitly supported by the CV."""
    name: str
    evidence: str = Field(
        description="Direct quote or summary from the CV that supports this match."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence that this is a solid, explicit match."
    )


class ObviousGap(BaseModel):
    """Requirements that are clearly missing or unlikely based on the CV."""
    name: str
    reason: str = Field(
        description="Why this is considered missing/unlikely (based on job vs CV)."
    )
    severity: str = Field(
        description="How critical this gap is for the role. "
                    "Recommended: 'low', 'medium', 'high'."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence that this is truly missing/unlikely."
    )


class InferredMatch(BaseModel):
    """Requirements that are not explicitly written but are likely present."""
    name: str
    inference_basis: str = Field(
        description=(
            "Explanation of how this skill/quality is inferred from education, "
            "frameworks, tools, domains or context in the CV."
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence that the candidate actually has this."
    )


class CandidateProfile(BaseModel):
    """Structured view of the candidate derived from the CV."""
    headline: Optional[str] = None
    main_roles: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    tools_and_tech: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    education_summary: Optional[str] = None
    experience_summary: Optional[str] = None
    other_signals: Optional[List[str]] = None


class MatchResult(BaseModel):
    """Full reasoning result for job ↔ CV comparison."""
    extracted_requirements: List[Requirement]
    candidate_profile: CandidateProfile
    explicit_matches: List[ExplicitMatch]
    obvious_gaps: List[ObviousGap]
    inferred_matches: List[InferredMatch]


# ---------- 2) LLM chain (robust prompt) ----------

parser = PydanticOutputParser(pydantic_object=MatchResult)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a senior technical recruiter and hiring manager with 15+ years "
                "experience in software, data, and machine learning roles.\n\n"
                "The job and CV JSON structures can vary a lot: different field names, "
                "free-text descriptions, nested sections, etc. Your job is to:\n"
                "1) Carefully understand the job description JSON (title, text, bullet points, "
                "requirements, responsibilities, etc.).\n"
                "2) Carefully understand the candidate CV JSON (summary, roles, tech stack, "
                "projects, education, etc.).\n"
                "3) Focus on the MOST IMPORTANT information only: core responsibilities, "
                "must-have skills, seniority, main tech stack, domains, leadership.\n"
                "   Ignore noise like hobbies unless they clearly support the role.\n"
                "4) Build a structured candidate_profile from whatever useful signals you find.\n"
                "5) Extract a clear list of requirements from the job (extracted_requirements):\n"
                "   - For each requirement, set:\n"
                "     * name (short requirement name)\n"
                "     * priority (e.g. 'must_have', 'nice_to_have', 'optional')\n"
                "     * category (e.g. 'skill', 'soft_skill', 'experience', "
                "'domain_knowledge', 'tool', 'other')\n"
                "     * details (short explanation in context)\n"
                "6) For each requirement, classify it into exactly one bucket:\n"
                "   - explicit_matches: clearly and explicitly present in the CV.\n"
                "   - obvious_gaps: clearly missing or unlikely from the CV.\n"
                "   - inferred_matches: not explicitly written, but strongly suggested by "
                "     education, frameworks, domains, tools, or very typical background.\n\n"
                "Reasoning rules:\n"
                "- Be strict for explicit_matches: require clear textual evidence in the CV.\n"
                "- For inferred_matches, only infer when there is strong contextual support.\n"
                "- Use obvious_gaps for important requirements that are absent or unlikely.\n"
                "- Always provide a confidence score in [0,1] for each item.\n\n"
                "Return ONLY JSON that matches the given schema.\n"
                "{format_instructions}"
            ),
        ),
        (
            "user",
            "Job description JSON:\n{job_json}\n\nCandidate CV JSON:\n{cv_json}"
        ),
    ]
)

llm = ChatOpenAI(
    model="gpt-5-nano",   # or "gpt-4.1" for stronger reasoning       # more deterministic, recruiter-like
)

chain = prompt | llm | parser


# ---------- 3) Public helpers ----------

def match_job_and_cv_from_dicts(job_json: dict, cv_json: dict) -> MatchResult:
    return chain.invoke(
        {
            "job_json": job_json,
            "cv_json": cv_json,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def match_job_and_cv_from_files(job_path: str, cv_path: str) -> MatchResult:
    with open(job_path, "r", encoding="utf-8") as f:
        job_json = json.load(f)
    with open(cv_path, "r", encoding="utf-8") as f:
        cv_json = json.load(f)
    return match_job_and_cv_from_dicts(job_json, cv_json)


# ---------- 4) CLI / quick test ----------

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Please set the OPENAI_API_KEY environment variable before running."
        )

    result = match_job_and_cv_from_files("job.json", "cv.json")
    print(result.model_dump_json(indent=2))

