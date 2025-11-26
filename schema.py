from pydantic import BaseModel, Field
from typing import Optional, List

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


# class MatchResult(BaseModel):
#     """Full reasoning result for job ↔ CV comparison."""
#     extracted_requirements: List[Requirement]
#     candidate_profile: CandidateProfile
#     explicit_matches: List[ExplicitMatch]
#     obvious_gaps: List[ObviousGap]
#     inferred_matches: List[InferredMatch]

class MatchResult(BaseModel):
    explicit_matches: List[ExplicitMatch]
    obvious_gaps: List[ObviousGap]
    inferred_matches: List[InferredMatch]
