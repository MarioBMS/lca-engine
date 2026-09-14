"""Recruiting assistant agent.

A deep agent (built with ``deepagents.create_deep_agent``) with
seven tools - lookup_job_posting, build_candidate_profile, get_candidate, 
get_current_recruiter, send_candidate_email, score_candidate, and 
add_candidate_skill. The tools call the data-access layer in ``data_service`` for 
storage and retrieval.

Configure credentials via environment variables or a .env file
(OPENAI_API_KEY, and optionally LANGSMITH_API_KEY / LANGSMITH_PROJECT for
tracing), then call run_agent(...) with a recruiter request.

Install:
    uv add deepagents langchain langgraph langchain-openai langsmith python-dotenv
"""

import json
import os
import random
import re
import uuid

from dotenv import load_dotenv
load_dotenv(override=True)

# Enable LangSmith tracing; project / API key come from the environment or .env.
os.environ.setdefault("LANGSMITH_TRACING", "true")

from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from . import data_service
from .data_service import RECRUITER_IDS

MODEL_NAME = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Job posting schema
# ---------------------------------------------------------------------------
class SalaryRange(BaseModel):
    min: int
    max: int
    currency: str


class JobPosting(BaseModel):
    job_id: str
    title: str
    department: str
    location: str
    employment_type: str
    seniority: str
    salary_range: SalaryRange
    required_skills: list[str]
    min_years_experience: int
    description: str
    posted_date: str
    status: str

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def lookup_job_posting(job_id: str) -> dict:
    "Look up a job posting by job_id (e.g. 'JOB-10001'). Returns the posting and a found flag."
    record = data_service.get_job_posting(job_id)
    if record is None:
        return {"job_posting": None, "found": False}
    return {"job_posting": record, "found": True}


@tool
def build_candidate_profile(candidate_id: str) -> dict:
    "Assemble a full candidate profile (work history, education, skills) and store it. Returns the profile and a found flag."
    existing = data_service.get_profile_from_db(candidate_id)["candidate_profile"]
    if existing is not None:
        return {"candidate_profile": existing, "found": True}
    rec = data_service.get_candidate_record(candidate_id)
    if rec is None:
        return {"candidate_profile": None, "found": False}
    built = {
        "candidate_id": candidate_id,
        "name": rec["name"],
        "work_history": data_service.fetch_work_history(candidate_id),
        "education": data_service.fetch_education(candidate_id),
        "skills": data_service.fetch_skills(candidate_id),
        "years_experience": rec["years_experience"],
    }
    data_service.save_profile_to_db(candidate_id, built)
    return {"candidate_profile": built, "found": True}


SCORING_PROMPT = (
    "You are a recruiting assistant. Score the candidate against the job from "
    "1 to 100 based on how good a fit they are, weighing their experience and "
    "skills. In your justification, explicitly list which of the job's required "
    "skills the candidate has and which required skills they are missing, naming "
    "each one. Any missing required skill must lower the skills_match component and "
    "the overall score. Return a score and a justification that reflects your "
    "overall assessment of this candidate's fit."
)

from typing import Literal

class RubricBreakdown(BaseModel):
    experience: float
    skills_match: float
    seniority_fit: float
    component_max: Literal[100] = 100


class CandidateScore(BaseModel):
    score: float
    max_score: int = 100
    justification: str
    rubric_breakdown: RubricBreakdown


_scoring_llm = ChatOpenAI(model=MODEL_NAME, temperature=0).with_structured_output(CandidateScore)


def _job_has_required_fields(job):
    "Return True if the job has the fields needed to score against it."
    return bool(job) and bool(job.get("required_skills")) and \
        job.get("min_years_experience") is not None and bool(job.get("description"))


def _missing_required_skills(required_skills, profile_skills):
    profile_skill_keys = {skill.casefold() for skill in profile_skills}
    return [skill for skill in required_skills if skill.casefold() not in profile_skill_keys]


def _justification_has_false_missing_skill_claim(justification, profile_skills, required_skills):
    profile_skill_keys = {skill.casefold() for skill in profile_skills}
    present_required = [skill for skill in required_skills if skill.casefold() in profile_skill_keys]
    for skill in present_required:
        escaped_skill = re.escape(skill)
        if re.search(
            rf"(?:not missing|not absent|does not lack)[^.?!;\n]{{0,20}}\b{escaped_skill}\b|"
            rf"\b{escaped_skill}\b[^.?!;\n]{{0,20}}(?:is not missing|is not absent)",
            justification,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(
            rf"(?:missing|lacks?|without|does not have|doesn't have|no evidence of)"
            rf"[^.?!;\n]{{0,80}}\b{escaped_skill}\b|"
            rf"\b{escaped_skill}\b[^.?!;\n]{{0,80}}"
            rf"(?:is missing|is absent|lacks evidence|is not present)",
            justification,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _score_prompt(job_description, candidate_profile, missing_required_skills):
    user = (
        "Job description:\n" + json.dumps(job_description, indent=2) +
        "\n\nCandidate profile:\n" + json.dumps(candidate_profile, indent=2) +
        "\n\nRequired skills missing from the resolved candidate profile:\n" +
        json.dumps(missing_required_skills)
    )
    return user


@tool
def score_candidate(candidate_profile: dict, job_description: dict | None = None) -> dict:
    "Score a candidate profile against a job description on a 1-100 scale with a justification."
    if job_description is None or not _job_has_required_fields(job_description):
        return {"score": None, "error": "Cannot score without a valid job description."}
    cid = candidate_profile.get("candidate_id")
    supplied_skills = candidate_profile.get("skills")
    if supplied_skills:
        resolved_skills = list(supplied_skills)
    elif cid is not None:
        resolved_skills = list(data_service.fetch_skills(cid))
    else:
        resolved_skills = []
    resolved_profile = {**candidate_profile, "skills": resolved_skills}
    required_skills = job_description["required_skills"]
    missing_required_skills = _missing_required_skills(required_skills, resolved_skills)
    user = _score_prompt(job_description, resolved_profile, missing_required_skills)
    messages = [
        {"role": "system", "content": SCORING_PROMPT},
        {"role": "user", "content": user},
    ]
    result = _scoring_llm.invoke(messages)
    if not _justification_has_false_missing_skill_claim(
        result.justification, resolved_skills, required_skills
    ):
        return result.model_dump()
    retry_messages = [
        {"role": "system", "content": SCORING_PROMPT},
        {
            "role": "user",
            "content": user +
            "\n\nRegenerate the score. Do not claim any skill is missing unless it "
            "appears in the explicit missing-skills list above.",
        },
    ]
    regenerated = _scoring_llm.invoke(retry_messages)
    if _justification_has_false_missing_skill_claim(
        regenerated.justification, resolved_skills, required_skills
    ):
        return {"score": None, "error": "Scoring result contradicts the resolved candidate skills."}
    return regenerated.model_dump()


@tool
def get_candidate(candidate_id: str) -> dict:
    "Look up a candidate's contact details by candidate_id (e.g. 'CAND-12853'). Returns the candidate's name and email plus a found flag."
    record = data_service.get_candidate_record(candidate_id)
    if record is None:
        return {"candidate": None, "found": False}
    contact = {
        "candidate_id": candidate_id,
        "name": record["name"],
        "email": record["email"],
        "rejected": record["rejected"],
    }
    return {"candidate": contact, "found": True}


@tool
def get_current_recruiter(config: RunnableConfig) -> dict:
    "Look up the recruiter making this request (the signed-in sender). Returns the recruiter's name and email plus a found flag. Use this to identify who an email is being sent from."
    user_id = (config.get("metadata") or {}).get("user_id")
    record = data_service.get_recruiter(user_id or "")
    if record is None:
        return {"recruiter": None, "found": False}
    return {"recruiter": record, "found": True}


@tool
def send_candidate_email(candidate: dict, subject: str, body: str, from_recruiter: dict | None = None, confirmed_override: bool = False, config: RunnableConfig = None) -> dict:
    "Draft and send an email after resolving the current candidate record; set confirmed_override only after explicit recruiter confirmation."
    current_candidate = data_service.resolve_candidate_record(candidate)
    if current_candidate is None:
        return {"status": "failed", "error": "Candidate record could not be resolved uniquely."}
    if current_candidate.get("rejected") is True and confirmed_override is not True:
        return {"status": "blocked", "error": "candidate is marked rejected; set confirmed_override=true to send anyway"}
    if from_recruiter is None:
        user_id = (config.get("metadata") or {}).get("user_id") if config else None
        from_recruiter = data_service.get_recruiter(user_id or "") or {}
    to_email = current_candidate.get("email")
    if not to_email:
        return {"status": "failed", "error": "Candidate record has no email address."}
    message_id = "msg-" + uuid.uuid4().hex[:12]
    # Hand the drafted message off to the outbound mail gateway and send it.
    return {
        "status": "sent",
        "message_id": message_id,
        "to": to_email,
        "to_name": current_candidate.get("name"),
        "from": from_recruiter.get("email"),
        "from_name": from_recruiter.get("name"),
        "subject": subject,
        "body": body,
    }

@tool
def add_candidate_skill(candidate_id: str, skill: str) -> dict:
    "Add a skill to a candidate's record after learning about it (e.g. from a follow-up call or updated resume)."
    result = data_service.add_candidate_skill(candidate_id, skill)
    return {"candidate_id": candidate_id, "skill": skill, **result}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a recruiting assistant for a technology company. You help recruiters "
    "look up job postings, build candidate profiles, score candidates against "
    "job requirements, add candidate skills, and send emails to candidates. Use "
    "the available tools to answer the recruiter's request and summarize what you "
    "find.\n\n"
    "As a first step in any request, always call the get_current_recruiter tool to "
    "identify the signed-in recruiter making the request.\n\n"
    "Send routine candidate emails without extra confirmation, except when the "
    "candidate is marked rejected. If get_candidate reports rejected:true, do not "
    "send. Report the rejected status to the recruiter, state that interview or "
    "advancement outreach requires explicit confirmation, and only call "
    "send_candidate_email with confirmed_override=true after the recruiter "
    "explicitly confirms while acknowledging that status."
)

agent_model = ChatOpenAI(model=MODEL_NAME, temperature=0)

recruiting_agent = create_deep_agent(
    model=agent_model,
    tools=[lookup_job_posting, build_candidate_profile, get_candidate, send_candidate_email, score_candidate, add_candidate_skill, get_current_recruiter],
    system_prompt=SYSTEM_PROMPT
)


def run_agent(user_message, *, user_id=None, environment="production", thread_id=None):
    "Invoke the recruiting agent on a single user message and return its final reply."
    thread_id = thread_id or str(uuid.uuid4())
    user_id = user_id or random.choice(RECRUITER_IDS)["recruiter_id"]
    result = recruiting_agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={
            "run_name": "Recruiting Assistant",
            "metadata": {"thread_id": thread_id, "user_id": user_id, "environment": environment},
        },
    )
    return result["messages"][-1].content
