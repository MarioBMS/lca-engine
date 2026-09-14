import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

package = types.ModuleType("recruiting_agent")
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules["recruiting_agent"] = package

from recruiting_agent import data_service
from recruiting_agent.recruiting_agent import CandidateScore, score_candidate, send_candidate_email


def invoke_email(candidate, **kwargs):
    return send_candidate_email.invoke({
        "candidate": candidate,
        "subject": "Interview invitation",
        "body": "Please choose an interview time.",
        **kwargs,
    })


def test_rejected_candidate_without_id_is_blocked():
    result = invoke_email({
        "name": "Priya Nair",
        "email": "priya.nair@example.com",
    })

    assert result == {
        "status": "blocked",
        "error": "candidate is marked rejected; set confirmed_override=true to send anyway",
    }


def test_non_rejected_candidate_sends_in_one_step():
    result = invoke_email({
        "candidate_id": "CAND-12853",
        "name": "Omar Okafor",
        "email": "omar.okafor@example.com",
    })

    assert result["status"] == "sent"
    assert result["to"] == "omar.okafor@example.com"


def test_added_skill_is_persisted_and_used_for_scoring(monkeypatch):
    candidate_id = "CAND-12853"
    original_skills = list(data_service.CANDIDATES[candidate_id]["skills"])
    data_service._PROFILES.pop(candidate_id, None)
    try:
        data_service.CANDIDATES[candidate_id]["skills"] = ["Python"]
        data_service.save_profile_to_db(candidate_id, {
            "candidate_id": candidate_id,
            "skills": ["Python"],
        })
        baseline_profile = {"candidate_id": candidate_id, "skills": ["Python"]}
        job = {
            "required_skills": ["Python", "Go"],
            "min_years_experience": 1,
            "description": "Build backend systems.",
        }

        class FakeScoringLLM:
            def invoke(self, messages):
                content = messages[-1]["content"]
                has_go = content.endswith("[]")
                return CandidateScore(
                    score=90 if has_go else 60,
                    justification="Python is present; Go is missing." if not has_go else "Python and Go are present.",
                    rubric_breakdown={"experience": 30, "skills_match": 30, "seniority_fit": 20},
                )

        monkeypatch.setattr("recruiting_agent.recruiting_agent._scoring_llm", FakeScoringLLM())
        baseline = score_candidate.invoke({"candidate_profile": baseline_profile, "job_description": job})
        updated = data_service.add_candidate_skill(candidate_id, "Go")
        assert updated["updated"] is True
        assert "Go" in data_service.fetch_skills(candidate_id)
        current = data_service.get_profile_from_db(candidate_id)["candidate_profile"]
        scored = score_candidate.invoke({"candidate_profile": current, "job_description": job})
        assert "Go is missing" not in scored["justification"]
        assert scored["score"] > baseline["score"]
    finally:
        data_service.CANDIDATES[candidate_id]["skills"] = original_skills
        data_service._PROFILES.pop(candidate_id, None)
