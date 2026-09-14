import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

package = types.ModuleType("recruiting_agent")
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules["recruiting_agent"] = package

from recruiting_agent.recruiting_agent import send_candidate_email


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
