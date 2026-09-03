"""Reviewer agent + machine, wired through the FakeLLM end of the cascade."""

from eval_harness.agents.reviewer import run_reviewer
from eval_harness.agents.step_evaluator import normalize_proposal, run_evaluator
from eval_harness.llm.fake import FakeLLM
from eval_harness.state.machine import CLOSED, CONTINUE, PAUSED, apply_review


def _reviewer_ctx(proposed_disposition: str) -> dict:
    return {
        "step": "reproduce_issue",
        "user_message": "that's sorted thanks — actually the export is also broken",
        "assistant_message": "Glad that helped!",
        "recent_turns": ["user: my login failed", "assistant: try resetting"],
        "proposed": {"disposition": proposed_disposition, "reason": "customer said thanks"},
    }


def test_reviewer_reverses_false_leave():
    fake = FakeLLM({"reviewer": ['{"disposition": "continue", "reason": "topic switch"}']})
    review, failed = run_reviewer(fake, _reviewer_ctx(CLOSED))
    assert failed is False
    final, outcome = apply_review(proposed=CLOSED, review=review)
    assert final == CONTINUE
    assert outcome == "revert_to_continue"


def test_reviewer_escalation_attempt_is_clamped():
    # Reviewer tries to escalate paused -> closed; the machine forbids it.
    fake = FakeLLM({"reviewer": ['{"disposition": "closed", "reason": "seems final"}']})
    review, failed = run_reviewer(fake, _reviewer_ctx(PAUSED))
    final, outcome = apply_review(proposed=PAUSED, review=review)
    assert final == PAUSED
    assert outcome == "clamp_no_escalate"


def test_malformed_reviewer_output_fails_soft():
    # Malformed on both base and repair -> failed -> soften closed to paused.
    fake = FakeLLM({"reviewer": ["the customer is definitely done, trust me"]})
    review, failed = run_reviewer(fake, _reviewer_ctx(CLOSED))
    assert review is None and failed is True
    final, outcome = apply_review(proposed=CLOSED, review=review, failed=failed)
    assert final == PAUSED
    assert outcome == "soften_fail_closed_to_paused"


def test_reviewer_recovers_via_repair_pass():
    fake = FakeLLM(
        {"reviewer": [["not json at all", '{"disposition": "continue", "reason": "switch"}']]}
    )
    review, failed = run_reviewer(fake, _reviewer_ctx(CLOSED))
    assert failed is False
    assert apply_review(proposed=CLOSED, review=review)[0] == CONTINUE


def test_evaluator_normalizes_and_defaults():
    assert normalize_proposal(None)["disposition"] == CONTINUE
    assert normalize_proposal({"disposition": "explode"})["disposition"] == CONTINUE
    good = normalize_proposal({"disposition": "closed", "extracted_facts": {"os": "macOS"}})
    assert good["disposition"] == CLOSED
    assert good["extracted_facts"] == {"os": "macOS"}


def test_evaluator_run_with_fake():
    fake = FakeLLM(
        {"step_evaluator": ['{"disposition": "advance", "extracted_facts": {"product": "Sync"}}']}
    )
    ctx = {
        "step": "identify_product",
        "user_message": "it's the Sync product",
        "assistant_message": "Which product?",
        "prev_assistant_message": "",
    }
    out = run_evaluator(fake, ctx)
    assert out["disposition"] == "advance"
    assert out["extracted_facts"] == {"product": "Sync"}
