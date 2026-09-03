import pytest

from eval_harness.agents.context_policy import POLICIES, project


def test_projects_only_allowed_fields():
    ctx = {
        "step": "reproduce_issue",
        "user_message": "hi",
        "assistant_message": "hello",
        "proposed": {"disposition": "closed"},
        "secret_internal_field": "should be dropped",
    }
    out = project("reviewer", ctx)
    assert "secret_internal_field" not in out
    assert set(out) <= (POLICIES["reviewer"].required | POLICIES["reviewer"].optional)
    assert out["proposed"] == {"disposition": "closed"}


def test_missing_required_raises():
    with pytest.raises(KeyError, match="missing required"):
        project("step_evaluator", {"step": "x", "user_message": "y"})  # no assistant_message


def test_optional_absence_is_fine():
    out = project("talker", {"step": "identify_product", "user_message": "hi"})
    assert out == {"step": "identify_product", "user_message": "hi"}
