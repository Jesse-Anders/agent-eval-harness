import pytest

from eval_harness.state.machine import (
    CLOSED,
    CONTINUE,
    PAUSED,
    apply_review,
    is_leave,
    next_step,
    soften_on_failure,
)


def test_leave_set():
    assert is_leave(PAUSED) and is_leave(CLOSED)
    assert not is_leave(CONTINUE)
    assert not is_leave("advance")


def test_next_step_walks_and_saturates():
    assert next_step("identify_product") == "reproduce_issue"
    assert next_step("resolved") == "resolved"
    assert next_step("bogus") == "identify_product"


# --- confirm / de-escalate / revert ------------------------------------------


@pytest.mark.parametrize("leave", [PAUSED, CLOSED])
def test_reviewer_confirms(leave):
    final, outcome = apply_review(proposed=leave, review={"disposition": leave})
    assert final == leave
    assert outcome == f"confirm_{leave}"


def test_reviewer_reverts_to_continue():
    # The money case: proposed leave, but it was really a topic switch -> stay.
    final, outcome = apply_review(proposed=CLOSED, review={"disposition": CONTINUE})
    assert final == CONTINUE
    assert outcome == "revert_to_continue"


def test_reviewer_downgrades_closed_to_paused():
    final, outcome = apply_review(proposed=CLOSED, review={"disposition": PAUSED})
    assert final == PAUSED
    assert outcome == "downgrade_to_paused"


# --- clamp: the reviewer can never escalate ----------------------------------


def test_clamp_blocks_escalation_paused_to_closed():
    final, outcome = apply_review(proposed=PAUSED, review={"disposition": CLOSED})
    assert final == PAUSED
    assert outcome == "clamp_no_escalate"


# --- fail-soft ----------------------------------------------------------------


def test_soften_on_failure_direct():
    assert soften_on_failure(CLOSED) == (PAUSED, "soften_fail_closed_to_paused")
    assert soften_on_failure(PAUSED) == (PAUSED, "soften_fail_keep_paused")


def test_failed_reviewer_softens_closed_to_paused():
    final, outcome = apply_review(proposed=CLOSED, review=None, failed=True)
    assert final == PAUSED
    assert outcome == "soften_fail_closed_to_paused"


def test_unparseable_reviewer_output_softens():
    final, outcome = apply_review(proposed=CLOSED, review=None)
    assert final == PAUSED
    assert outcome == "soften_fail_closed_to_paused"


def test_garbage_disposition_softens():
    final, outcome = apply_review(proposed=CLOSED, review={"disposition": "explode"})
    assert final == PAUSED
    assert outcome == "soften_fail_closed_to_paused"


# --- guard: reviewer must not run on a non-leave ------------------------------


def test_non_leave_proposal_is_untouched():
    final, outcome = apply_review(proposed=CONTINUE, review={"disposition": CLOSED})
    assert final == CONTINUE
    assert outcome == "skip_not_leave"
