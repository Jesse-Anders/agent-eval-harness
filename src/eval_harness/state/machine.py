"""The disposition state machine and the reviewer merge rule.

Every turn, the cheap evaluator proposes a *disposition* for the session. Two of
them are "leaves" — the high-stakes decisions to stop engaging:

    continue  keep working the current step        (not a leave)
    advance   current step done, move to the next  (not a leave)
    paused    park the session, resume later       (leave)
    closed    resolve and end the ticket           (leave)

The leaves form a commitment ladder ``continue < paused < closed``. The reviewer
only ever runs on a proposed leave, and this module encodes the two guarantees
that make the cascade sound:

* **Clamp** — the reviewer may confirm a leave or walk it *down* the ladder, but
  never *up*. It cannot escalate ``paused`` to ``closed``. Structurally, stage 2
  can only reduce commitment, so it can never turn a correct "stay" into a leave
  — i.e. it can never introduce a false negative that the recall-tuned filter
  was designed to avoid.
* **Soften-on-failure** — if the reviewer cannot run (timeout / unparseable
  output), a proposed ``closed`` is softened to ``paused`` rather than trusted.
  When unsure, park; don't commit to the irreversible end state.
"""

from __future__ import annotations

# Dispositions the evaluator may propose.
CONTINUE = "continue"
ADVANCE = "advance"
PAUSED = "paused"
CLOSED = "closed"

DISPOSITIONS: tuple[str, ...] = (CONTINUE, ADVANCE, PAUSED, CLOSED)

# The "leave" set: the decisions the reviewer adjudicates.
LEAVE: frozenset[str] = frozenset({PAUSED, CLOSED})

# Commitment ladder for the reviewer merge (higher = more committal).
_RANK: dict[str, int] = {CONTINUE: 0, PAUSED: 1, CLOSED: 2}

# The ordered intake steps the talker walks a ticket through.
INTAKE_STEPS: tuple[str, ...] = (
    "identify_product",
    "reproduce_issue",
    "gather_environment",
    "propose_resolution",
    "resolved",
)


def normalize(value: object) -> str:
    return str(value or "").strip().lower()


def is_leave(disposition: object) -> bool:
    return normalize(disposition) in LEAVE


def next_step(step: str) -> str:
    """Return the step after ``step`` (the terminal step returns itself)."""
    s = normalize(step)
    if s not in INTAKE_STEPS:
        return INTAKE_STEPS[0]
    i = INTAKE_STEPS.index(s)
    return INTAKE_STEPS[min(i + 1, len(INTAKE_STEPS) - 1)]


def soften_on_failure(proposed: str) -> tuple[str, str]:
    """Fail-soft: ``closed`` -> ``paused``; ``paused`` stays ``paused``."""
    p = normalize(proposed)
    if p == CLOSED:
        return PAUSED, "soften_fail_closed_to_paused"
    if p == PAUSED:
        return PAUSED, "soften_fail_keep_paused"
    return (p or CONTINUE), "soften_fail_noop"


def apply_review(
    *,
    proposed: str,
    review: dict | None,
    failed: bool = False,
) -> tuple[str, str]:
    """Merge the reviewer's decision with the evaluator's proposed leave.

    Returns ``(final_disposition, outcome_tag)``. ``outcome_tag`` is a compact,
    greppable label for what happened (``confirm_*``, ``revert_to_continue``,
    ``downgrade_to_paused``, ``clamp_no_escalate``, ``soften_fail_*``,
    ``skip_not_leave``) — the same style of audit tag the production system logs.
    """
    proposed_n = normalize(proposed)
    if proposed_n not in LEAVE:
        # The reviewer should not have been invoked; leave the proposal untouched.
        return (proposed_n or CONTINUE), "skip_not_leave"

    if failed or not isinstance(review, dict):
        return soften_on_failure(proposed_n)

    reviewed = normalize(review.get("disposition"))
    if reviewed not in _RANK:
        return soften_on_failure(proposed_n)

    # Clamp: the reviewer can confirm or de-escalate, never escalate.
    if _RANK[reviewed] > _RANK[proposed_n]:
        return proposed_n, "clamp_no_escalate"

    if reviewed == proposed_n:
        return reviewed, f"confirm_{reviewed}"
    if reviewed == CONTINUE:
        return CONTINUE, "revert_to_continue"
    return reviewed, f"downgrade_to_{reviewed}"
