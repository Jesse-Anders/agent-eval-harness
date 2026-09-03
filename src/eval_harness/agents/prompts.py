"""Prompt builders — prompts are versioned code, not loose strings.

Every prompt is produced by a function that takes structured inputs and returns
a string. Versions are explicit (``v1``, ``v2``) so a prompt change is a diff
under review and the ``compare`` command can A/B two versions over the same
scenario and report which decisions flipped.

Domain: an automated intake assistant for a fictional SaaS ("Meridian"). It
walks a support ticket through a fixed sequence of steps. None of this text is
tied to any real product.
"""

from __future__ import annotations

from ..state.machine import INTAKE_STEPS

DEFAULT_VERSION = "v1"

_STEP_GUIDE = {
    "identify_product": "Determine which Meridian product/area the issue is in.",
    "reproduce_issue": "Get concrete steps to reproduce the problem.",
    "gather_environment": "Collect environment details (OS, app version, browser).",
    "propose_resolution": "Offer a concrete next step or workaround.",
    "resolved": "The ticket is resolved; confirm and wrap up.",
}


def _steps_block() -> str:
    return "\n".join(f"  {i + 1}. {s} — {_STEP_GUIDE[s]}" for i, s in enumerate(INTAKE_STEPS))


# --------------------------------------------------------------------------- #
# Talker — writes dialogue only. It never decides state.
# --------------------------------------------------------------------------- #


def build_talker_system(*, version: str = DEFAULT_VERSION) -> str:
    return (
        "You are the intake assistant for Meridian, a SaaS product. You write the "
        "next single message to the customer: warm, brief, one question at a time. "
        "You are ONLY the voice — you never decide whether to advance, pause, or "
        "close the ticket. Do not mention internal steps or state.\n\n"
        f"The intake proceeds through these steps:\n{_steps_block()}"
    )


def build_talker_user(*, step: str, user_message: str, recent_turns: list[str]) -> str:
    history = "\n".join(recent_turns) if recent_turns else "(start of conversation)"
    return (
        f"Current step: {step}\n\n"
        f"Recent conversation:\n{history}\n\n"
        f"Customer just said:\n{user_message}\n\n"
        "Write your next message."
    )


# --------------------------------------------------------------------------- #
# Step evaluator (cheap first-line filter) — proposes a disposition + facts.
# Tuned for RECALL on leaves: when the customer might be trying to stop, propose
# the leave. A false "keep going" is unrecoverable (the reviewer never sees it);
# a false leave is cheap (the reviewer removes it).
# --------------------------------------------------------------------------- #


def build_evaluator_system(*, version: str = DEFAULT_VERSION) -> str:
    common = (
        "You classify the state of a Meridian support-intake conversation. "
        "Return ONLY a JSON object:\n"
        '  {"disposition": "continue|advance|paused|closed", '
        '"extracted_facts": {..}, "reason": ".."}\n\n'
        "Dispositions:\n"
        "  continue — the customer is still engaged on the current step.\n"
        "  advance  — the current step is satisfied; move to the next step.\n"
        "  paused   — the customer wants to stop for now and resume later.\n"
        "  closed   — the customer considers the issue done / wants to end.\n\n"
        "extracted_facts: a flat object of any concrete facts stated this turn "
        "(product, version, os, browser, repro steps, etc.). Omit anything not stated.\n"
    )
    if version == "v2":
        # v2 tightens the leave rule: a stop signal must be about the WHOLE session,
        # not merely the current topic. This is the fix that flips the topic-switch case.
        return common + (
            "\nLEAVE RULE (v2): Propose paused or closed ONLY when the stop signal "
            "clearly refers to the whole session ('let's stop for today', 'we're done "
            "here'). If the customer is done with one topic but raises or implies "
            "another, that is 'continue' or 'advance', NOT a leave. When genuinely "
            "unsure whether a stop is real, prefer paused so it can be reviewed."
        )
    return common + (
        "\nLEAVE RULE (v1): If the customer signals they are finished, thanks you off, "
        "or asks to stop, propose a leave (paused or closed). When unsure, prefer "
        "proposing the leave rather than continuing."
    )


def build_evaluator_user(
    *, step: str, user_message: str, assistant_message: str, prev_assistant_message: str
) -> str:
    return (
        f"Current step: {step}\n"
        f"Previous assistant message: {prev_assistant_message or '(none)'}\n"
        f"Assistant message this turn: {assistant_message or '(none)'}\n\n"
        f"Customer message:\n{user_message}\n\n"
        "Classify this turn."
    )


# --------------------------------------------------------------------------- #
# Reviewer (expensive final adjudicator) — runs ONLY on a proposed leave.
# Conservative: confirm a leave only when the customer truly wants to stop.
# It may confirm or walk a leave down, never escalate it.
# --------------------------------------------------------------------------- #


def build_reviewer_system(*, version: str = DEFAULT_VERSION) -> str:
    return (
        "You are the leave reviewer for a Meridian support-intake conversation. The "
        "first-line classifier proposed that the session should PAUSE or CLOSE. Your "
        "job is to catch false leaves.\n\n"
        "Return ONLY a JSON object: {\"disposition\": \"continue|paused|closed\", "
        '"reason": ".."}\n\n'
        "Decide:\n"
        "  continue — the customer is NOT actually leaving. Common trap: they finished "
        "one topic ('that's sorted, thanks') and immediately raise or imply another "
        "issue. A topic switch is not a leave.\n"
        "  paused   — the customer wants to stop for now and resume later.\n"
        "  closed   — the customer clearly wants the whole ticket done and ended.\n\n"
        "You may confirm the proposed leave or reduce it; never increase it. If you "
        "cannot tell, choose the less final option."
    )


def build_reviewer_user(
    *,
    step: str,
    proposed: dict,
    user_message: str,
    assistant_message: str,
    recent_turns: list[str],
) -> str:
    history = "\n".join(recent_turns) if recent_turns else "(no earlier turns)"
    return (
        f"Current step: {step}\n"
        f"First-line proposal: {proposed.get('disposition')} "
        f"(reason: {proposed.get('reason') or 'n/a'})\n\n"
        f"Recent conversation (oldest first):\n{history}\n\n"
        f"Assistant message this turn: {assistant_message or '(none)'}\n"
        f"Customer message this turn:\n{user_message}\n\n"
        "Is this really a leave? Return your decision."
    )
