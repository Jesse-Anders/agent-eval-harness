"""Leave reviewer — the expensive final adjudicator.

Runs ONLY when the evaluator proposes a *new* leave. Returns the parsed review
(or ``None``) plus a ``failed`` flag; the caller feeds both into
``state.machine.apply_review``, which applies the clamp and the soften-on-failure
policy. The reviewer itself never mutates state — it only advises.
"""

from __future__ import annotations

from ..json_repair import chat_json_with_repair
from ..llm.client import LLMClient
from . import prompts
from .context_policy import project

ROLE = "reviewer"


def run_reviewer(
    client: LLMClient, context: dict, *, version: str = prompts.DEFAULT_VERSION
) -> tuple[dict | None, bool]:
    ctx = project(ROLE, context)
    system = prompts.build_reviewer_system(version=version)
    user = prompts.build_reviewer_user(
        step=ctx["step"],
        proposed=ctx["proposed"],
        user_message=ctx["user_message"],
        assistant_message=ctx.get("assistant_message") or "",
        recent_turns=ctx.get("recent_turns") or [],
    )
    parsed = chat_json_with_repair(
        client, role=ROLE, system=system, user=user, cfg={"temperature": 0.1}
    )
    return parsed, parsed is None
