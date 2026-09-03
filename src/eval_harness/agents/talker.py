"""Talker agent — produces the next customer-facing message. No state decisions."""

from __future__ import annotations

from ..llm.client import LLMClient
from . import prompts
from .context_policy import project

ROLE = "talker"


def run_talker(client: LLMClient, context: dict, *, version: str = prompts.DEFAULT_VERSION) -> str:
    ctx = project(ROLE, context)
    system = prompts.build_talker_system(version=version)
    user = prompts.build_talker_user(
        step=ctx["step"],
        user_message=ctx["user_message"],
        recent_turns=ctx.get("recent_turns") or [],
    )
    return client.chat(role=ROLE, system=system, user=user, cfg={"temperature": 0.4}).strip()
