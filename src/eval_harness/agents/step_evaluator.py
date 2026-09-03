"""Step evaluator — the cheap first-line filter.

Runs every turn. Proposes a disposition and extracts facts. Tuned for recall on
leaves: it over-proposes ``paused``/``closed`` rather than risk missing a real
leave, and relies on the reviewer to remove the false positives. On its own
parse failure it defaults to ``continue`` — the safe, non-committal fallback.
"""

from __future__ import annotations

from ..json_repair import chat_json_with_repair
from ..llm.client import LLMClient
from ..state.machine import CONTINUE, DISPOSITIONS, normalize
from . import prompts
from .context_policy import project

ROLE = "step_evaluator"


def normalize_proposal(parsed: dict | None) -> dict:
    if not isinstance(parsed, dict):
        return {
            "disposition": CONTINUE,
            "extracted_facts": {},
            "reason": "evaluator_parse_failed",
            "_ok": False,
        }
    disposition = normalize(parsed.get("disposition"))
    if disposition not in DISPOSITIONS:
        disposition = CONTINUE
    facts = parsed.get("extracted_facts")
    return {
        "disposition": disposition,
        "extracted_facts": facts if isinstance(facts, dict) else {},
        "reason": str(parsed.get("reason") or ""),
        "_ok": True,
    }


def run_evaluator(
    client: LLMClient, context: dict, *, version: str = prompts.DEFAULT_VERSION
) -> dict:
    ctx = project(ROLE, context)
    system = prompts.build_evaluator_system(version=version)
    user = prompts.build_evaluator_user(
        step=ctx["step"],
        user_message=ctx["user_message"],
        assistant_message=ctx["assistant_message"],
        prev_assistant_message=ctx.get("prev_assistant_message") or "",
    )
    parsed = chat_json_with_repair(
        client, role=ROLE, system=system, user=user, cfg={"temperature": 0.2}
    )
    return normalize_proposal(parsed)
