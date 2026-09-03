"""Turn-by-turn transcript replay.

Feeds a recorded transcript through the cascade one user turn at a time, with a
fresh synthetic session id, and records a per-turn artifact (inputs, agent
outputs, the state delta) for every turn. The reviewer fires only when the
evaluator proposes a *new* leave — exactly the production gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..agents import prompts
from ..agents.reviewer import run_reviewer
from ..agents.step_evaluator import run_evaluator
from ..agents.talker import run_talker
from ..llm.client import LLMClient
from ..state.machine import apply_review, is_leave, normalize
from ..state.session import Session


def _load_scenario(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("turns"), list) or not data["turns"]:
        raise ValueError(f"{path}: scenario needs a non-empty 'turns' list")
    return data


def run_scenario(
    scenario: dict,
    client: LLMClient,
    *,
    version: str = prompts.DEFAULT_VERSION,
    out_dir: str | Path | None = None,
) -> dict:
    """Replay one scenario. Returns a run record; writes artifacts if out_dir given."""
    session = Session()
    name = str(scenario.get("name") or "scenario")
    recent: list[str] = []
    prev_assistant = ""
    turns_out: list[dict] = []

    for i, turn in enumerate(scenario["turns"]):
        user_message = str(turn.get("user") or "").strip()
        step_before = session.step

        assistant = run_talker(
            client,
            {"step": step_before, "user_message": user_message, "recent_turns": list(recent)},
            version=version,
        )

        proposal = run_evaluator(
            client,
            {
                "step": step_before,
                "user_message": user_message,
                "assistant_message": assistant,
                "prev_assistant_message": prev_assistant,
            },
            version=version,
        )

        proposed = proposal["disposition"]
        # Gate: reviewer runs only on a *new* proposed leave.
        gate = is_leave(proposed) and proposed != session.disposition
        review: dict | None = None
        reviewer_failed = False
        if gate:
            review, reviewer_failed = run_reviewer(
                client,
                {
                    "step": step_before,
                    "user_message": user_message,
                    "assistant_message": assistant,
                    "recent_turns": list(recent),
                    "proposed": proposal,
                },
                version=version,
            )
            final, outcome = apply_review(
                proposed=proposed, review=review, failed=reviewer_failed
            )
        else:
            final, outcome = proposed, "no_review"

        delta = session.apply(disposition=final, extracted_facts=proposal["extracted_facts"])
        session.turn = i + 1

        record = {
            "turn": i,
            "step_before": step_before,
            "user": user_message,
            "assistant": assistant,
            "proposal": proposal,
            "reviewer_ran": gate,
            "review": review,
            "reviewer_failed": reviewer_failed,
            "final_disposition": final,
            "outcome": outcome,
            "state_delta": delta,
            "expect": turn.get("expect"),
        }
        turns_out.append(record)

        recent.append(f"user: {user_message}")
        recent.append(f"assistant: {assistant}")
        recent[:] = recent[-8:]
        prev_assistant = assistant

        if is_leave(final):
            break

    run = {
        "scenario": name,
        "prompt_version": version,
        "session_id": session.session_id,
        "turns": turns_out,
        "final": {
            "disposition": session.disposition,
            "step": session.step,
            "ended": session.ended,
            "turn_count": len(turns_out),
            "facts": dict(session.facts),
        },
        "expected": scenario.get("expected") or {},
    }

    if out_dir is not None:
        _write_artifacts(out_dir, run)
    return run


def _write_artifacts(out_dir: str | Path, run: dict) -> None:
    base = Path(out_dir) / run["scenario"]
    turns_dir = base / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)
    for rec in run["turns"]:
        (turns_dir / f"turn_{rec['turn']:02d}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (base / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_scenario_file(
    path: str | Path,
    client: LLMClient,
    *,
    version: str = prompts.DEFAULT_VERSION,
    out_dir: str | Path | None = None,
) -> dict:
    return run_scenario(_load_scenario(path), client, version=version, out_dir=out_dir)


__all__ = ["run_scenario", "run_scenario_file", "normalize"]
