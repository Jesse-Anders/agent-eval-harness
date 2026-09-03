"""Per-run session state carried through a replay.

A fresh synthetic session id is minted on every run so replayed runs never
collide with each other or with any recorded data — the run is deterministic in
its *decisions*, not in its identity.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .machine import ADVANCE, CONTINUE, INTAKE_STEPS, is_leave, next_step, normalize


def new_session_id() -> str:
    return f"synsess-{uuid.uuid4().hex[:12]}"


@dataclass
class Session:
    session_id: str = field(default_factory=new_session_id)
    step: str = INTAKE_STEPS[0]
    disposition: str = CONTINUE
    ended: bool = False
    turn: int = 0
    facts: dict[str, str] = field(default_factory=dict)

    def apply(self, *, disposition: str, extracted_facts: dict | None = None) -> dict:
        """Apply a decided disposition + extracted facts, returning the state delta."""
        before = {"step": self.step, "disposition": self.disposition, "ended": self.ended}
        d = normalize(disposition)

        new_facts: dict[str, str] = {}
        for k, v in (extracted_facts or {}).items():
            key = str(k).strip()
            val = str(v).strip()
            if key and val and self.facts.get(key) != val:
                new_facts[key] = val
        self.facts.update(new_facts)

        self.disposition = d
        if d == ADVANCE:
            self.step = next_step(self.step)
        if is_leave(d):
            self.ended = True

        return {
            "before": before,
            "after": {"step": self.step, "disposition": self.disposition, "ended": self.ended},
            "new_facts": new_facts,
        }


__all__ = ["Session", "new_session_id", "CONTINUE"]
