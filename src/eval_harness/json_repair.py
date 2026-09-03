"""Strict-JSON agent output with a single repair pass.

LLMs asked for JSON occasionally wrap it in prose or code fences, or emit a
trailing comma. Rather than crash the turn, we try to parse leniently, and if
that fails we give the model exactly one more attempt with an explicit
instruction appended. If the repair attempt also fails we return ``None`` and
let the caller apply a fail-soft policy — never an exception on the hot path.
"""

from __future__ import annotations

import json
from typing import Protocol

# A repair retry re-sends the original prompt with this suffix appended. It is
# part of the provider contract: fakes and adapters can detect a repair call by
# testing for this tail.
JSON_REPAIR_SUFFIX = (
    "\n\nYour previous response was not valid JSON. "
    "Return ONLY a valid JSON object with the required keys."
)


class _ChatFn(Protocol):
    def __call__(
        self, *, role: str, system: str, user: str, json: bool = False, cfg: dict | None = None
    ) -> str: ...


def try_parse_json_object(text: str | None) -> dict | None:
    """Best-effort parse of a JSON object from ``text``.

    Strips a ```json``` code fence, then falls back to slicing between the first
    ``{`` and last ``}``. Returns ``None`` for anything that is not a JSON object.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            s = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(s[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def chat_json_with_repair(
    client: _ChatFn,
    *,
    role: str,
    system: str,
    user: str,
    cfg: dict | None = None,
) -> dict | None:
    """Call the model for JSON, retrying once with a repair instruction.

    Returns the parsed object, or ``None`` if both the first response and the
    repair attempt are unparseable.
    """
    raw = client.chat(role=role, system=system, user=user, json=True, cfg=cfg)
    parsed = try_parse_json_object(raw)
    if parsed is not None:
        return parsed
    repaired = client.chat(
        role=role, system=system, user=user + JSON_REPAIR_SUFFIX, json=True, cfg=cfg
    )
    return try_parse_json_object(repaired)
