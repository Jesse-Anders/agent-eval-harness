"""Deterministic, fixture-driven LLM used in tests and offline replay.

Responses are scripted per role and consumed in call order, so a whole
multi-agent turn is reproducible with no network and no credentials. This is the
piece that makes the cascade runnable in CI.

Fixture shape::

    {
      "name": "topic_switch",
      "responses": {
        "talker": ["...", "..."],                 # one per turn
        "step_evaluator": ["{...json...}", ...],   # one per turn
        "reviewer": ["{...json...}"]               # one per turn the reviewer runs
      }
    }

Each entry is either a string (returned for both the first call and any repair
retry) or a two-element ``[first, repaired]`` list, which lets a fixture exercise
the JSON repair path: the first call returns malformed text, the repair retry
returns clean JSON. A bare malformed string that is never repaired drives the
fail-soft path instead.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

from ..json_repair import JSON_REPAIR_SUFFIX

_REPAIR_TAIL = JSON_REPAIR_SUFFIX.strip()

Entry = str | list[str]


class FakeLLM:
    """An :class:`~eval_harness.llm.client.LLMClient` backed by scripted output."""

    def __init__(self, responses: dict[str, list[Entry]], *, name: str = "fake") -> None:
        self.name = name
        self._responses: dict[str, list[Entry]] = {r: list(v) for r, v in responses.items()}
        self._index: dict[str, int] = {r: 0 for r in self._responses}
        self._active: dict[str, Entry] = {}
        self.calls: list[dict] = []

    @classmethod
    def from_fixture(cls, path: str | Path) -> FakeLLM:
        p = Path(path)
        data = _json.loads(p.read_text(encoding="utf-8"))
        return cls(data.get("responses") or {}, name=str(data.get("name") or p.stem))

    @staticmethod
    def _split(entry: Entry) -> tuple[str, str]:
        if isinstance(entry, list):
            first = entry[0] if entry else ""
            return first, (entry[1] if len(entry) > 1 else first)
        return entry, entry

    def chat(
        self,
        *,
        role: str,
        system: str,
        user: str,
        json: bool = False,
        cfg: dict | None = None,
    ) -> str:
        if role not in self._responses:
            raise KeyError(f"FakeLLM has no scripted responses for role {role!r}")

        if user.rstrip().endswith(_REPAIR_TAIL):
            entry = self._active.get(role)
            if entry is None:
                raise RuntimeError(f"repair call for role {role!r} before any base call")
            _, repaired = self._split(entry)
            self.calls.append({"role": role, "repair": True, "response": repaired})
            return repaired

        i = self._index[role]
        entries = self._responses[role]
        if i >= len(entries):
            raise IndexError(
                f"FakeLLM exhausted scripted responses for role {role!r} "
                f"(call #{i + 1}, have {len(entries)})"
            )
        self._index[role] = i + 1
        self._active[role] = entries[i]
        first, _ = self._split(entries[i])
        self.calls.append({"role": role, "repair": False, "response": first})
        return first
