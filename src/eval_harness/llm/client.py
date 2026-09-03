"""The single LLM interface every agent talks to.

Agents never import a provider SDK. They depend only on :class:`LLMClient`, so
the same proposer/reviewer code runs against a deterministic fake in tests and a
real provider under ``make replay``. ``role`` is first-class because model
*tiering* is the whole point: different roles resolve to different model tiers
(a cheap first-line filter vs. an expensive reviewer).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def chat(
        self,
        *,
        role: str,
        system: str,
        user: str,
        json: bool = False,
        cfg: dict | None = None,
    ) -> str:
        """Return the model's completion for a system+user prompt.

        ``role`` selects the model tier for the call. ``json=True`` is a hint
        that the caller expects a JSON object (adapters may set a response
        format); parsing/repair is handled by ``json_repair`` regardless.
        ``cfg`` carries optional per-call hints (e.g. ``temperature``).
        """
        ...
