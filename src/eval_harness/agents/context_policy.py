"""Least-privilege context per agent.

Each agent declares the context fields it is allowed to see. Projecting the full
turn context through the policy keeps prompts focused and makes the information
each role depends on explicit and auditable (and it stops an agent from quietly
growing a dependency on a field it was never meant to read).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextPolicy:
    agent: str
    required: frozenset[str]
    optional: frozenset[str]

    def project(self, context: dict) -> dict:
        missing = sorted(k for k in self.required if k not in context)
        if missing:
            raise KeyError(f"{self.agent}: missing required context {missing}")
        allowed = self.required | self.optional
        return {k: v for k, v in context.items() if k in allowed}


POLICIES: dict[str, ContextPolicy] = {
    "talker": ContextPolicy(
        agent="talker",
        required=frozenset({"step", "user_message"}),
        optional=frozenset({"recent_turns"}),
    ),
    "step_evaluator": ContextPolicy(
        agent="step_evaluator",
        required=frozenset({"step", "user_message", "assistant_message"}),
        optional=frozenset({"prev_assistant_message"}),
    ),
    "reviewer": ContextPolicy(
        agent="reviewer",
        required=frozenset({"step", "user_message", "proposed"}),
        optional=frozenset({"assistant_message", "recent_turns"}),
    ),
}


def project(agent: str, context: dict) -> dict:
    return POLICIES[agent].project(context)
