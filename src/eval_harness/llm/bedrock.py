"""AWS Bedrock adapter — the real provider behind the same LLMClient interface.

This is where model *tiering* lives: the ``reviewer`` role resolves to a more
capable (more expensive) model, every other role to a cheaper default. That is
the production shape — you spend the expensive model only on the small set of
turns the reviewer actually runs on.

boto3 is imported lazily inside ``__init__`` so the offline test/replay path
never needs the ``[bedrock]`` extra or any credentials. Nothing in the core
package imports this module at load time.
"""

from __future__ import annotations

import os

# Role -> model tier. Only the reviewer gets the expensive model.
_EXPENSIVE_ROLES = frozenset({"reviewer"})

_DEFAULT_CHEAP_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
_DEFAULT_EXPENSIVE_MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"


class BedrockLLM:
    def __init__(self, *, region: str | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - exercised only with the extra
            raise RuntimeError(
                "BedrockLLM needs the optional dependency: pip install -e '.[bedrock]'"
            ) from exc
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = boto3.client("bedrock-runtime", region_name=self._region)
        self._cheap_model = os.environ.get("EVAL_HARNESS_MODEL_DEFAULT", _DEFAULT_CHEAP_MODEL)
        self._expensive_model = os.environ.get(
            "EVAL_HARNESS_MODEL_REVIEWER", _DEFAULT_EXPENSIVE_MODEL
        )

    def model_for(self, role: str) -> str:
        return self._expensive_model if role in _EXPENSIVE_ROLES else self._cheap_model

    def chat(
        self,
        *,
        role: str,
        system: str,
        user: str,
        json: bool = False,
        cfg: dict | None = None,
    ) -> str:
        cfg = cfg or {}
        response = self._client.converse(
            modelId=self.model_for(role),
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={
                "temperature": float(cfg.get("temperature", 0.2)),
                "maxTokens": int(cfg.get("max_tokens", 512)),
            },
        )
        parts = response["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts).strip()
