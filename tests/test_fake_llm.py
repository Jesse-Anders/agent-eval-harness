import json
from pathlib import Path

import pytest

from eval_harness.llm.client import LLMClient
from eval_harness.llm.fake import FakeLLM


def test_serves_responses_per_role_in_order():
    fake = FakeLLM({"talker": ["hi", "bye"], "reviewer": ["{}"]})
    assert fake.chat(role="talker", system="s", user="u") == "hi"
    assert fake.chat(role="reviewer", system="s", user="u") == "{}"
    assert fake.chat(role="talker", system="s", user="u") == "bye"


def test_satisfies_client_protocol():
    assert isinstance(FakeLLM({"r": ["x"]}), LLMClient)


def test_unknown_role_raises():
    with pytest.raises(KeyError):
        FakeLLM({"talker": ["hi"]}).chat(role="ghost", system="s", user="u")


def test_exhaustion_raises_with_helpful_message():
    fake = FakeLLM({"r": ["one"]})
    fake.chat(role="r", system="s", user="u")
    with pytest.raises(IndexError, match="exhausted"):
        fake.chat(role="r", system="s", user="u")


def test_from_fixture_roundtrip(tmp_path: Path):
    fx = tmp_path / "fx.json"
    fx.write_text(json.dumps({"name": "demo", "responses": {"r": ["a"]}}), encoding="utf-8")
    fake = FakeLLM.from_fixture(fx)
    assert fake.name == "demo"
    assert fake.chat(role="r", system="s", user="u") == "a"
