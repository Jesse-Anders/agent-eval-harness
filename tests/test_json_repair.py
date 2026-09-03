from eval_harness.json_repair import (
    JSON_REPAIR_SUFFIX,
    chat_json_with_repair,
    try_parse_json_object,
)
from eval_harness.llm.fake import FakeLLM


def test_parses_plain_object():
    assert try_parse_json_object('{"a": 1}') == {"a": 1}


def test_strips_code_fence():
    assert try_parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_slices_object_out_of_prose():
    assert try_parse_json_object('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_rejects_non_object():
    assert try_parse_json_object("[1, 2, 3]") is None
    assert try_parse_json_object("not json") is None
    assert try_parse_json_object("") is None
    assert try_parse_json_object(None) is None


def test_repair_succeeds_on_second_pass():
    # First call returns garbage; the repair retry returns clean JSON.
    fake = FakeLLM({"r": [["totally not json", '{"ok": true}']]})
    out = chat_json_with_repair(fake, role="r", system="s", user="u")
    assert out == {"ok": True}
    assert [c["repair"] for c in fake.calls] == [False, True]


def test_repair_gives_up_and_returns_none():
    # Malformed on both the base call and the repair retry -> None (fail-soft).
    fake = FakeLLM({"r": ["still not json"]})
    assert chat_json_with_repair(fake, role="r", system="s", user="u") is None
    assert len(fake.calls) == 2


def test_no_repair_when_first_pass_parses():
    fake = FakeLLM({"r": ['{"ok": 1}']})
    assert chat_json_with_repair(fake, role="r", system="s", user="u") == {"ok": 1}
    assert len(fake.calls) == 1


def test_repair_suffix_is_appended_once():
    seen: list[str] = []

    class Spy:
        def chat(self, *, role, system, user, json=False, cfg=None):
            seen.append(user)
            return "nope"

    chat_json_with_repair(Spy(), role="r", system="s", user="base")
    assert seen[0] == "base"
    assert seen[1] == "base" + JSON_REPAIR_SUFFIX
