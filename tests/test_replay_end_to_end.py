"""End-to-end replay of every shipped scenario through the FakeLLM."""

from pathlib import Path

import pytest

from eval_harness.llm.fake import FakeLLM
from eval_harness.replay.report import build_report
from eval_harness.replay.runner import run_scenario_file
from eval_harness.replay.scoring import score_run

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = sorted((ROOT / "scenarios").glob("*.json"))


def _run(name: str) -> dict:
    scenario = ROOT / "scenarios" / f"{name}.json"
    fixture = ROOT / "fixtures" / f"{name}.json"
    return run_scenario_file(scenario, FakeLLM.from_fixture(fixture, version="v1"))


def test_scenarios_exist():
    assert {p.stem for p in SCENARIOS} == {
        "fact_alignment",
        "happy_path",
        "hold_missing_env",
        "reviewer_failsoft",
        "topic_switch",
    }


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_every_scenario_passes(path: Path):
    fixture = ROOT / "fixtures" / path.name
    run = run_scenario_file(path, FakeLLM.from_fixture(fixture, version="v1"))
    assert score_run(run)["ok"], f"{path.stem} did not meet its declared expectations"


def test_fresh_session_id_each_run():
    a = _run("happy_path")["session_id"]
    b = _run("happy_path")["session_id"]
    assert a != b and a.startswith("synsess-")


def test_topic_switch_reviewer_reverses_false_positive():
    run = _run("topic_switch")
    score = score_run(run)
    corrections = score["dispositions"]["reviewer_corrections"]
    expected = {"turn": 2, "from": "closed", "to": "continue", "outcome": "revert_to_continue"}
    assert expected in corrections
    assert score["dispositions"]["stage1"]["fp"] == 1
    assert score["dispositions"]["final"]["fp"] == 0
    assert score["dispositions"]["false_positives_recovered"] == 1


def test_failsoft_softens_close_to_paused():
    run = _run("reviewer_failsoft")
    turn = run["turns"][1]
    assert turn["proposal"]["disposition"] == "closed"
    assert turn["reviewer_failed"] is True
    assert turn["final_disposition"] == "paused"
    assert turn["outcome"] == "soften_fail_closed_to_paused"


def test_aggregate_shows_precision_recovery():
    runs = [_run(p.stem) for p in SCENARIOS]
    agg = build_report(runs)["summary"]
    assert agg["ok_count"] == len(SCENARIOS)
    assert agg["stage1"]["recall"] == 1.0
    assert agg["final"]["recall"] == 1.0
    assert agg["stage1"]["fp"] == 1
    assert agg["final"]["fp"] == 0
    assert agg["final"]["precision"] == 1.0
    assert agg["false_positives_recovered"] == 1


def test_artifacts_written(tmp_path: Path):
    fixture = ROOT / "fixtures" / "happy_path.json"
    run_scenario_file(
        ROOT / "scenarios" / "happy_path.json",
        FakeLLM.from_fixture(fixture, version="v1"),
        out_dir=tmp_path,
    )
    assert (tmp_path / "happy_path" / "run.json").exists()
    assert (tmp_path / "happy_path" / "turns" / "turn_00.json").exists()
