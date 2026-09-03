from eval_harness.replay.scoring import _confusion, aggregate, score_facts, score_run


def test_confusion_counts():
    truth = [True, True, False, False]
    pred = [True, False, True, False]
    c = _confusion(truth, pred)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)
    assert c["recall"] == 0.5
    assert c["precision"] == 0.5


def test_score_facts_partial():
    f = score_facts({"a": "1", "b": "2"}, {"a": "1", "c": "3"})
    assert f["matched"] == 1
    assert f["recall"] == 0.5
    assert f["precision"] == 0.5
    assert f["missing"] == ["b=2"]
    assert f["unexpected"] == ["c=3"]


def test_score_facts_case_insensitive():
    f = score_facts({"os": "macOS"}, {"os": "macos"})
    assert f["recall"] == 1.0


def test_score_facts_empty_expected():
    f = score_facts({}, {"x": "1"})
    assert f["recall"] is None
    assert f["f1"] is None


def test_score_run_flags_disposition_mismatch():
    run = {
        "scenario": "x",
        "final": {"disposition": "closed", "step": "resolved", "turn_count": 1, "facts": {}},
        "expected": {"final_disposition": "continue"},
        "turns": [
            {
                "turn": 0,
                "proposal": {"disposition": "closed"},
                "final_disposition": "closed",
                "outcome": "confirm_closed",
                "expect": {"disposition": "continue"},
            }
        ],
    }
    score = score_run(run)
    assert score["ok"] is False
    assert score["checks"]["final_disposition"]["ok"] is False
    assert score["turns_ok"] is False


def test_aggregate_empty():
    assert aggregate([]) == {
        "runs": 0,
        "ok_count": 0,
        "fail_count": 0,
        "labeled_turns": 0,
        "stage1": {},
        "final": {},
        "false_positives_recovered": 0,
    }
