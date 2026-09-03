"""Score a replayed run against the expectations declared in its scenario.

The headline metric is a confusion matrix over the *leave* class, computed twice
per run: once for the cheap filter's raw proposals (stage 1) and once for the
post-reviewer decisions (stage 2). The difference between them is exactly what
the expensive reviewer buys — recall preserved, precision recovered.
"""

from __future__ import annotations

from ..state.machine import is_leave, normalize


def _rate(numer: int, denom: int) -> float | None:
    return round(numer / denom, 4) if denom else None


def _confusion(truth: list[bool], pred: list[bool]) -> dict:
    tp = sum(1 for t, p in zip(truth, pred, strict=True) if t and p)
    fp = sum(1 for t, p in zip(truth, pred, strict=True) if not t and p)
    fn = sum(1 for t, p in zip(truth, pred, strict=True) if t and not p)
    tn = sum(1 for t, p in zip(truth, pred, strict=True) if not t and not p)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": _rate(tp, tp + fn),
        "precision": _rate(tp, tp + fp),
    }


def _fact_pairs(facts: dict) -> set[str]:
    return {
        f"{str(k).strip().lower()}={str(v).strip().lower()}"
        for k, v in (facts or {}).items()
        if str(k).strip() and str(v).strip()
    }


def score_facts(expected: dict, actual: dict) -> dict:
    exp = _fact_pairs(expected)
    act = _fact_pairs(actual)
    matched = exp & act
    precision = _rate(len(matched), len(act))
    recall = _rate(len(matched), len(exp))
    if precision and recall and (precision + recall):
        f1 = round(2 * precision * recall / (precision + recall), 4)
    else:
        f1 = 0.0 if exp else None
    return {
        "expected_count": len(exp),
        "actual_count": len(act),
        "matched": len(matched),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing": sorted(exp - act),
        "unexpected": sorted(act - exp),
    }


def _check(expected, actual) -> dict | None:
    if expected is None:
        return None
    return {"expected": expected, "actual": actual, "ok": normalize(expected) == normalize(actual)}


def score_run(run: dict) -> dict:
    expected = run.get("expected") or {}
    final = run.get("final") or {}

    checks = {
        "final_disposition": _check(expected.get("final_disposition"), final.get("disposition")),
        "final_step": _check(expected.get("final_step"), final.get("step")),
        "final_turn": (
            None
            if expected.get("final_turn") is None
            else {
                "expected": expected["final_turn"],
                "actual": final.get("turn_count"),
                "ok": expected["final_turn"] == final.get("turn_count"),
            }
        ),
    }

    truth: list[bool] = []
    stage1: list[bool] = []
    finals: list[bool] = []
    per_turn_ok: list[bool] = []
    corrections: list[dict] = []

    for rec in run.get("turns") or []:
        exp = rec.get("expect") or {}
        exp_disp = exp.get("disposition")
        if rec.get("proposal", {}).get("disposition") != rec.get("final_disposition"):
            corrections.append(
                {
                    "turn": rec["turn"],
                    "from": rec["proposal"]["disposition"],
                    "to": rec["final_disposition"],
                    "outcome": rec["outcome"],
                }
            )
        if exp_disp is None:
            continue
        truth.append(is_leave(exp_disp))
        stage1.append(is_leave(rec["proposal"]["disposition"]))
        finals.append(is_leave(rec["final_disposition"]))
        per_turn_ok.append(normalize(exp_disp) == normalize(rec["final_disposition"]))

    stage1_cm = _confusion(truth, stage1)
    final_cm = _confusion(truth, finals)
    dispositions = {
        "labeled_turns": len(truth),
        "stage1": stage1_cm,
        "final": final_cm,
        "reviewer_corrections": corrections,
        "false_positives_recovered": stage1_cm["fp"] - final_cm["fp"],
    }

    facts = score_facts(expected.get("expected_facts") or {}, final.get("facts") or {})

    declared_checks = [c for c in checks.values() if c is not None]
    checks_ok = all(c["ok"] for c in declared_checks)
    turns_ok = all(per_turn_ok) if per_turn_ok else True
    facts_ok = facts["recall"] in (None, 1.0)
    ok = checks_ok and turns_ok and facts_ok

    return {
        "scenario": run.get("scenario"),
        "prompt_version": run.get("prompt_version"),
        "ok": ok,
        "checks": checks,
        "checks_ok": checks_ok,
        "turns_ok": turns_ok,
        "facts_ok": facts_ok,
        "facts": facts,
        "dispositions": dispositions,
    }


def aggregate(scores: list[dict]) -> dict:
    """Roll per-run disposition confusion matrices into one portfolio-level table."""

    def _sum(stage: str) -> dict:
        keys = ("tp", "fp", "fn", "tn")
        tot = {k: sum(s["dispositions"][stage][k] for s in scores) for k in keys}
        tot["recall"] = _rate(tot["tp"], tot["tp"] + tot["fn"])
        tot["precision"] = _rate(tot["tp"], tot["tp"] + tot["fp"])
        return tot

    return {
        "runs": len(scores),
        "ok_count": sum(1 for s in scores if s["ok"]),
        "fail_count": sum(1 for s in scores if not s["ok"]),
        "labeled_turns": sum(s["dispositions"]["labeled_turns"] for s in scores),
        "stage1": _sum("stage1") if scores else {},
        "final": _sum("final") if scores else {},
        "false_positives_recovered": sum(
            s["dispositions"]["false_positives_recovered"] for s in scores
        ),
    }
