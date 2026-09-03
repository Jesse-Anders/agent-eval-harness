"""Render scored runs as JSON and a readable Markdown/console summary."""

from __future__ import annotations

import json
from pathlib import Path

from .scoring import aggregate, score_run


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _confusion_table(stage1: dict, final: dict) -> str:
    rows = [
        "| stage | tp | fp | fn | tn | recall | precision |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, m in (("stage 1 (cheap filter)", stage1), ("after reviewer", final)):
        rows.append(
            f"| {label} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
            f"{_pct(m['recall'])} | {_pct(m['precision'])} |"
        )
    return "\n".join(rows)


def render_run(run: dict, score: dict) -> str:
    lines: list[str] = []
    status = "PASS" if score["ok"] else "FAIL"
    lines.append(f"## `{run['scenario']}` — {status}  (prompts {run['prompt_version']})")
    lines.append("")
    fin = run["final"]
    lines.append(
        f"- final: disposition=`{fin['disposition']}` step=`{fin['step']}` "
        f"turns=`{fin['turn_count']}`"
    )
    corr = score["dispositions"]["reviewer_corrections"]
    if corr:
        lines.append("- reviewer corrections:")
        for c in corr:
            lines.append(f"    - turn {c['turn']}: `{c['from']}` → `{c['to']}` ({c['outcome']})")
    f = score["facts"]
    if f["expected_count"]:
        lines.append(
            f"- fact alignment: f1=`{_pct(f['f1'])}` "
            f"(recall {_pct(f['recall'])}, precision {_pct(f['precision'])})"
            + (f"; missing {f['missing']}" if f["missing"] else "")
        )
    d = score["dispositions"]
    if d["labeled_turns"]:
        lines.append("")
        lines.append(_confusion_table(d["stage1"], d["final"]))
    lines.append("")
    return "\n".join(lines)


def build_report(runs: list[dict]) -> dict:
    scored = [score_run(r) for r in runs]
    return {
        "summary": aggregate(scored),
        "runs": [{"run": r, "score": s} for r, s in zip(runs, scored, strict=True)],
    }


def render_report(report: dict) -> str:
    agg = report["summary"]
    lines = ["# Replay report", ""]
    lines.append(
        f"**{agg['ok_count']}/{agg['runs']} scenarios pass** · "
        f"{agg['labeled_turns']} labeled turns · "
        f"{agg['false_positives_recovered']} false-positive leave(s) recovered by the reviewer"
    )
    lines.append("")
    lines.append("### Leave-detection accuracy (all scenarios)")
    lines.append("")
    lines.append(_confusion_table(agg["stage1"], agg["final"]))
    lines.append("")
    lines.append(
        "> The cheap filter keeps recall high (misses no real leave) but admits "
        "false positives; the reviewer removes them, lifting precision without "
        "touching recall."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    for entry in report["runs"]:
        lines.append(render_run(entry["run"], entry["score"]))
    return "\n".join(lines).rstrip() + "\n"


def write_report(out_dir: str | Path, report: dict) -> tuple[Path, Path]:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / "report.json"
    md_path = base / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_report(report), encoding="utf-8")
    return json_path, md_path
