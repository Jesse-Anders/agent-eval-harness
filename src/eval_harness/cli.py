"""Command-line entry point: replay | report | compare.

Offline by default (FakeLLM from a fixture). Set ``--provider bedrock`` (or
``EVAL_HARNESS_PROVIDER=bedrock``) to run the same scenarios against a real
model when credentials and the ``[bedrock]`` extra are present.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agents import prompts
from .llm.client import LLMClient
from .llm.fake import FakeLLM
from .replay import report as report_mod
from .replay.runner import run_scenario_file
from .replay.scoring import score_run


def _fixture_for(scenario_path: Path) -> Path:
    return scenario_path.parent.parent / "fixtures" / scenario_path.name


def _make_client(scenario_path: Path, provider: str, version: str) -> LLMClient:
    if provider == "fake":
        fx = _fixture_for(scenario_path)
        if not fx.exists():
            raise SystemExit(f"no fixture for {scenario_path.name} (expected {fx})")
        return FakeLLM.from_fixture(fx, version=version)
    if provider == "bedrock":
        from .llm.bedrock import BedrockLLM  # imported lazily; needs [bedrock] extra

        return BedrockLLM()
    raise SystemExit(f"unknown provider {provider!r}")


def _provider(args) -> str:
    return args.provider or os.environ.get("EVAL_HARNESS_PROVIDER", "fake")


def cmd_replay(args) -> int:
    path = Path(args.scenario)
    client = _make_client(path, _provider(args), args.version)
    run = run_scenario_file(path, client, version=args.version, out_dir=args.out)
    score = score_run(run)
    print(report_mod.render_run(run, score))
    print(f"artifacts: {Path(args.out) / run['scenario']}/")
    return 0 if score["ok"] else 1


def cmd_report(args) -> int:
    scenarios = sorted(Path(args.scenarios).glob("*.json"))
    if not scenarios:
        raise SystemExit(f"no scenarios in {args.scenarios}")
    runs = []
    for path in scenarios:
        client = _make_client(path, _provider(args), args.version)
        runs.append(run_scenario_file(path, client, version=args.version, out_dir=args.out))
    report = report_mod.build_report(runs)
    json_path, md_path = report_mod.write_report(args.out, report)
    print(report_mod.render_report(report))
    print(f"\nwrote {md_path} and {json_path}")
    return 0 if report["summary"]["fail_count"] == 0 else 1


def cmd_compare(args) -> int:
    path = Path(args.scenario)
    versions = [v.strip() for v in args.prompts.split(",") if v.strip()]
    if len(versions) != 2:
        raise SystemExit("--prompts expects exactly two versions, e.g. v1,v2")
    runs = {}
    for v in versions:
        client = _make_client(path, _provider(args), v)  # fresh client per version
        runs[v] = run_scenario_file(path, client, version=v, out_dir=None)

    a, b = versions
    print(f"# Prompt A/B on `{path.stem}`: {a} vs {b}\n")
    turns_a = {t["turn"]: t for t in runs[a]["turns"]}
    turns_b = {t["turn"]: t for t in runs[b]["turns"]}
    changed = 0
    for i in sorted(set(turns_a) | set(turns_b)):
        da = turns_a.get(i, {}).get("final_disposition")
        db = turns_b.get(i, {}).get("final_disposition")
        if da != db:
            changed += 1
            user = (turns_a.get(i) or turns_b.get(i) or {}).get("user", "")
            print(f"- turn {i}: `{da}` → `{db}`  (user: {user[:60]!r})")
    if not changed:
        print("No final decisions changed.")

    def _reviewer_calls(run: dict) -> int:
        return sum(1 for t in run["turns"] if t["reviewer_ran"])

    print(f"\n{changed} decision(s) changed between {a} and {b}.")
    print("\n| version | result | expensive reviewer calls |")
    print("|---|---|---|")
    for v in versions:
        s = score_run(runs[v])
        print(f"| {v} | {'PASS' if s['ok'] else 'FAIL'} | {_reviewer_calls(runs[v])} |")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval-harness")
    parser.add_argument("--provider", choices=["fake", "bedrock"], default=None)
    parser.add_argument("--version", default=prompts.DEFAULT_VERSION, help="prompt version")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_replay = sub.add_parser("replay", help="replay one scenario")
    p_replay.add_argument("--scenario", required=True)
    p_replay.add_argument("--out", default="artifacts")
    p_replay.set_defaults(func=cmd_replay)

    p_report = sub.add_parser("report", help="replay all scenarios and summarize")
    p_report.add_argument("--scenarios", default="scenarios")
    p_report.add_argument("--out", default="artifacts")
    p_report.set_defaults(func=cmd_report)

    p_compare = sub.add_parser("compare", help="A/B two prompt versions on a scenario")
    p_compare.add_argument("--scenario", required=True)
    p_compare.add_argument("--prompts", required=True, help="two versions, e.g. v1,v2")
    p_compare.add_argument("--out", default="artifacts")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
