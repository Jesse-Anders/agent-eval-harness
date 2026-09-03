.PHONY: help install test lint replay report compare clean

PY ?= python
SCENARIO ?= scenarios/topic_switch.json
OUT ?= artifacts

help:
	@echo "Targets:"
	@echo "  make install              Install package + dev deps (editable)"
	@echo "  make test                 Run pytest (offline, FakeLLM, no key needed)"
	@echo "  make lint                 Run ruff"
	@echo "  make replay SCENARIO=...  Replay one scenario (fake LLM by default)"
	@echo "  make report               Replay all scenarios and print the summary"
	@echo "  make compare SCENARIO=... PROMPTS=v1,v2   A/B a prompt version over a scenario"
	@echo ""
	@echo "  Set EVAL_HARNESS_PROVIDER=bedrock (with AWS creds + pip install -e .[bedrock])"
	@echo "  to run replay/compare against a real model instead of the fake."

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

replay:
	$(PY) -m eval_harness.cli replay --scenario $(SCENARIO) --out $(OUT)

report:
	$(PY) -m eval_harness.cli report --scenarios scenarios --out $(OUT)

compare:
	$(PY) -m eval_harness.cli compare --scenario $(SCENARIO) --prompts $(PROMPTS) --out $(OUT)

clean:
	rm -rf $(OUT) .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
