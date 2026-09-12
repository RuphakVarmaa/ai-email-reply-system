SHELL := /bin/bash
PYTHON := PYTHONPATH=src .venv/bin/python

.PHONY: setup build-dataset demo run ablate validate-metric test clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet numpy scikit-learn pyyaml requests pytest pytest-asyncio
	@echo "✅ Virtual environment ready."

build-dataset:
	$(PYTHON) -m email_reply build

demo:
	$(PYTHON) -m email_reply demo

run:
	$(PYTHON) -m email_reply run --mode rag

run-small:
	$(PYTHON) -m email_reply run --mode rag --n 5

ablate:
	@echo "Running ablation: zero_shot"
	$(PYTHON) -m email_reply run --mode zero_shot --n 12
	@echo "Running ablation: no_kb"
	$(PYTHON) -m email_reply run --mode no_kb --n 12
	@echo "Running ablation: nn"
	$(PYTHON) -m email_reply run --mode nn --n 12

validate-metric:
	$(PYTHON) -m email_reply.eval.validate_runner

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

clean:
	rm -rf data/corpus/ data/cache/ reports/ data/dataset_manifest.json
