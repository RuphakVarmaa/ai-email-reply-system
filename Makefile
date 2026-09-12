SHELL := /bin/bash
PYTHON := PYTHONPATH=src .venv/bin/python

.PHONY: setup preprocess eval-template eval-nn eval-llm eval-llm-small test clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet numpy scikit-learn pyyaml requests pytest pytest-asyncio
	@echo "✅ Virtual environment ready."

preprocess:
	$(PYTHON) scripts/preprocess.py

eval-template:
	$(PYTHON) -m email_reply.pipeline --mode template --n 220

eval-nn:
	$(PYTHON) -m email_reply.pipeline --mode nn --n 220

eval-llm-small:
	$(PYTHON) -m email_reply.pipeline --mode llm --n 10 --judge

eval-llm:
	$(PYTHON) -m email_reply.pipeline --mode llm --n 30 --judge

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

clean:
	rm -rf data/processed/ data/cache/ reports/
