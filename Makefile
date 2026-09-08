PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install test lint eval eval-baseline analyse layer synth clean

install:
	$(PIP) install -e ".[dev]"

# No credentials, no network. If this needs either, something moved out of
# guide_gap/bedrock.py that should not have.
test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check .

# Costs money: one Bedrock call per coverage check, times the repeat count.
# Embeddings come from .cache, so re-runs only pay for the judgement calls.
eval:
	$(PY) evals/run.py --repeat 3

# Free after the first run. The threshold baseline makes no generation calls at all.
eval-baseline:
	$(PY) evals/run.py --repeat 3 --no-entail --out evals/results-threshold.json

analyse:
	$(PY) scripts/analyse.py --drafts --scores --write

# The Lambda layer: numpy, pyyaml, and the package itself, built for the Lambda
# runtime rather than for this laptop. --platform and --only-binary are both
# required; without them pip installs a macOS wheel that fails at import time in
# Lambda with a message about a missing .so, which is a slow thing to diagnose.
layer:
	rm -rf infra/layer/python
	mkdir -p infra/layer/python
	$(PIP) install \
		--platform manylinux2014_x86_64 \
		--implementation cp \
		--python-version 3.12 \
		--only-binary=:all: \
		--target infra/layer/python \
		numpy pyyaml
	cp -r guide_gap infra/layer/python/guide_gap
	touch infra/layer/python/.gitkeep

synth: layer
	npx --yes aws-cdk@2 synth

clean:
	rm -rf infra/layer/python out/drafts .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
