.PHONY: install relock lint format test env docs golden clean

# uv is the only Python tool chain on this box (AGENTS.md §5). Resolve it rather
# than assume PATH: a Makefile invoked from a stripped environment must not
# silently fall back to a system pip.
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

# Prefer the project venv's binaries over whatever PATH happens to hold. The
# DONE-WHEN gate is "`make lint test` green in a clean clone", and a clean clone
# has not sourced any activate script — a Makefile that calls a bare `pytest`
# either fails there or, worse, silently tests against some other environment's
# packages. Falls back to PATH so `make test` still works inside an already
# activated venv.
VENV := .venv
RUFF := $(if $(wildcard $(VENV)/bin/ruff),$(VENV)/bin/ruff,ruff)
PYTEST := $(if $(wildcard $(VENV)/bin/pytest),$(VENV)/bin/pytest,pytest)
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

# ACCEL selects the accelerator extra, and has NO DEFAULT ON PURPOSE. One
# lockfile serves two hosts that are both linux-x86_64 — the local gfx1151/ROCm
# box and the EPYC campaign pod — so nothing about the machine distinguishes
# them at resolve time. A default here would install one host's wheels on the
# other silently, which is the exact failure this split exists to prevent, so
# the choice is made out loud or not at all.
#
#   local box     make install ACCEL=rocm
#   campaign pod  make install ACCEL=cpu
#
# ACCEL=cuda is registered and deferred; see pyproject.toml.
ACCEL_VALID := rocm cpu cuda
ifneq ($(filter $(MAKECMDGOALS),install relock),)
  ifeq ($(ACCEL),)
    $(error ACCEL is unset. Choose the accelerator extra explicitly: make $(MAKECMDGOALS) ACCEL=rocm (local box) or ACCEL=cpu (campaign pod))
  endif
  ifeq ($(filter $(ACCEL),$(ACCEL_VALID)),)
    $(error ACCEL=$(ACCEL) is not one of: $(ACCEL_VALID))
  endif
endif

# --frozen: install exactly what uv.lock pins, never re-resolve. An unfrozen
# install makes "green in a clean clone" mean "green against whatever versions
# existed this morning", which is not the claim the gate is supposed to make.
# Use `make relock` to change dependencies on purpose.
install:
	$(UV) sync --frozen --extra $(ACCEL) --extra dev

relock:
	$(UV) lock
	$(UV) sync --frozen --extra $(ACCEL) --extra dev

lint:
	$(UV) lock --check
	$(RUFF) check src/ tests/ scripts/
	$(RUFF) format --check src/ tests/ scripts/

format:
	$(RUFF) format src/ tests/ scripts/
	$(RUFF) check --fix src/ tests/ scripts/

test:
	$(PYTEST) tests/ -q

# Torch/device sanity. Not part of `make test` — tests must pass CPU-only and
# must never depend on GPU availability (AGENTS.md §5).
env:
	$(PYTHON) scripts/check_env.py

# Generated reference docs. docs/vocab.md is regenerated and compared by
# tests/test_vocab.py, so a vocabulary change that forgets this target fails the
# build instead of leaving a plausible, wrong reference behind.
docs:
	$(PYTHON) scripts/dump_vocab.py
	$(PYTHON) scripts/render_derivations.py

# The loop's liveness check: does it still turn, in the time a person will wait?
# CPU always — a check that needs the accelerator stops being run the moment the
# accelerator is busy, and this box's GPU is shared.
golden:
	$(PYTHON) scripts/golden.py

clean:
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# `make bench` arrives with the first benchmark script (chunk 7 — the search
# core is the first thing on this project worth profiling). Declaring an empty
# target now would only teach the habit of running something that measures
# nothing.
