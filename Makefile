# ---------------------------------------------------------------------------
# Mario RL PPO — build targets
# ---------------------------------------------------------------------------
# Usage:
#   make                  → show this help
#   make test-fast        → the tests that need no emulator
#   make test-emulator    → the short emulator tests (raw routes, wrapped env, whole game)
#   make test-slow        → every NOOP offset, vec-env rollouts, policy and training goldens
#   make test             → everything
# ---------------------------------------------------------------------------

.DEFAULT_GOAL := help
.PHONY: help test-fast test-emulator test-slow test

help:
	@echo ""
	@echo "Mario RL PPO — available targets"
	@echo "--------------------------------"
	@echo "  make test-fast       Tests that need no emulator"
	@echo "  make test-emulator   Short emulator tests (what CI runs besides test-fast)"
	@echo "  make test-slow       Full NOOP-offset sweeps, vec envs, policy and training goldens"
	@echo "  make test            The whole suite"
	@echo ""

test-fast:
	pytest -q -m "not emulator and not slow and not integration and not local_golden"

test-emulator:
	pytest -q -m "emulator and not slow and not integration and not local_golden"

test-slow:
	pytest -q -m "slow or integration or local_golden"

test:
	pytest -q
