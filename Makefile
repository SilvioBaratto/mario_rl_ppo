# ---------------------------------------------------------------------------
# Mario RL PPO — build targets
# ---------------------------------------------------------------------------
# Usage:
#   make                  → show this help
#   make install          → install the pinned runtime and the dev tools
#   make lint             → ruff check + format check + pyright
#   make test-fast        → the tests that need no emulator
#   make test-emulator    → the short emulator tests (raw routes, wrapped env, whole game)
#   make test-slow        → every NOOP offset, vec envs, checkpoint and training goldens
#   make test             → everything
#   make smoke            → a 2,048-step training run on two workers
#   make docker-check     → build the image and check routes, permissions and compose
# ---------------------------------------------------------------------------

.DEFAULT_GOAL := help
.PHONY: help install lint test-fast test-emulator test-slow test smoke docker-check

help:
	@echo ""
	@echo "Mario RL PPO — available targets"
	@echo "--------------------------------"
	@echo "  make install         Install requirements.txt and the dev tools"
	@echo "  make lint            ruff check + ruff format --check + pyright"
	@echo "  make test-fast       Tests that need no emulator"
	@echo "  make test-emulator   Short emulator tests (what CI runs besides test-fast)"
	@echo "  make test-slow       Full NOOP-offset sweeps, vec envs, checkpoint and training goldens"
	@echo "  make test            The whole suite"
	@echo "  make smoke           2,048-step training run on two workers"
	@echo "  make docker-check    Build the image and check demo routes, permissions and compose"
	@echo ""

# On Linux the default torch wheel pulls several GB of CUDA libraries; the CPU wheel is enough
# for CI and containers. On macOS the default wheel is the right one.
install:
	pip install -r requirements.txt
	pip install "ruff>=0.17,<0.18" "pyright>=1.1"

lint:
	ruff check .
	ruff format --check .
	pyright

test-fast:
	pytest -q -m "not emulator and not slow and not integration and not local_golden"

test-emulator:
	pytest -q -m "emulator and not slow and not integration and not local_golden"

test-slow:
	pytest -q -m "slow or integration or local_golden"

test:
	pytest -q

smoke:
	python train.py --total-timesteps 2048 --num-envs 2 --eval-freq 0 --device cpu \
		--exp-name smoke
	test -f runs/smoke/checkpoints/final_model.zip

docker-check:
	docker build -t mario-rl-ppo .
	docker run --rm mario-rl-ppo
	docker run --rm mario-rl-ppo python -c "import sys; sys.path.insert(0, 'src'); \
from environments.env_factory import load_demo_route; \
assert all(load_demo_route(f'SuperMarioBros-1-{s}-v0') for s in (1, 2, 3, 4))"
	docker run --rm mario-rl-ppo sh -c 'touch /app/.w && rm /app/.w'
	docker compose config -q
