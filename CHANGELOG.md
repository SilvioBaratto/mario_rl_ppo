# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `tests/`: a behaviour-preservation suite. Golden traces captured from the unmodified code pin
  the public API and command-line defaults, the demonstration routes, the wrapped training
  environment for every NOOP offset on 1-1 to 1-4, the chained whole-game run, vec-env rollouts,
  behaviour-cloning datasets, the reference checkpoint and a short training run with resume.
  Unit tests cover levels, the reverse curriculum, episode progress, the wrappers, frontier
  restarts on a fake emulator, the route files, the CNN and the logger.
- `pyproject.toml` with pytest, ruff and pyright configuration, a `Makefile` (`lint`,
  `test-fast`, `test-emulator`, `test-slow`, `smoke`, `docker-check`) and a GitHub Actions
  workflow that runs lint, type-check, tests, a training smoke run and the container checks.
- `LICENSE` (MIT), `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, issue and pull request templates.
- `CLAUDE.md`, the design notes for the emulator pipeline, the curriculum and the checkpoints, is
  now tracked (it was git-ignored) and describes the new layout and the test tiers.
- `mario.infrastructure.nes.NesEmulator`, a protocol listing the emulator internals the code relies
  on, and `create_mario_env` overloads; `pyright` reports no findings and gates CI.
- `mario --version`, and `mario.__version__` read from the installed metadata.
- A ruff rule that bans a plain `PPO.load`: saved checkpoints only load through
  `load_mario_model`.

### Changed

- The code is an installable package, `mario`, split into `domain` (pure rules), `application`
  (use cases), `infrastructure` (wrappers, vec envs, model, files) and `cli`; the 1,049-line
  `env_factory.py` is gone. `tests/test_layering.py` enforces the dependency rule.
- One command replaces the four scripts: `mario train`, `mario evaluate`, `mario record` and
  `mario find-route` (also `python -m mario ...`). The flags and their defaults are unchanged.
  `train.py`, `evaluate.py`, `play_and_record.py`, `find_route.py` and `setup.py` are removed.
- Demonstration routes are package data under `mario/infrastructure/demo_routes/`; `pyproject.toml`
  replaces `setup.py` and carries the pinned dependencies.
- The defaults that were retyped in many places (random-start frames 30, stall cutoff 150, frame
  stack 4, observation size 84, the PPO hyperparameters) are defined once in
  `mario.domain.config`; the commands and the environment builders read them from there. The
  values are unchanged.
- The code is formatted with `ruff format` (line length 100) and passes `ruff check`; type
  annotations use builtin generics and `X | None`.

### Fixed

- The README no longer claims YAML configs, Weights & Biases, a generated `config.yaml` or an
  example evaluation that was never produced by the code; its results table comes from a real run.
- Requirements no longer pin packages the code never imports (`wandb`, `pyyaml`, `torchvision`,
  `black`, `flake8`); development tools are the `dev` extra.
- The container's `/app` is writable by the non-root user, so `evaluate` and `record` can write
  their outputs; the compose `evaluate` and `record` services now write into mounted folders.

[Unreleased]: https://github.com/SilvioBaratto/mario_rl_ppo/commits/main
