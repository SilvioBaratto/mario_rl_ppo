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
- `LICENSE` (MIT), `CODE_OF_CONDUCT.md`, issue and pull request templates.

### Changed

- The code is formatted with `ruff format` (line length 100) and passes `ruff check`; type
  annotations use builtin generics and `X | None`.

### Fixed

- The container's `/app` is writable by the non-root user, so `evaluate` and `record` can write
  their outputs; the compose `evaluate` and `record` services now write into mounted folders.

[Unreleased]: https://github.com/SilvioBaratto/mario_rl_ppo/commits/main
