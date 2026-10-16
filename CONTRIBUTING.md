# Contributing

Thanks for your interest in improving **mario_rl_ppo**, a PPO agent that learns Super Mario Bros
from pixels. The project is a research codebase: results have to be reproducible, so changes are
held to "prove it did not move" as much as "prove it works".

## Getting set up

```bash
git clone https://github.com/SilvioBaratto/mario_rl_ppo
cd mario_rl_ppo
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install
```

Python 3.13 is required (gym-super-mario-bros 9.x and nes-py 9.x). On Linux install the CPU torch
wheel first, as the [README](README.md#installation) shows.

## The development loop (test-first)

Every change earns a test. The loop we follow for each task:

1. **RED**: write a failing test that pins the behaviour you want.
2. **GREEN**: write the minimum code to make it pass.
3. **Regression**: run the whole suite and keep it green.
4. **Check**: `make lint` (`ruff check`, `ruff format --check`, `pyright`).
5. **Commit**: one focused commit per logical change.

```bash
make test-fast       # no emulator, about a second
make test-emulator   # raw routes, the wrapped env, the whole game; under a minute
make test-slow       # every NOOP offset, worker processes, the checkpoint, a training run; minutes
```

## Behaviour-preserving changes and the golden traces

`tests/golden` pins what training, evaluation and checkpoint loading do. A refactor must leave it
untouched: if a golden fails, the change moved behaviour, whatever it was meant to do.

- Do not regenerate a golden inside a refactor commit. If behaviour has to change on purpose, change
  the golden in its own commit and explain why (`python -m tests.capture_golden <name>`; it keeps a
  golden only when two runs agree).
- Tests reach the code through `tests/adapter.py`. A module move changes that file and nothing else.
- Pixel and tensor hashes are compared only on the platform that captured them; everything else must
  match everywhere.

## Architecture: the dependency rule

The package is organised in layers; **dependencies always point inward**, and no two layers import
each other:

```
cli  ->  application  ->  domain
 |                          ^
 +--> infrastructure  ------+
```

- `domain/`: pure rules (constants, levels, route arithmetic, the curriculum, episode summaries).
  Plain Python plus numpy. No emulator, no framework, no file access.
- `application/`: use cases that take their collaborators as arguments (`evaluate_agent`,
  `record_game`). Depends only on `domain`.
- `infrastructure/`: adapters: the Gymnasium wrappers, vec envs, the policy network, file and
  route I/O, SB3 callbacks, route search. Depends only on `domain`.
- `cli/`: argument parsing and the wiring of concrete adapters into use cases. The one place that
  may import everything, and the one place that sets the SDL variables, before importing torch or
  the emulator.

The rule is not a convention: `tests/test_layering.py` parses every module's imports and fails the
build on a forbidden one. Put pure logic in `domain`, a new use case in `application`, and anything
that touches the emulator, the disk or a framework in `infrastructure`.

## Things that must not change by accident

- The observation pipeline (wrapper order, 4×84×84 uint8, the blanked status bar) and the action
  set are part of every saved checkpoint. `CLAUDE.md` lists the reasons.
- Load checkpoints with `mario.infrastructure.model.load_mario_model`. A plain `PPO.load` fails on
  every old checkpoint, and ruff rejects it.
- `*.json` is git-ignored in general; the demonstration routes and `tests/**/*.json` are the
  exceptions. Check `git status` shows a new route or golden before you commit.

## Numbers are never written by hand

Every number in the README comes from a run of the command next to it. If a change moves a number,
re-run the command and update the table from its output.

## Commit and PR style

- [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): lowercase imperative
  summary`, with `feat`, `fix`, `docs`, `test`, `perf`, `refactor`, `style`, `build`, `ci`, `chore`.
- A body that says *why*, with the evidence (what you ran, what it showed), wrapped near 72 columns.
- Topic branches named `<type>/<kebab-name>`, merged into `main` through a pull request so CI runs
  and the checklist applies.
- Fill in the PR checklist; make sure `make lint` and the test tiers pass.

By contributing you agree that your work is licensed under the project's [MIT License](LICENSE).
