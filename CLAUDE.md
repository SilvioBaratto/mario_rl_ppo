# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PPO agent (Stable Baselines3) that learns Super Mario Bros from pixels via `gym-super-mario-bros`. An installable package, `mario`, with one command (`mario train | evaluate | record | find-route`, also `python -m mario`) on four layers: `domain` (pure rules), `application` (use cases), `infrastructure` (wrappers, vec envs, model, files) and `cli`. `tests/` guards behaviour against golden traces (see Testing).

## Commands

Install once with `make install` (the pinned `requirements.txt`, then `pip install -e ".[dev]"`); the `mario` command and `import mario` then work from anywhere. Layer rule: `domain` imports nothing from the package, `application` and `infrastructure` import only `domain`, `cli` may import all (`tests/test_layering.py`).

```bash
pip install -r requirements.txt   # exact pins; Python >= 3.13 (gym-super-mario-bros 9.x / nes-py 9.x)

# Train (defaults: SuperMarioBros-1-1-v0, 8 envs, batch 256, 10 epochs, device auto)
mario train --total-timesteps 10000000 --exp-name my_exp
# Multi-level: one policy on several levels (use a multiple of the level count for --num-envs)
mario train --levels 1-1,1-2,1-3 --num-envs 9 --exp-name world1_multi
# Fast smoke run: two rollouts, no eval env (--eval-freq 0 skips EvalCallback)
mario train --total-timesteps 2048 --num-envs 2 --eval-freq 0 --exp-name smoke
# Resume: --total-timesteps is the overall target; the step counter carries over
mario train --resume runs/my_exp/checkpoints/checkpoint_5000000 --total-timesteps 10000000 --exp-name my_exp_cont

# Evaluate (also: --render, --record-video --video-dir DIR, --stochastic, --noop-max,
#   --levels 1-1,1-2,1-3,1-4 for a per-level summary; JSON is keyed by env id under "levels")
#   With --env-id SuperMarioBros-v0 each episode also reports levels cleared and lives lost (GameProgress)
# Writes eval_results_ppo_<timestamp>.json to the cwd
mario evaluate --model-path runs/my_exp/checkpoints/best_model --n-episodes 10

# Search the emulator for a demonstration route (Go-Explore style, shortest flag route per search;
#   refuses to overwrite without --force). 1-2 used --searches 24 --budget 150000
mario find-route --level 1-4 --searches 21

# Record one episode to MP4 (--max-steps is also the env time limit)
mario record --model-path runs/my_exp/checkpoints/best_model
# Whole World 1 in one video: --env-id SuperMarioBros-v0 --stop-at-level 2-1

tensorboard --logdir runs/my_exp/logs

# Lint, format check, types (ruff + pyright; config in pyproject.toml) and the tests
make lint
make test-fast        # no emulator;  make test-emulator, make test-slow for the rest
```

Model paths work with or without `.zip` (`resolve_model_path`).

Docker: `docker-compose build`, `docker-compose up train`; other services are behind profiles (`--profile resume|eval|record|monitor`). The image (`python:3.13-slim`) installs CPU-only torch wheels, sets SDL to dummy drivers and runs as non-root user `mario`; `./runs` is bind-mounted. `.dockerignore` keeps `runs/`, `.git` and checkpoints out of the build context.

## Current best model

`runs/world1_game3/checkpoints/checkpoint_17894136` (17.9M steps) clears World 1 (1-1 to 1-4) in the whole game without losing a life: greedy 31/31 NOOP offsets, stochastic 15/30; single-stage greedy 1-1 20/20, 1-2 18/20, 1-3 20/20, 1-4 18/20. Recorded with `mario record --env-id SuperMarioBros-v0 --stop-at-level 2-1` (`videos/world1_no_death.mp4`). Trained with `--levels 1-1,1-2,1-2,1-3,1-4,game,game,game,game,game --num-envs 10 --frontier-starts 0.5 --demo-starts --demo-bc 10`; checkpoints after it in that run used an earlier SkipFrame.

## Architecture

### Environment pipeline (`mario/infrastructure/{wrappers,frontier,factory}.py`)

`create_mario_env` builds, in this load-bearing order:

`gym_super_mario_bros.make` (alias of `gymnasium.make`) → `JoypadSpace(SIMPLE_MOVEMENT)` (7 actions) → `NoopReset(30)` → `FrontierStart` (only with `--frontier-starts`) → `CustomReward` → `SkipFrame(4)` → `StallTruncation(150)` → `TimeLimit(2000)` → `GrayResizeObservation(84)` (cv2, uint8; the top `HUD_ROWS` = 32 raw rows, the status bar, are blacked out first)

`create_vec_env` accepts one env id or a list (multi-level training): `assign_levels` gives worker i `env_ids[i % len]`, so every rollout mixes levels and repeating an id weights that level (also in evaluation and `best_model` selection); `parse_levels('1-1,1-2')` builds the ids. `create_eval_env` with a list makes one in-process env per level, and `mario train` sets `n_eval_episodes = --eval-episodes × levels` so SB3's `evaluate_policy` covers each level equally.

`create_vec_env` wraps N of these in `SubprocVecEnv` (`DummyVecEnv` when `num_envs == 1`) → `VecMonitor` → `VecFrameStack(4, channels_order='last')`, giving `(84, 84, 4)` uint8 observations. Passing that env to `PPO` makes SB3 add `VecTransposeImage` (channels-first) and scale to `[0, 1]` on the training device. `create_eval_env` applies `VecTransposeImage` itself so `EvalCallback` sees the same env type.

- Everything below the VecEnv is native **Gymnasium** (5-tuple `step` with `terminated`/`truncated`, `reset` returns `(obs, info)`); there is no `gym` or shimmy. The commands consume SB3 vec envs, which return the 4-tuple `(obs, reward, done, info)`, and SB3 maps `truncated` to `info["TimeLimit.truncated"]` (bootstrapped correctly by PPO).
- **Env id semantics differ:** single-stage ids (`SuperMarioBros-W-S-vN`, the default `1-1`) terminate on death or flag; `SuperMarioBros-v0` is the whole game, one episode spanning 3 lives and level transitions.
- `CustomReward` detects events from `info`, not from termination, so it is correct for both: +50 on the `flag_get` rising edge, −50 per life lost (`life` decreasing, or termination without flag — the last death wraps `life` 0→255). World/stage for the 4-4 / 7-4 maze penalties come from `info` too; `current_x` resets on respawn or level change so the 7-4 backtrack check doesn't misfire. Reward is scaled /10 (~300 is strong).
- `StallTruncation` cuts an episode (as truncation, so PPO bootstraps) after 150 agent steps without a new furthest x, tracked per (world, stage, life) segment. Stuck greedy episodes otherwise ran the full 2000 steps and made late-run evaluation cost more than training.
- `FrontierStart` (training only, `frontier_prob` > 0) snapshots the emulator with nes-py 9's `dump_state()` while Mario is grounded (RAM 0x001D == 0, not dying), one snapshot per 32 px of x per level, and on a fraction of resets restores one 50-400 px behind the level's frontier (median furthest x of the last 20 episodes). After `load_state` it must call the gsmb env's `_did_reset()`: gym-super-mario-bros keeps reward trackers (last time, max x, score, coins) in Python and they'd otherwise produce a bogus first reward. It sits inside `CustomReward`/`StallTruncation` so their `reset()` re-reads the restored `info`.
- Demonstration routes (`mario/infrastructure/demo_routes/<W>-<S>.json`, loaded by `load_demo_route`) are frame-level joypad bytes from a fresh NOOP-free reset plus waypoints `{frame, x}`; the last waypoint's x is the curriculum goal. With `--demo-starts`, frontier restarts on a level with a route run a reverse curriculum. The controller, `ReverseCurriculum`, lives in the main process (`ReverseCurriculumCallback` in `mario/infrastructure/callbacks.py`) because native snapshots can't be pickled: each worker replays its route once (`replay_demo_route`) and keeps, for every `DEMO_STRIDE` (8)-th frame, the state `demo_warmup_frames(frame_stack)` (12) + 1 frames earlier plus the route bytes up to the frame, verifying waypoint x (a mismatch disables the route with a warning). A restore plays the first byte to redraw the screen, then `FrontierStart.step` forces the remaining 12 bytes over the agent's actions (3 agent steps). Why: `VecFrameStack` zero-fills on reset, and from a zero-padded stack the policy behaves as at a level start and releases A; one frame without A mid-ascent makes SMB1 lock in falling gravity for the rest of the jump, so without warm-up every mid-jump start fails regardless of skill (the curriculum stalled at the first rising state). For the same reason demo restores never use a NOOP redraw frame. Workers report each demo episode as `info['demo_outcome']` on the *next* reset, read from `training_env.unwrapped.reset_infos` (VecEnv wrappers keep their own stale copy; SubprocVecEnv also resends the last reset info on non-done steps, so check `dones`). The controller pushes the start frame to workers with `env_method('set_demo_cursor', level, frame)` (found through Gymnasium's `get_wrapper_attr`). Curriculum position isn't checkpointed. Re-record a route if the emulator or the level-start sequence changes. The 1-3 route runs to the flag (219 agent steps; `frames` is `agent_actions` expanded ×4 and cut at the flag frame, so both forms are the same trajectory): the short hop onto the intermediate tree-top, then no A while landing and a fresh 7-step A press for the gap jump (SMB1 only jumps on a fresh press). End routes on a survivable state: one ending at full speed near a tree edge taught the agent to run into the pit. Its open-loop actions only work after even NOOP offsets (odd ones die at x=1724; 4-frame agent steps fix an episode's frame parity), so `build_demo_dataset` keeps 16 of 31 offsets. The 1-1 route (270 agent steps, flag after all 31 NOOP offsets) is agent-derived: one stochastic flag run of `runs/world1_flag` from a NOOP-free reset, the most offset-robust then shortest of 45. It replaced a `mario find-route` route (406 steps) that held run-right against the first pipe for ten steps before backing off; BC cloned that into a greedy policy that stalled at the pipe (x=435) on every 1-1 episode. Search routes can carry such aimless stretches; check a route's steps per 100 px before training on it. `mario find-route` therefore spends each search's whole budget and keeps its shortest flag route instead of the first: the first 1-2 route (524 steps) wandered 74 steps among the block columns at x 200-400, and the greedy agent looped there (x=386) in every whole-game run. The 1-2 route is now that search's output (24 searches of 150k steps: 363 steps, longest stretch without progress 8 steps, flag after 26/31 NOOP offsets). The 1-4 route (to the axe; `flag_get` is world-over in castles) is `mario find-route` output (byte-identical on re-run with the same seeds/budget) and works after 28 of 31 offsets. nes-py snapshots are not complete there: restoring a 1-4 state with identical RAM can still diverge hundreds of frames later (cause outside RAM, not identified; 1-3 restores are exact). The curriculum only needs playable states, so it is unaffected; BC data comes from full replays, never restores.
- Behaviour cloning (`mario/infrastructure/cloning.py`, `--demo-bc N`): routes may carry `agent_actions` (SIMPLE_MOVEMENT index per agent step; 1-3's was derived from the frame bytes and verified to reach the last waypoint after every 0-30 NOOP offset). `build_demo_dataset` replays them through `create_mario_env` and stacks/transposes like `VecFrameStack` + `VecTransposeImage` (offset 0 is byte-identical to the VecEnv pipeline). `DemoCloningCallback` runs N cross-entropy steps with the policy's optimizer in `_on_rollout_start`, i.e. after the previous update and before sampling, so PPO's stored log-probs match the policy it trains. Needed because the curriculum cannot cross a decision the policy gives ~0 probability (1-3 frame 184: release A before landing, then a fresh press).
- `NoopReset` exists because the NES emulator is deterministic: without random starts, every deterministic-policy episode is identical. Its RNG is seeded by `reset(seed=...)`; `create_mario_env` seeds env `i` with `seed + i`.
- `CustomReward` sits inside `SkipFrame`, so shaping runs per raw NES frame; `TimeLimit` counts agent steps (post-skip).
- Whole-game level transitions (`SuperMarioBros-v0`) are made to look like single-stage resets, because single-stage training does not transfer otherwise (a policy clearing single-stage 1-2 18/20 failed it in 40/40 whole-game runs). `SkipFrame` sends no buttons from the frame the flag is touched until the next level starts (the game ignores input then, except on the frame the new level begins: the agent's last 1-1 action used to start 1-2 with Mario at speed 2 and A held, and SMB1 only jumps on a fresh press), and ends the agent step on the frame `(world, stage)` changes with `info['level_start']`; `LevelStartFrameStack` (used by `create_vec_env`) then zero-fills that env's stacked frames as on reset. Single-stage episodes end at the flag, so training on them is unaffected. Even so, a whole-game entry is not any single-stage reset plus NOOPs (1-2 is entered a few frames further into Mario's drop than the enemy phase implies), so with `game` in `--levels`, `mario train` adds `build_game_demo_dataset`: the world-1 routes chained through the whole game, pairs from the levels after 1-1 as a run enters them (1-2 only: the 1-3 route needs even frame parity and the chain enters 1-3 on odd). In v0 a level can end inside the agent step that touched the flag, so no returned info shows `flag_get`; check for a level change too. Greedy whole-game episodes are deterministic once 1-1 is: every NOOP offset reaches 1-2 in the same state, so later levels pass or fail for all 31 offsets together; also run `mario evaluate --env-id SuperMarioBros-v0 --stochastic`. nes-py restores of whole-game states are not exact either, so test whole-game routes by replaying from the game start, not from a restored entry.
- `SubprocVecEnv` starts workers with `forkserver` (macOS), which re-imports `__main__`: any entry point that builds a multi-env VecEnv needs an `if __name__ == "__main__":` guard (`mario/__main__.py` and the console script have one). Workers ignore SIGINT (set in `make_env` when `multiprocessing.parent_process()` is set) so Ctrl+C is handled only by the main process.

### Model (`mario/infrastructure/model.py`)

`MarioCNN` is an SB3 `BaseFeaturesExtractor`: 4× Conv2d(32, 3×3, stride 2) → Linear 512 (SB3's `ortho_init` initializes it). It infers channels-first vs channels-last with the heuristic `obs_shape[0] <= 4`; channels-first is the normal, fast path (permuting channels-last input costs ~3× on MPS).

`mario_policy_kwargs()` is the single source of `policy_kwargs`; `load_mario_model()` resolves the path and loads with `custom_objects={'policy_kwargs': mario_policy_kwargs(), ...}` so checkpoints load regardless of MarioCNN's pickled import path.

### Coupling that must stay in sync

- Training and every loader must use `mario_policy_kwargs()`; don't build `policy_kwargs` inline.
- Observation format is part of the checkpoint: changing frame stack, resize, grayscale, dtype or the action set breaks loading existing checkpoints. Checkpoints from before the uint8 switch (float32 `(84, 84, 4)` observations) no longer load. Checkpoints trained before HUD blanking (up to `runs/world1_all_cont`) still load but were trained with the status bar visible: their numbers come from the old pipeline. Why blank it: those policies keyed on the score/coin digits (freezing that display cut greedy 1-1 clears from 17 to 6 of 31 NOOP offsets), and in the whole game (`SuperMarioBros-v0`) score and coins carry over from earlier levels, so 1-2 to 1-4 show digits never seen on single-stage training levels. A policy clearing single-stage 1-3 10/10 died in 1-3 in all 10 whole-game runs that reached it.
- Fixed PPO hyperparameters live in `mario/domain/config.py` (`LEARNING_RATE`, `CLIP_RANGE`, `N_STEPS`, `GAMMA`, ...), as do the other shared defaults (`NOOP_MAX`, `STALL_STEPS`, `FRAME_STACK`, `OBS_SIZE`, `SEED`); `--batch-size`, `--n-epochs` are CLI flags. `--resume` re-applies learning rate, clip range, batch size and epochs through `custom_objects`.

### Training outputs (`runs/<exp_name>/`, default name `ppo_<env-suffix>_<timestamp>`)

- `checkpoints/checkpoint_<num_timesteps>.zip`: `CheckpointCallback` (`mario/infrastructure/callbacks.py`), every `save_freq // num_envs` calls; names use the model's timestep counter, so they keep increasing across resumes.
- `checkpoints/best_model.zip` and `evaluations.npz` in the run root: SB3 `EvalCallback`, `--eval-episodes` (default 3) deterministic episodes on a separate env seeded `seed + 1000`, every `eval_freq // num_envs` calls.
- `checkpoints/final_model.zip`; `checkpoints/interrupted_<ts>.zip` on Ctrl+C or SIGTERM (SIGTERM is converted to `KeyboardInterrupt`); env closes tolerate already-dead workers.
- `logs/`: TensorBoard (`PPO_N/`, including `rollout/ep_rew_mean` from `VecMonitor`) plus `training_<ts>.log` from `mario.infrastructure.run_logger.setup_logger`.

### Performance (Apple M4, 8 envs, measured)

- The PPO update, not the emulator, dominates wall time: ~40k CNN forward/backward passes per 4,096-sample rollout at 10 epochs.
- Default (MPS, batch 256, 10 epochs): ~450 steps/s end to end; CPU: ~155 (CNN-bound). Emulator floor is ~1,900 raw frames/s per process.
- Learning quality (single seed, 1-1): defaults clear the level deterministically from 800k steps (~40 min). `--n-epochs 4` is ~1.4x faster per step but never reached the flag in 1M steps; don't trade epochs for speed.
- Torch defaults to 4 threads on the M4; more threads (efficiency cores) are slower.

### Headless rendering and devices

- Every command calls `mario.cli.common.configure_sdl` (`SDL_VIDEODRIVER=dummy` / `SDL_AUDIODRIVER=dummy`; `evaluate --render` / `--record-video` keep the video driver) before importing torch, Stable Baselines3 or the emulator, so those imports live inside the command handlers; keep it that way.
- Video frames are read from `env.envs[0].unwrapped.screen` (raw 240×256 RGB, a zero-copy view of the emulator buffer, so it must be `.copy()`'d) and skipped on the `done` step, when the VecEnv has already auto-reset; written with imageio at 30 fps.
- Gymnasium fixes the render mode at construction: `mario evaluate --render` builds its env with `render_mode='human'` via the `render_mode` parameter of `create_vec_env` / `create_mario_env`.
- `--device auto` picks cuda > mps > cpu; MPS is the fastest choice on Apple Silicon with the channels-first pipeline.

## Testing

`make test-fast` (no emulator), `make test-emulator` (a minute) and `make test-slow` (minutes: every NOOP offset, worker processes, the reference checkpoint, a training run with resume). CI runs lint, pyright, the first two tiers, a training smoke run and the container checks.

- `tests/golden/*.json` were captured from the code *before* the package restructure (baseline tag `baseline/pre-refactor`, commit 70b9012) in the pinned stack, each produced twice and kept only when both runs agreed. They pin the API and CLI defaults, the route files byte for byte, every NOOP offset on 1-1 to 1-4, the chained whole game, vec-env rollouts, frontier/demo restarts, cloning datasets, the checkpoint layout and greedy play, a 2048-step training run (CPU weights are bit-identical) with resume, and an evaluate report. The captured NOOP outcomes reproduce the sets recorded above.
- Pixel and tensor hashes (`platform_bound`) are compared only on the platform that captured them (`darwin-arm64`); `exact` values must match everywhere (a Linux container passes them). Tests that need `runs/world1_game3/...` skip without it.
- Tests reach the code only through `tests/adapter.py`. Never regenerate a golden inside a refactor commit (`python -m tests.capture_golden <name>` exists for a deliberate behaviour change, in its own commit).
- `VecMonitor` episode returns differ in the last float32 bit between `DummyVecEnv` and `SubprocVecEnv` (float32 vs float64 rewards in SB3), so that comparison uses a tolerance.
- A plain `stable_baselines3.PPO.load` fails on every checkpoint saved before the restructure; the loader overrides the pickled `policy_kwargs`. Ruff bans `PPO.load` outside `model.py`.
