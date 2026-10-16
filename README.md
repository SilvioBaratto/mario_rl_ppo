# Super Mario Bros Reinforcement Learning with PPO

[![CI](https://github.com/SilvioBaratto/mario_rl_ppo/actions/workflows/ci.yml/badge.svg)](https://github.com/SilvioBaratto/mario_rl_ppo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

A PPO agent ([Stable Baselines3](https://stable-baselines3.readthedocs.io/)) that learns Super Mario Bros from pixels with `gym-super-mario-bros`. It trains on one level or several, can practise hard obstacles from emulator snapshots, learns from recorded demonstration routes with a reverse curriculum and behaviour cloning, and clears World 1 in the whole game without losing a life.

> A research project, not a library. The pinned dependencies, the emulator's determinism and the saved checkpoints are part of what it reproduces: see [Development](#development) for the tests that guard them.

## Features

- **PPO from pixels**: Stable Baselines3 with a small custom CNN on four stacked 84×84 grayscale frames
- **Vectorized training** in worker processes, on one level or one policy across several levels
- **Frontier starts** and a **reverse curriculum** along recorded demonstration routes, plus **behaviour cloning** for decisions that exploration never finds
- **Route search**: `mario find-route` finds a demonstration with a Go-Explore style emulator search
- **Evaluation and recording**: per-level success rates, whole-game runs counted by lives lost, MP4 videos
- **Layered package** (`domain`, `application`, `infrastructure`, `cli`) with a dependency rule enforced by a test, and golden traces that prove a refactor leaves training, evaluation and checkpoints untouched
- **Experiment management**: a directory per run with checkpoints, TensorBoard logs, graceful save on Ctrl+C or SIGTERM

## Installation

### Prerequisites
- Python 3.13+ (required by gym-super-mario-bros 9.x / nes-py 9.x)
- (Optional) Docker and Docker Compose
- (Optional) A CUDA GPU, or Apple Silicon (MPS), for faster training

### Setup (local)

```bash
git clone https://github.com/SilvioBaratto/mario_rl_ppo
cd mario_rl_ppo

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# The pinned runtime, then the package (editable) with the dev tools
make install
```

`make install` runs `pip install -r requirements.txt` and `pip install -e ".[dev]"`. On Linux the default torch wheel pulls several GB of CUDA libraries; for a CPU-only machine install torch first from the CPU index:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
make install
```

### Setup (Docker)

```bash
docker-compose build
```

### Verify the installation

```bash
mario --version
mario train --total-timesteps 2048 --num-envs 2 --eval-freq 0 --exp-name smoke   # `make smoke`
```

The `mario` command is also available as `python -m mario`.

## Quick Start

### Training

#### Basic Training
```bash
# Train with default settings (10M timesteps, 8 parallel environments)
mario train --total-timesteps 10000000 --num-envs 8 --exp-name my_experiment
```

#### Multi-Level Training
```bash
# One policy trained on 1-1, 1-2 and 1-3 together; workers are split round-robin
# across the levels, so use a multiple of the level count for --num-envs
mario train --levels 1-1,1-2,1-3 --num-envs 9 --total-timesteps 3000000 --exp-name world1_multi

# Per-level report, including an unseen level to measure generalization
mario evaluate --model-path runs/world1_multi/checkpoints/best_model --levels 1-1,1-2,1-3,1-4
```
Each evaluation during training runs `--eval-episodes` episodes on every level, and
`best_model` tracks the mean reward across them. Repeating a level weights it: with
`--levels 1-1,1-2,1-3,1-3,1-3 --num-envs 9`, 1-3 gets 5 of 9 workers and 3x the
evaluation episodes, which suits fine-tuning a multi-level model on its weakest level
(`--resume <checkpoint>` keeps the step count, so set `--total-timesteps` to the new total). For reference, a single-level 1-1
model (1M steps) clears 1-1 every time but reaches only x≈190 on 1-2, x≈420 on 1-3
and x≈230 on 1-4.

#### Frontier Starts (practising hard obstacles)
```bash
# Half of the training resets restart from a saved emulator state 50-400 px behind the
# point where recent episodes on that level usually end
mario train --levels 1-1,1-2,1-3 --num-envs 9 --frontier-starts 0.5 --exp-name world1_frontier
```
Episodes otherwise always restart at the level start, so an obstacle deep in a level (the
tall tree-top jump at x≈783 in 1-3) is attempted once per long run-up the agent has already
mastered. `FrontierStart` snapshots the emulator whenever Mario stands on solid ground and
tracks each level's frontier (median furthest x of the last 20 episodes), so the restart
point moves forward as the agent improves. Evaluation, `mario evaluate` and
`mario record` always start at the real level start. With frontier starts enabled,
`rollout/ep_rew_mean` mixes full and partial episodes, so judge progress by evaluation.

#### Reverse Curriculum from a Demonstration (obstacles exploration can't find)
Some obstacles need an action sequence the policy essentially never samples. In 1-3 the
gap at x≈783 is only crossable with a *short* hop (A held 4-12 frames) onto an
intermediate tree-top followed by a full jump; the agent always does a long jump onto a
dead-end platform. `mario/infrastructure/demo_routes/1-3.json` records one successful route to
the flag (frame-level joypad bytes from a fresh reset, with waypoint x positions that a replay
must reproduce). A route must not end in a doomed state: an earlier version stopped 15 px
before a tree's edge at full speed, every action from there died at x≈957, and the agent
learned to follow it into the pit. Its open-loop actions reach the flag after every even NOOP
offset but die at x=1724 after odd ones: agent steps are 4 frames, so an episode's frame
parity is fixed, and the timing there differs by one frame.

With `--demo-starts`, the frontier restarts on a level with a route run a reverse
curriculum (Salimans & Chen, 2018, *Learning Montezuma's Revenge from a Single
Demonstration*): each worker replays the route once and keeps an emulator state every 8
frames (two agent steps), mid-jump included. Restarts begin at the current start frame
(75%) or a later one (25%, rehearsal), and the start moves one state earlier each time 5 of
the last 10 episodes from it reach the route's end. A restart lands 12 frames before its
start frame and replays the route's inputs for the first three agent steps: the frame
stack is zero-filled after every reset, and from a zero-padded stack the policy acts as at
a level start and lets go of A, which in mid-jump cuts the jump short for good. Starting next to the goal and moving
back, the agent only has to discover a couple of new decisions at a time, until it starts at
the level start. `mario train` logs every move and writes `curriculum/<level>_start_frame` to
TensorBoard; the position is not saved, so `--resume` starts the curriculum over (a trained
policy walks back through the mastered part quickly). Placing the agent at the
demonstration's waypoints alone did not work: it learned to play on from there, never the
actions that get there.

The curriculum alone still stops where a decision has essentially zero probability under
the policy: in 1-3 it stalled at frame 184, where Mario must *not* press A while landing on
the tree-top (SMB1 only jumps on a fresh A press) and then hold A for 7 steps. The policy
presses A in mid-air with probability ~0.9995, so no rollout ever showed PPO the
alternative. `--demo-bc N` adds behaviour cloning: routes carry `agent_actions` (the same
route as one SIMPLE_MOVEMENT action per agent step), `build_demo_dataset` plays them
through the training observation pipeline once per NOOP offset 0..`--noop-max`, and
`DemoCloningCallback` takes N cross-entropy steps on those pairs before every rollout
(`demo/bc_loss`, `demo/bc_accuracy` in TensorBoard). With 10 steps the curriculum crossed
frame 184 and the hop within 10k steps after being stuck there for 1.9M.
```bash
mario train --resume runs/world1_reverse/checkpoints/best_model \
  --levels 1-1,1-2,1-3,1-3,1-3 --num-envs 9 --total-timesteps 6500000 \
  --frontier-starts 0.5 --demo-starts --demo-bc 10 --exp-name world1_bc
```

#### Finding a Route (`mario find-route`)
A level without a route can get one by searching the emulator directly, no trained policy
needed. Each search is Go-Explore style: an archive keeps, per 16×16 px cell of Mario's
position, the emulator state and the shortest known action sequence reaching it; restored
cells are expanded with random sticky actions biased to the right, until a step reports
`flag_get` (in castles, touching the axe). Open-loop routes depend on the NOOP offset (on
1-3 and 1-4 the same actions fail after some offsets), so the tool replays every
flag route after NOOP offsets 0..`--max-noop` and keeps the one that works for the most;
behaviour cloning then uses exactly those offsets. The output is deterministic for a given
`--seed`/`--searches`/`--budget`.
```bash
mario find-route --level 1-4 --searches 21   # writes mario/infrastructure/demo_routes/1-4.json
```
For 1-4 every search reached the axe within seconds; the kept route (288 steps, over Bowser
from the lift) works after 28 of 31 offsets, the next best after 15. Snapshots of 1-4 are
not perfectly restorable in nes-py (a restored state with identical RAM can diverge hundreds
of frames later); the curriculum only needs playable states, so this does not matter for
training.

#### Expected Learning Curve (World 1-1)
Measured with the defaults on an Apple M4 (8 envs, MPS, single seed): the
deterministic policy first clears 1-1 at 800k steps (10/10 evaluation episodes,
~40 min of training). `--n-epochs 4` runs ~1.4x more steps/s but did not reach
the flag once within 1M steps, so keep the default 10 epochs.

#### Resume Training
```bash
# Continue a run until it reaches 20M total timesteps (the step counter carries over)
mario train \
  --resume runs/my_experiment/checkpoints/checkpoint_10000000 \
  --total-timesteps 20000000 \
  --exp-name my_experiment_continued
```

### Evaluation

#### Basic Evaluation
```bash
# Evaluate a trained model over 10 episodes
mario evaluate \
  --model-path runs/my_experiment/checkpoints/best_model \
  --n-episodes 10
```

#### Evaluation with Video Recording
```bash
# Record videos of evaluation episodes
mario evaluate \
  --model-path runs/my_experiment/checkpoints/best_model \
  --n-episodes 10 \
  --record-video \
  --video-dir evaluation_videos
```

#### Live Rendering
```bash
# Watch the agent play in real-time
mario evaluate \
  --model-path runs/my_experiment/checkpoints/best_model \
  --render \
  --n-episodes 5
```

### Recording Gameplay Videos

`mario record` provides a simple way to record a single gameplay episode with a trained model.

#### Basic Recording
```bash
# Record one game with auto-generated filename
mario record \
  --model-path runs/my_experiment/checkpoints/best_model
```

#### Custom Output Path
```bash
# Specify output video path
mario record \
  --model-path runs/my_experiment/checkpoints/best_model \
  --output my_mario_gameplay.mp4
```

#### Extended Recording
```bash
# Record with higher step limit for longer episodes
mario record \
  --model-path runs/my_experiment/checkpoints/best_model \
  --max-steps 10000
```

#### Full Game (World 1 without losing a life)
`SuperMarioBros-v0` is the whole game in one episode (3 lives, level transitions included;
gym-super-mario-bros skips the intro of 1-2, so each level starts as in single-stage
training). `--stop-at-level 2-1` ends the video once 1-4 is cleared, and the summary line
says how many levels were cleared before the first life was lost:
```bash
mario record --model-path runs/my_experiment/checkpoints/best_model \
  --env-id SuperMarioBros-v0 --stop-at-level 2-1 --output world1.mp4
# Over many episodes: "Levels cleared before the first life lost" and ">= N levels" counts
mario evaluate --model-path runs/my_experiment/checkpoints/best_model \
  --env-id SuperMarioBros-v0 --n-episodes 20
```
A clean World 1 needs every level to be reliable on its own: at 90% per level, only
0.9^4 ≈ 66% of runs get through all four without dying.

#### Recording Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to trained model (.zip optional) |
| `--output` | auto | Output video path (auto-generates timestamp-based name) |
| `--env-id` | SuperMarioBros-1-1-v0 | Environment ID |
| `--frame-stack` | 4 | Number of frames to stack |
| `--max-steps` | 5000 | Maximum agent steps (also the episode time limit) |
| `--noop-max` | 30 | Max random NOOP frames before the episode starts |
| `--seed` | 42 | Random seed |
| `--stochastic` | False | Sample actions instead of the deterministic policy |
| `--device` | auto | Device to use (cpu/cuda/mps/auto) |
| `--stop-at-level` | None | With `SuperMarioBros-v0`, end the recording when this level starts (e.g. `2-1`) |

The script outputs:
- Real-time progress updates every 100 steps
- Final game statistics (steps, reward, max distance, completion status)
- MP4 video at 30 fps with filename indicating completion status

## Project Structure

```
mario_rl_ppo/
├── mario/
│   ├── domain/            # Pure rules: config constants, levels, route arithmetic,
│   │                      #   the reverse curriculum, episode summaries
│   ├── application/       # Use cases: evaluate_agent, record_game
│   ├── infrastructure/    # Adapters: Gymnasium wrappers, vec envs, FrontierStart,
│   │   │                  #   MarioCNN and model loading, behaviour cloning,
│   │   │                  #   SB3 callbacks, route search, logging
│   │   └── demo_routes/   # Recorded routes for 1-1 to 1-4 (package data)
│   └── cli/               # `mario train | evaluate | record | find-route`
├── tests/                 # Unit tests, golden traces (tests/golden) and the layering test
├── runs/                  # Training outputs (git-ignored)
├── pyproject.toml         # Package metadata, dependencies, ruff/pytest/pyright config
├── requirements.txt       # Exact pins used by Docker and CI
├── Makefile               # lint, tests, smoke run, container checks
├── Dockerfile
├── docker-compose.yml
├── CHANGELOG.md
├── CONTRIBUTING.md
└── CLAUDE.md              # Design notes for the emulator pipeline, curriculum and checkpoints
```

**The dependency rule.** `domain` imports nothing from the package and only numpy from outside it. `application` and `infrastructure` import only `domain`. `cli` is the composition root and may import all of them. `tests/test_layering.py` reads every module's imports with `ast` and fails on a violation.

## Architecture

### Custom CNN Feature Extractor

The `MarioCNN` class (`mario/infrastructure/model.py`) implements a custom convolutional neural network optimized for Super Mario Bros:

**Architecture**:
- **Conv Layer 1**: 32 filters, 3x3 kernel, stride 2, padding 1
- **Conv Layer 2**: 32 filters, 3x3 kernel, stride 2, padding 1
- **Conv Layer 3**: 32 filters, 3x3 kernel, stride 2, padding 1
- **Conv Layer 4**: 32 filters, 3x3 kernel, stride 2, padding 1
- **FC Layer**: 512 units

**Key Features**:
- Orthogonal weight initialization for stable training
- Automatic channels-first/last format detection
- Handles both (H, W, C) and (C, H, W) input formats
- Designed for 4-frame stacked grayscale observations (uint8 frames, scaled to [0,1] by SB3 on the training device)

### Environment Preprocessing Pipeline

1. **Base Environment**: `gym-super-mario-bros` 9.x (Gymnasium API); default `SuperMarioBros-1-1-v0`, where an episode ends on death or flag
2. **Action Mapping**: `JoypadSpace` with SIMPLE_MOVEMENT (7 discrete actions)
3. **Random Starts**: `NoopReset` (0-30 NOOP frames after each reset, so episodes from the deterministic emulator differ)
4. **Custom Reward Shaping**: `CustomReward` (score bonus, +50 flag, -50 per life lost, /10 scaling)
5. **Frame Skipping**: `SkipFrame` (skip=4 for temporal abstraction)
6. **Stall Cutoff**: `StallTruncation` (truncate after 150 agent steps without a new furthest x; stuck greedy episodes otherwise run 2000 steps)
7. **Time Limit**: Gymnasium `TimeLimit` (max 2000 steps)
8. **Grayscale + Resize**: `GrayResizeObservation` (cv2, 84x84 uint8)
9. **Vectorization**: `SubprocVecEnv` for true multiprocessing (faster than DummyVecEnv)
10. **Episode Statistics**: `VecMonitor` (feeds `rollout/ep_rew_mean` / `ep_len_mean` in TensorBoard)
11. **Frame Stacking**: `VecFrameStack` with 4 frames; SB3 adds `VecTransposeImage` so the CNN gets channels-first input

### PPO Hyperparameters

Hyperparameters for Mario (`mario/domain/config.py`, plus the two flags):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 1e-4 | Fixed learning rate |
| `n_steps` | 512 | Steps per environment per update |
| `batch_size` | 256 | Samples per gradient update (`--batch-size`) |
| `n_epochs` | 10 | PPO update epochs per rollout (`--n-epochs`) |
| `gamma` | 0.9 | Discount factor (lower = values immediate rewards more) |
| `gae_lambda` | 1.0 | GAE parameter |
| `clip_range` | 0.2 | Fixed PPO clip range |
| `clip_range_vf` | None | VF clipping disabled |
| `ent_coef` | 0.01 | Entropy coefficient |
| `vf_coef` | 0.5 | Value function coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |

## Command Line Arguments

### Training (`mario train`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--env-id` | SuperMarioBros-1-1-v0 | Environment ID |
| `--levels` | None | Train on several levels together, e.g. `1-1,1-2,1-3` (overrides `--env-id`); repeating a level gives it more workers |
| `--num-envs` | 8 | Number of parallel environments |
| `--frame-stack` | 4 | Number of frames to stack |
| `--total-timesteps` | 10000000 | Total timesteps to reach (also the target when resuming) |
| `--batch-size` | 256 | PPO minibatch size in samples |
| `--n-epochs` | 10 | PPO epochs per rollout |
| `--eval-freq` | 50000 | Evaluation frequency |
| `--eval-episodes` | 3 | Episodes per evaluation |
| `--save-freq` | 100000 | Model save frequency |
| `--noop-max` | 30 | Max random NOOP frames after each reset (0 disables) |
| `--stall-steps` | 150 | Truncate an episode after this many steps without a new furthest x (0 disables) |
| `--frontier-starts` | 0.0 | Fraction of training resets that restart just behind where episodes usually end (0 disables) |
| `--demo-starts` | False | With `--frontier-starts`, run a reverse curriculum along recorded demonstration routes |
| `--demo-bc` | 0 | Behaviour-cloning steps on routes' `agent_actions` before every rollout (0 disables) |
| `--exp-name` | auto | Experiment name |
| `--seed` | 42 | Random seed |
| `--device` | auto | Device (cpu/cuda/mps/auto) |
| `--resume` | None | Path to checkpoint to resume from (.zip optional) |

### Evaluation (`mario evaluate`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to trained model (.zip optional) |
| `--env-id` | SuperMarioBros-1-1-v0 | Environment ID |
| `--noop-max` | 30 | Max random NOOP frames after each reset (0 makes deterministic episodes identical) |
| `--levels` | None | Evaluate on several levels, e.g. `1-1,1-2,1-3,1-4`; prints a per-level summary |
| `--stall-steps` | 150 | End an episode after this many steps without a new furthest x (0 disables) |
| `--frame-stack` | 4 | Number of frames to stack |
| `--n-episodes` | 10 | Number of evaluation episodes |
| `--render` | False | Render environment live |
| `--record-video` | False | Record evaluation videos |
| `--video-dir` | eval_videos | Video output directory |
| `--deterministic` | True | Use deterministic policy |
| `--stochastic` | False | Use stochastic policy |
| `--seed` | 42 | Random seed |
| `--device` | auto | Device to use |

## Results

The reference model is `runs/world1_game3/checkpoints/checkpoint_17894136` (17.9M steps), trained with `--levels 1-1,1-2,1-2,1-3,1-4,game,game,game,game,game --num-envs 10 --frontier-starts 0.5 --demo-starts --demo-bc 10`. It clears World 1 in the whole game without losing a life; `CLAUDE.md` has the full numbers (31/31 greedy NOOP offsets, 15/30 stochastic).

Single-stage greedy play, 10 episodes per level, produced by

```bash
mario evaluate --model-path runs/world1_game3/checkpoints/checkpoint_17894136 \
  --levels 1-1,1-2,1-3,1-4 --n-episodes 10 --device cpu --seed 42
```

| Level | Success | Average reward | Best x |
|-------|---------|----------------|--------|
| 1-1 | 100% | 314.85 | 3161 |
| 1-2 | 90% | 282.82 | 3161 |
| 1-3 | 100% | 255.61 | 2425 |
| 1-4 | 90% | 216.74 | 2244 |

Every number here comes from a run of that command, not from memory; re-run it and update the table from its output. A checkpoint is not tracked in git, so the command needs your own `runs/` directory.

### Evaluation metrics

`mario evaluate` reports, per level and in a JSON file (`eval_results_ppo_<timestamp>.json`):
- **Episode rewards and lengths**, and their mean and standard deviation
- **Success rate**: the percentage of episodes where Mario reaches the flag
- **Max distance**: the furthest x position reached
- **Action distribution** per episode, and for whole-game ids the levels cleared before the first life lost

## Training Tips

### Recommended Settings

1. **Use 8+ parallel environments**: Improves sample efficiency
   ```bash
   mario train --num-envs 8
   ```

2. **Train for 10M+ timesteps**: Mario requires substantial training
   ```bash
   mario train --total-timesteps 10000000
   ```

3. **Monitor with TensorBoard**: Track training progress
   ```bash
   tensorboard --logdir runs/my_experiment/logs
   ```

4. **Save frequently during long training**: Protect against interruptions
   ```bash
   mario train --save-freq 50000
   ```

### Troubleshooting

#### Choosing a Device on Apple Silicon
- `--device auto` picks MPS, the fastest option here: on an Apple M4 with 8 envs, ~450 steps/s on MPS vs ~155 steps/s on CPU, because the PPO update is bound by the CNN
- If MPS misbehaves, fall back to CPU (about 3x slower):
  ```bash
  mario train --device cpu
  ```

#### Low Performance After Training
- **Solution**: Increase training timesteps to 10M+
- **Learning Rate**: Fixed at 1e-4 for stable training

#### Training Instability
- **Solution**: The current settings use clip_range=0.2 and gamma=0.9
- **Batch Size**: 256-sample minibatches (16 per 4096-sample rollout) with 10 epochs per update

#### Agent Not Progressing
- **Solution**: Check `rollout/ep_rew_mean` and `eval/mean_reward` in TensorBoard
- **Evaluation**: Run periodic evaluations to identify issues early
- **Video Recording**: Enable to visually inspect behavior

#### Environment Getting Stuck
- **Solution**: The `TimeLimit` wrapper automatically resets episodes after 2000 steps

## Advanced Usage

### Custom Training with Python API

```python
from mario.infrastructure.factory import create_vec_env
from mario.infrastructure.model import mario_policy_kwargs
from stable_baselines3 import PPO

# Create environment (SubprocVecEnv re-imports your script, so keep this under an `if __name__ == "__main__":` guard)
env = create_vec_env(num_envs=8, env_id="SuperMarioBros-1-1-v0", frame_stack=4, seed=42)

# Create PPO agent
agent = PPO(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=256,
    n_epochs=10,
    gamma=0.9,
    gae_lambda=1.0,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=mario_policy_kwargs(),
    tensorboard_log="./logs",
    device="auto",
    verbose=1,
)

# Train
agent.learn(total_timesteps=10_000_000)

# Save
agent.save("my_mario_model")
```

### Loading and Using Trained Models

Always load through `load_mario_model`: checkpoints pickle the CNN under an import path that no longer exists, and only its `policy_kwargs` override makes them load (ruff bans a plain `PPO.load`).

```python
from mario.infrastructure.factory import create_vec_env
from mario.infrastructure.model import load_mario_model

# Create environment
env = create_vec_env(num_envs=1, env_id="SuperMarioBros-1-1-v0")

# Load model (.zip optional)
agent = load_mario_model("runs/my_exp/checkpoints/best_model", env=env)

# Run agent
obs = env.reset()
for _ in range(1000):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done[0]:
        obs = env.reset()
```

## Docker

### Using Docker Compose

```bash
# Build the image
docker-compose build

# Train (default: 10M timesteps)
docker-compose up train

# Resume training from checkpoint
docker-compose --profile resume up resume

# Evaluate trained model
docker-compose --profile eval up evaluate

# Record gameplay video
docker-compose --profile record up record

# Monitor with TensorBoard (http://localhost:6006)
docker-compose --profile monitor up tensorboard

# Train + TensorBoard together
docker-compose --profile monitor up train tensorboard

# Stop all services
docker-compose down
```

### Using Docker Directly

```bash
# Build
docker build -t mario-rl-ppo .

# Train
docker run --rm -v $(pwd)/runs:/app/runs mario-rl-ppo \
  mario train --total-timesteps 10000000 --device cpu --exp-name docker_run

# Evaluate (the report is written to the working directory, so run it inside runs/)
docker run --rm -v $(pwd)/runs:/app/runs -w /app/runs mario-rl-ppo \
  mario evaluate --model-path /app/runs/docker_run/checkpoints/best_model --n-episodes 10

# Record video
docker run --rm -v $(pwd)/runs:/app/runs -v $(pwd)/videos:/app/videos mario-rl-ppo \
  mario record --model-path runs/docker_run/checkpoints/best_model --output videos/docker_run.mp4
```

## Experiment Management

### Directory Structure

Each training run automatically creates:

```
runs/<exp_name>/
├── checkpoints/
│   ├── best_model.zip           # Best performing model
│   ├── checkpoint_100000.zip    # Regular checkpoint
│   ├── checkpoint_200000.zip
│   └── final_model.zip          # Final model after training
├── logs/
│   ├── PPO_1/                   # TensorBoard logs
│   │   ├── events.out.tfevents.*
│   │   └── ...
│   └── training_<timestamp>.log
├── evaluations.npz              # Evaluation history (EvalCallback)
└── videos/                      # Created empty; the evaluation CLI writes its own videos
```

### Checkpointing Strategy

- **best_model.zip**: Automatically saved when mean episode reward improves
- **checkpoint_N.zip**: Saved every `save_freq` timesteps
- **final_model.zip**: Saved at the end of training
- **interrupted_*.zip**: Saved on SIGINT/SIGTERM

## Technical Details

### Reward Function

Custom reward shaping for improved learning:

```
r = (base_reward + score_bonus + completion_bonus) / 10

where:
  base_reward = x_new - x_old + time_penalty  (default env reward)
  score_bonus = (current_score - previous_score) / 40  (game score bonus)
  completion_bonus = +50 when the flag is reached, -50 for every life lost

Flag and death events are detected from the env's info (flag_get edge, life
counter), so they fire correctly on single-stage ids and on the multi-level
SuperMarioBros-v0.

Final reward is scaled by 0.1 for stable training
```

**Note**: Due to the /10 scaling, reported rewards are ~10x smaller than with default rewards. A reward of ~300 indicates strong performance.

### Frame Stacking

- Frames are stacked channels-last as (84, 84, 4) uint8; SB3's `VecTransposeImage` hands the CNN channels-first (4, 84, 84)
- The custom CNN automatically detects and handles the format
- Provides temporal information for the agent to perceive motion

### Device Support

Automatic device selection in order of preference:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU (fallback)

**Recommendation**: keep `auto`. With uint8 channels-first observations, MPS trains ~3x faster than CPU on Apple Silicon; use `--device cpu` only if MPS misbehaves.

## Development

```bash
make install        # pinned runtime, the package, and the dev extra (pytest, ruff, pyright)
make lint           # ruff check, ruff format --check, pyright
make test-fast      # tests that need no emulator
make test-emulator  # short emulator tests: raw routes, the wrapped env, the whole game
make test-slow      # every NOOP offset, worker processes, the checkpoint and a training run
make smoke          # a 2,048-step training run on two workers
make docker-check   # build the image; check the routes, permissions and compose file
```

### The golden traces

`tests/golden` holds traces captured from the code before it was restructured: the public API and command-line defaults, the demonstration routes, the wrapped environment for every NOOP offset 0-30 on 1-1 to 1-4, the chained whole-game run, vec-env rollouts, behaviour-cloning datasets, the reference checkpoint, a short training run with resume, and an evaluation report. The tests replay the current code against them, so a refactor that changes training, evaluation or checkpoint loading fails a test.

- Exact values (rewards, infos, emulator state) must match on every platform. Hashes of pixels and tensors depend on the OpenCV and BLAS builds and are compared only on the platform that captured them.
- Tests that need `runs/world1_game3/...` skip when the checkpoint is absent.
- Never regenerate a golden inside a refactor commit. If behaviour has to change on purpose, change the golden in its own commit and say why (`python -m tests.capture_golden <name>`).

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: a test first, `make lint` and `make test-fast` green, one focused commit per change in [Conventional Commits](https://www.conventionalcommits.org/) style.

## References

### Papers

- **PPO**: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv preprint arXiv:1707.06347.

### Libraries and Tools

- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) - Super Mario Bros environment for OpenAI Gym
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - PyTorch implementations of RL algorithms
- [nes-py](https://github.com/Kautenja/nes-py) - NES emulator for Python

## License

[MIT License](LICENSE)

## Acknowledgments

- OpenAI Gym Super Mario Bros environment by Christian Kauten
- Stable Baselines3 team for excellent RL implementations
- PyTorch team for the deep learning framework
