# Super Mario Bros Reinforcement Learning with PPO

A production-ready implementation of Proximal Policy Optimization (PPO) for training an AI agent to play Super Mario Bros. This project features a custom CNN architecture, comprehensive evaluation tools, and a modular codebase designed for experimentation and research.

## Features

### Core Implementation
- **PPO Algorithm**: Stable and sample-efficient policy gradient method from Stable Baselines3
- **Custom CNN Architecture**: Purpose-built feature extractor optimized for frame-based observations
- **Vectorized Training**: Parallel environment execution for improved sample efficiency
- **Advanced Preprocessing**: Grayscale conversion, frame stacking, native Gymnasium API

### Production Features
- **Modular Architecture**: Clean separation of agents, environments, and utilities
- **Comprehensive Evaluation**: Detailed metrics, video recording, and performance analysis
- **Flexible Configuration**: YAML-based configs and extensive command-line options
- **Experiment Management**: Automatic directory structure, checkpointing, and logging
- **Graceful Shutdown**: Signal handling to save models on interruption
- **Multiple Backends**: TensorBoard and Weights & Biases integration

## Installation

### Prerequisites
- Python 3.13+ (required by gym-super-mario-bros 9.x / nes-py 9.x)
- pip
- (Optional) Docker and Docker Compose
- (Optional) CUDA-compatible GPU for faster training

### Setup (Local)

```bash
# Clone the repository
git clone https://github.com/SilvioBaratto/mario-rl-ppo.git
cd mario-rl-ppo

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup (Docker)

```bash
# Clone the repository
git clone https://github.com/SilvioBaratto/mario-rl-ppo.git
cd mario-rl-ppo

# Build the Docker image
docker-compose build
```

### Verify Installation

```bash
# Local: Quick test run (100k timesteps, 2 environments)
python train.py --total-timesteps 100000 --num-envs 2 --exp-name test_run

# Docker: Quick test run
docker-compose run --rm train python train.py --total-timesteps 100000 --num-envs 2 --exp-name test_run
```

## Quick Start

### Training

#### Basic Training
```bash
# Train with default settings (10M timesteps, 8 parallel environments)
python train.py --total-timesteps 10000000 --num-envs 8 --exp-name my_experiment
```

#### Multi-Level Training
```bash
# One policy trained on 1-1, 1-2 and 1-3 together; workers are split round-robin
# across the levels, so use a multiple of the level count for --num-envs
python train.py --levels 1-1,1-2,1-3 --num-envs 9 --total-timesteps 3000000 --exp-name world1_multi

# Per-level report, including an unseen level to measure generalization
python evaluate.py --model-path runs/world1_multi/checkpoints/best_model --levels 1-1,1-2,1-3,1-4
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
python train.py --levels 1-1,1-2,1-3 --num-envs 9 --frontier-starts 0.5 --exp-name world1_frontier
```
Episodes otherwise always restart at the level start, so an obstacle deep in a level (the
tall tree-top jump at x≈783 in 1-3) is attempted once per long run-up the agent has already
mastered. `FrontierStart` snapshots the emulator whenever Mario stands on solid ground and
tracks each level's frontier (median furthest x of the last 20 episodes), so the restart
point moves forward as the agent improves. Evaluation, `evaluate.py` and
`play_and_record.py` always start at the real level start. With frontier starts enabled,
`rollout/ep_rew_mean` mixes full and partial episodes, so judge progress by evaluation.

#### Demonstration Starts (obstacles exploration can't find)
Some obstacles need an action sequence the policy essentially never samples. In 1-3 the
gap at x≈783 is only crossable with a *short* hop (A held 4-12 frames) onto an
intermediate tree-top followed by a full jump; the agent always does a long jump onto a
dead-end platform. `src/environments/demo_routes/1-3.json` records one successful route
(frame-level joypad bytes from a fresh reset, with waypoint x positions that a replay must
reproduce). With `--demo-starts`, each worker replays its level's route once and adds the
waypoints to the frontier restart points while they lie ahead of the agent's frontier;
demo-started episodes are excluded from the frontier statistics.
```bash
python train.py --resume runs/world1_frontier/checkpoints/best_model \
  --levels 1-1,1-2,1-3,1-3,1-3 --num-envs 9 --total-timesteps 3100000 \
  --frontier-starts 0.5 --demo-starts --exp-name world1_demo
```

#### Expected Learning Curve (World 1-1)
Measured with the defaults on an Apple M4 (8 envs, MPS, single seed): the
deterministic policy first clears 1-1 at 800k steps (10/10 evaluation episodes,
~40 min of training). `--n-epochs 4` runs ~1.4x more steps/s but did not reach
the flag once within 1M steps, so keep the default 10 epochs.

#### Resume Training
```bash
# Continue a run until it reaches 20M total timesteps (the step counter carries over)
python train.py \
  --resume runs/my_experiment/checkpoints/checkpoint_10000000 \
  --total-timesteps 20000000 \
  --exp-name my_experiment_continued
```

### Evaluation

#### Basic Evaluation
```bash
# Evaluate a trained model over 10 episodes
python evaluate.py \
  --model-path runs/my_experiment/checkpoints/best_model \
  --n-episodes 10
```

#### Evaluation with Video Recording
```bash
# Record videos of evaluation episodes
python evaluate.py \
  --model-path runs/my_experiment/checkpoints/best_model \
  --n-episodes 10 \
  --record-video \
  --video-dir evaluation_videos
```

#### Live Rendering
```bash
# Watch the agent play in real-time
python evaluate.py \
  --model-path runs/my_experiment/checkpoints/best_model \
  --render \
  --n-episodes 5
```

### Recording Gameplay Videos

The `play_and_record.py` script provides a simple way to record a single gameplay episode with a trained model.

#### Basic Recording
```bash
# Record one game with auto-generated filename
python play_and_record.py \
  --model-path runs/my_experiment/checkpoints/best_model
```

#### Custom Output Path
```bash
# Specify output video path
python play_and_record.py \
  --model-path runs/my_experiment/checkpoints/best_model \
  --output my_mario_gameplay.mp4
```

#### Extended Recording
```bash
# Record with higher step limit for longer episodes
python play_and_record.py \
  --model-path runs/my_experiment/checkpoints/best_model \
  --max-steps 10000
```

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

The script outputs:
- Real-time progress updates every 100 steps
- Final game statistics (steps, reward, max distance, completion status)
- MP4 video at 30 fps with filename indicating completion status

## Project Structure

```
mario-rl-ppo/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── ppo_agent.py          # PPO implementation with custom CNN
│   ├── environments/
│   │   ├── __init__.py
│   │   └── env_factory.py        # Environment wrappers and VecEnv factories
│   └── utils/
│       ├── __init__.py
│       └── logger.py             # Logging utilities
├── runs/                          # Training outputs (auto-generated)
│   └── <exp_name>/
│       ├── checkpoints/          # Model checkpoints
│       ├── logs/                 # TensorBoard logs
│       └── videos/               # Training evaluation videos
├── train.py                      # Main training script
├── evaluate.py                   # Evaluation script
├── play_and_record.py            # Video recording script
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker Compose configuration
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Architecture

### Custom CNN Feature Extractor

The `MarioCNN` class (src/agents/ppo_agent.py) implements a custom convolutional neural network optimized for Super Mario Bros:

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

Optimized hyperparameters for Mario (train.py):

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

### Training (train.py)

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
| `--demo-starts` | False | With `--frontier-starts`, also restart from waypoints of recorded demonstration routes |
| `--exp-name` | auto | Experiment name |
| `--seed` | 42 | Random seed |
| `--device` | auto | Device (cpu/cuda/mps/auto) |
| `--resume` | None | Path to checkpoint to resume from (.zip optional) |

### Evaluation (evaluate.py)

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

### Evaluation Metrics

The evaluation script provides comprehensive metrics:
- **Episode Rewards**: Total reward per episode
- **Episode Lengths**: Number of steps per episode
- **Success Rate**: Percentage of episodes where Mario reaches the flag
- **Max Distance**: Furthest x-position reached
- **Action Distribution**: Analysis of action usage

### Example Output

```
==================================================
EVALUATION RESULTS
==================================================
Episodes evaluated: 10
Success rate: 80.0%
Average reward: 2847.50 +/- 324.18
Average episode length: 512.3 +/- 89.7
Max reward: 3200.00
Min reward: 2100.00
Max distance reached: 3161
==================================================
```

## Training Tips

### Recommended Settings

1. **Use 8+ parallel environments**: Improves sample efficiency
   ```bash
   python train.py --num-envs 8
   ```

2. **Train for 10M+ timesteps**: Mario requires substantial training
   ```bash
   python train.py --total-timesteps 10000000
   ```

3. **Monitor with TensorBoard**: Track training progress
   ```bash
   tensorboard --logdir runs/my_experiment/logs
   ```

4. **Save frequently during long training**: Protect against interruptions
   ```bash
   python train.py --save-freq 50000
   ```

### Troubleshooting

#### Choosing a Device on Apple Silicon
- `--device auto` picks MPS, the fastest option here: on an Apple M4 with 8 envs, ~450 steps/s on MPS vs ~155 steps/s on CPU, because the PPO update is bound by the CNN
- If MPS misbehaves, fall back to CPU (about 3x slower):
  ```bash
  python train.py --device cpu
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
from src.environments.env_factory import create_vec_env
from src.agents.ppo_agent import mario_policy_kwargs
from stable_baselines3 import PPO

# Create environment (run from the repo root; SubprocVecEnv needs a __main__ guard)
env = create_vec_env(
    num_envs=8,
    env_id='SuperMarioBros-1-1-v0',
    frame_stack=4,
    seed=42
)

# Create PPO agent
agent = PPO(
    'CnnPolicy',
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
    tensorboard_log='./logs',
    device='auto',
    verbose=1
)

# Train
agent.learn(total_timesteps=10_000_000)

# Save
agent.save('my_mario_model')
```

### Loading and Using Trained Models

```python
from src.environments.env_factory import create_vec_env
from src.agents.ppo_agent import load_mario_model

# Create environment
env = create_vec_env(num_envs=1, env_id='SuperMarioBros-1-1-v0')

# Load model (.zip optional)
agent = load_mario_model('runs/my_exp/checkpoints/best_model', env=env)

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
  python train.py --total-timesteps 10000000 --device cpu --exp-name docker_run

# Evaluate
docker run --rm -v $(pwd)/runs:/app/runs mario-rl-ppo \
  python evaluate.py --model-path runs/docker_run/checkpoints/best_model --n-episodes 10

# Record video
docker run --rm -v $(pwd)/runs:/app/runs -v $(pwd)/videos:/app/videos mario-rl-ppo \
  python play_and_record.py --model-path runs/docker_run/checkpoints/best_model
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
│   └── PPO_1/                   # TensorBoard logs
│       ├── events.out.tfevents.*
│       └── ...
├── videos/
│   ├── eval-SuperMarioBros-v0-episode-0.mp4
│   └── ...
└── config.yaml                  # Configuration used for this run
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

### Running Tests

```bash
# Install development dependencies
pip install pytest black flake8

# Run tests (if implemented)
pytest

# Format code
black .

# Check code style
flake8 src/ train.py evaluate.py
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

### Papers

- **PPO**: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv preprint arXiv:1707.06347.

### Libraries and Tools

- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) - Super Mario Bros environment for OpenAI Gym
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - PyTorch implementations of RL algorithms
- [nes-py](https://github.com/Kautenja/nes-py) - NES emulator for Python

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenAI Gym Super Mario Bros environment by Christian Kauten
- Stable Baselines3 team for excellent RL implementations
- PyTorch team for the deep learning framework