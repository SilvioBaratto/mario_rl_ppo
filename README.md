# Super Mario Bros Reinforcement Learning with PPO

A production-ready implementation of Proximal Policy Optimization (PPO) for training an AI agent to play Super Mario Bros. This project features a custom CNN architecture, comprehensive evaluation tools, and a modular codebase designed for experimentation and research.

## Features

### Core Implementation
- **PPO Algorithm**: Stable and sample-efficient policy gradient method from Stable Baselines3
- **Custom CNN Architecture**: Purpose-built feature extractor optimized for frame-based observations
- **Vectorized Training**: Parallel environment execution for improved sample efficiency
- **Advanced Preprocessing**: Grayscale conversion, frame stacking, and proper gym/gymnasium compatibility

### Production Features
- **Modular Architecture**: Clean separation of agents, environments, and utilities
- **Comprehensive Evaluation**: Detailed metrics, video recording, and performance analysis
- **Flexible Configuration**: YAML-based configs and extensive command-line options
- **Experiment Management**: Automatic directory structure, checkpointing, and logging
- **Graceful Shutdown**: Signal handling to save models on interruption
- **Multiple Backends**: TensorBoard and Weights & Biases integration

## Installation

### Prerequisites
- Python 3.8+
- pip
- (Optional) Docker and Docker Compose
- (Optional) CUDA-compatible GPU for faster training

### Setup (Local)

```bash
# Clone the repository
git clone https://github.com/yourusername/mario-rl-ppo.git
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
git clone https://github.com/yourusername/mario-rl-ppo.git
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

#### Training with Video Recording
```bash
# Record evaluation videos every 5 evaluations
python train.py \
  --total-timesteps 10000000 \
  --num-envs 8 \
  --record-video \
  --video-freq 5 \
  --exp-name mario_with_videos
```

#### Training with Config File
```bash
# Use a YAML configuration file
python train.py --config runs/ppo_with_videos/config.yaml
```

#### Resume Training
```bash
# Resume from a checkpoint
python train.py \
  --resume runs/my_experiment/checkpoints/checkpoint_1000000 \
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

## Project Structure

```
mario-rl-ppo/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── ppo_agent.py          # PPO implementation with custom CNN
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── env_factory.py        # Environment creation utilities
│   │   └── wrappers.py           # Custom environment wrappers
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

The `MarioCNN` class (src/agents/ppo_agent.py:14-66) implements a custom convolutional neural network optimized for Super Mario Bros:

**Architecture** (Kaggle notebook style):
- **Conv Layer 1**: 32 filters, 3x3 kernel, stride 2, padding 1
- **Conv Layer 2**: 32 filters, 3x3 kernel, stride 2, padding 1
- **Conv Layer 3**: 32 filters, 3x3 kernel, stride 2, padding 1
- **Conv Layer 4**: 32 filters, 3x3 kernel, stride 2, padding 1
- **FC Layer**: 512 units + Dropout(0.2)

**Key Features**:
- Automatic channels-first/last format detection
- Handles both (H, W, C) and (C, H, W) input formats
- Designed for 4-frame stacked grayscale observations

### Environment Preprocessing Pipeline

1. **Base Environment**: `gym-super-mario-bros` (Gym v0.21 API)
2. **Action Mapping**: `JoypadSpace` with SIMPLE_MOVEMENT (7 discrete actions)
3. **Reward Scaling**: `NormalizeRewardEnv` (optional scaling)
4. **Frame Skipping**: `SkipFrame` (skip=4 for temporal abstraction)
5. **Time Limit**: `TimeLimitWrapper` (max 2000 steps to prevent getting stuck)
6. **Grayscale Conversion**: `GrayScaleObservation` (keep_dim=True)
7. **Resize**: `ResizeEnv` (84x84)
8. **API Compatibility**: `GymV21CompatibilityV0` (Shimmy wrapper)
9. **Vectorization**: `DummyVecEnv` for parallel environments
10. **Frame Stacking**: `VecFrameStack` with 4 frames, channels-last format

### PPO Hyperparameters

Optimized hyperparameters for fast training (train.py:246-264):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | linear_schedule(2.5e-4) | Linear decay from 2.5e-4 to 0 |
| `n_steps` | 128 | Steps per environment per update |
| `batch_size` | 256 | Samples per gradient update |
| `n_epochs` | 4 | PPO update epochs per batch |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE parameter |
| `clip_range` | linear_schedule(0.1) | Linear decay from 0.1 to 0 |
| `clip_range_vf` | None | VF clipping disabled |
| `ent_coef` | 0.01 | Entropy coefficient |
| `vf_coef` | 0.5 | Value function coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |

## Command Line Arguments

### Training (train.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--env-id` | SuperMarioBros-v0 | Environment ID |
| `--num-envs` | 8 | Number of parallel environments |
| `--frame-stack` | 4 | Number of frames to stack |
| `--total-timesteps` | 10000000 | Total training timesteps |
| `--eval-freq` | 50000 | Evaluation frequency |
| `--save-freq` | 100000 | Model save frequency |
| `--exp-name` | auto | Experiment name |
| `--seed` | 42 | Random seed |
| `--device` | auto | Device (cpu/cuda/mps/auto) |
| `--resume` | None | Path to checkpoint to resume from |

### Evaluation (evaluate.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to trained model |
| `--env-id` | SuperMarioBros-v0 | Environment ID |
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

#### Training Slow or Getting Stuck on Apple Silicon (MPS)
- **Solution**: Use CPU instead of MPS for more stable performance
  ```bash
  python train.py --device cpu --total-timesteps 10000000
  ```
- **Reason**: MPS support in Stable Baselines3 is incomplete, and PPO is optimized for CPU

#### Low Performance After Training
- **Solution**: Increase training timesteps to 20M+
- **Learning Rate**: The linear schedule already decays from 2.5e-4 to 0

#### Training Instability
- **Solution**: The current settings use conservative clip_range (0.1 with linear decay)
- **Batch Size**: Current 256 provides stable gradient estimates

#### Agent Not Progressing
- **Solution**: Check reward signal in TensorBoard
- **Evaluation**: Run periodic evaluations to identify issues early
- **Video Recording**: Enable to visually inspect behavior

#### Environment Getting Stuck
- **Solution**: The `TimeLimitWrapper` automatically resets episodes after 2000 steps

## Advanced Usage

### Custom Training with Python API

```python
from src.environments.env_factory import create_vec_env
from src.agents.ppo_agent import MarioCNN
from stable_baselines3 import PPO

# Create environment
env = create_vec_env(
    num_envs=8,
    env_id='SuperMarioBros-v0',
    frame_stack=4,
    seed=42
)

# Define linear schedule for learning rate
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

# Policy kwargs with custom CNN
policy_kwargs = dict(
    features_extractor_class=MarioCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

# Create PPO agent
agent = PPO(
    'CnnPolicy',
    env,
    learning_rate=linear_schedule(2.5e-4),
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=linear_schedule(0.1),
    ent_coef=0.01,
    policy_kwargs=policy_kwargs,
    tensorboard_log='./logs',
    device='cpu',  # Recommended for stability
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
from src.agents.ppo_agent import MarioCNN
from stable_baselines3 import PPO

# Create environment
env = create_vec_env(num_envs=1, env_id='SuperMarioBros-v0')

# Policy kwargs for loading
policy_kwargs = dict(
    features_extractor_class=MarioCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

# Load model
agent = PPO.load(
    'runs/my_exp/checkpoints/best_model',
    env=env,
    custom_objects={'policy_kwargs': policy_kwargs}
)

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

The environment uses the default gym-super-mario-bros reward:

```
r = v + c + d

where:
  v = x_new - x_old  (velocity/progress to the right)
  c = clock_old - clock_new  (time penalty)
  d = -15 if died, else 0  (death penalty)

Reward is clipped to [-15, 15]
```

### Frame Stacking

- Observations are stacked as (H, W, C) where C=4 (channels-last)
- The custom CNN automatically detects and handles the format
- Provides temporal information for the agent to perceive motion

### Device Support

Automatic device selection in order of preference:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU (fallback)

**Recommendation**: For stable training, use `--device cpu`. MPS support in Stable Baselines3 is incomplete and may cause slowdowns or freezes.

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