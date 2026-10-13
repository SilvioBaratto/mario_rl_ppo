"""
Behaviour cloning on demonstration routes, mixed into PPO training
"""

import warnings
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from environments.env_factory import create_mario_env


def build_demo_dataset(routes: Sequence[Dict[str, Any]], frame_stack: int = 4,
                       max_noop: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Play each route's ``agent_actions`` through the training observation pipeline.

    Every route is played once per NOOP offset 0..``max_noop`` (raw frames
    idled after the reset, as ``NoopReset`` does in training), so the clones
    cover the slightly shifted states training episodes actually reach.
    Observations are stacked like ``VecFrameStack`` (zero-filled at the
    reset, newest frame last) and transposed channels-first like the
    ``VecTransposeImage`` SB3 adds, i.e. exactly what the policy sees.

    Args:
        routes: Routes from ``load_demo_route`` that have ``agent_actions``.
        frame_stack: Frames per observation, as in training.
        max_noop: Largest NOOP offset to replay; match ``--noop-max``.

    Returns:
        (observations, actions): uint8 ``(N, frame_stack, 84, 84)`` and int64
        ``(N,)``. Offsets whose replay no longer reaches the route's last
        waypoint are skipped with a warning, since their actions would no
        longer be a demonstration.
    """
    all_obs, all_actions = [], []
    for route in routes:
        env = create_mario_env(route['env_id'], noop_max=0, stall_steps=0, use_custom_reward=False,
                               max_steps=len(route['agent_actions']) + 1)
        goal_x = route['waypoints'][-1]['x']
        try:
            for noop in range(max_noop + 1):
                frame, _ = env.reset()
                nes = env.unwrapped
                for _ in range(noop):
                    nes.step(0)
                if noop:
                    frame = env.observation(nes.screen)
                stack = np.zeros(frame.shape[:2] + (frame_stack,), dtype=np.uint8)
                stack[..., -1:] = frame
                obs, max_x, failed = [], 0, False
                for action in route['agent_actions']:
                    obs.append(stack.transpose(2, 0, 1).copy())
                    frame, _, terminated, truncated, info = env.step(action)
                    stack = np.roll(stack, -1, axis=-1)
                    stack[..., -1:] = frame
                    max_x = max(max_x, info['x_pos'])
                    if terminated or truncated:
                        failed = True
                        break
                if failed or max_x < goal_x:
                    warnings.warn(f"{route['env_id']} agent_actions miss the route end after {noop} NOOP "
                                  f"frames (x={max_x}, goal {goal_x}); skipping that offset")
                    continue
                all_obs.extend(obs)
                all_actions.extend(route['agent_actions'])
        finally:
            env.close()
    if not all_obs:
        return np.zeros((0, frame_stack, 84, 84), dtype=np.uint8), np.zeros(0, dtype=np.int64)
    return np.stack(all_obs), np.asarray(all_actions, dtype=np.int64)


class DemoCloningCallback(BaseCallback):
    """Take a few behaviour-cloning steps on demonstration pairs after every PPO update.

    Some decisions are never explored: in 1-3 the policy presses A in mid-air
    with probability ~0.9995 just before landing on the tree-top where the
    gap jump must start with a fresh A press, so no rollout ever shows PPO
    the alternative. Cross-entropy on the demonstration's actions moves
    those probabilities directly while PPO keeps optimising the return.

    The steps run when a rollout starts, i.e. after the previous update and
    before any action is sampled, so the log-probabilities PPO stores for the
    rollout still match the policy it then trains. They share the policy's
    optimizer and gradient clipping. TensorBoard gets ``demo/bc_loss`` and
    ``demo/bc_accuracy`` (greedy action == demonstrated action).

    Args:
        observations: Channels-first uint8 observations from ``build_demo_dataset``.
        actions: The demonstrated action for each observation.
        steps: Gradient steps per PPO update.
        batch_size: Pairs sampled per step.
        seed: Seed of the minibatch sampler.
    """

    def __init__(self, observations: np.ndarray, actions: np.ndarray, steps: int = 10,
                 batch_size: int = 256, seed: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.steps = steps
        self.batch_size = batch_size
        self._observations = observations
        self._actions = actions
        self._rng = np.random.default_rng(seed)
        self._obs_tensor = None
        self._act_tensor = None

    def _on_training_start(self) -> None:
        device = self.model.policy.device
        self._obs_tensor = torch.as_tensor(self._observations, device=device)
        self._act_tensor = torch.as_tensor(self._actions, device=device)

    def _on_rollout_start(self) -> None:
        policy = self.model.policy
        policy.set_training_mode(True)
        losses = []
        for _ in range(self.steps):
            idx = torch.as_tensor(self._rng.integers(len(self._actions), size=self.batch_size),
                                  device=policy.device)
            _, log_prob, _ = policy.evaluate_actions(self._obs_tensor[idx], self._act_tensor[idx])
            loss = -log_prob.mean()
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.model.max_grad_norm)
            policy.optimizer.step()
            losses.append(loss.item())
        policy.set_training_mode(False)

        with torch.no_grad():
            correct = 0
            for start in range(0, len(self._actions), 1024):
                obs = self._obs_tensor[start:start + 1024]
                greedy = policy.get_distribution(obs).distribution.probs.argmax(dim=1)
                correct += (greedy == self._act_tensor[start:start + 1024]).sum().item()
        self.logger.record('demo/bc_loss', float(np.mean(losses)))
        self.logger.record('demo/bc_accuracy', correct / len(self._actions))

    def _on_step(self) -> bool:
        return True
