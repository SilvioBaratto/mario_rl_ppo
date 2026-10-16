"""Stable Baselines3 callbacks for a training run: checkpoints and the reverse curriculum."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """Save the model every ``save_freq`` calls, named by total timesteps.

    Uses the model's timestep counter rather than the call count, so names
    keep increasing across resumed runs.
    """

    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = self.save_path / f"checkpoint_{self.num_timesteps}"
            self.model.save(str(checkpoint_path))
            if self.verbose > 0:
                print(f"Checkpoint saved: {checkpoint_path.name}")
        return True


class ReverseCurriculumCallback(BaseCallback):
    """Run a ``ReverseCurriculum`` against the training workers.

    Reads each finished demo episode's outcome from the VecEnv's reset infos
    (workers report it on the reset that follows the episode), and pushes the
    start frame to every worker whenever it moves. TensorBoard gets
    ``curriculum/<level>_start_frame`` (0 once a level's curriculum is done).
    """

    def __init__(self, curriculum, logger, verbose=0):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.log = logger

    def _broadcast(self):
        for level, frame in self.curriculum.cursors().items():
            self.training_env.env_method("set_demo_cursor", level, frame)

    def _on_training_start(self):
        self._broadcast()

    def _on_step(self):
        # Only SubprocVecEnv/DummyVecEnv store reset infos; the wrappers SB3
        # and create_vec_env add on top keep an unused copy of their own.
        reset_infos = self.training_env.unwrapped.reset_infos
        moved = False
        for done, reset_info in zip(self.locals["dones"], reset_infos):
            outcome = reset_info.get("demo_outcome") if done else None
            if outcome is None:
                continue
            counts = self.curriculum.record(outcome)
            if counts is not None:
                moved = True
                frame = self.curriculum.cursors()[outcome["level"]]
                where = (
                    f"frame {frame}" if frame is not None else "the level start (curriculum done)"
                )
                self.log.info(
                    f"Curriculum {outcome['level']}: {counts[0]}/{counts[1]} reached the "
                    f"route end from frame {outcome['frame']} (x={outcome['x']}); "
                    f"now starting from {where} at {self.num_timesteps:,} steps"
                )
        if moved:
            self._broadcast()
        for level, frame in self.curriculum.cursors().items():
            self.logger.record(f"curriculum/{level}_start_frame", frame or 0)
        return True
