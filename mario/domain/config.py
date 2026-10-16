"""Constants that every layer shares: the default level and the facts about the emulator."""

from __future__ import annotations

DEFAULT_ENV_ID = "SuperMarioBros-1-1-v0"

# Two agent steps between reverse-curriculum start states: fine enough that
# each step back asks for only a couple of new decisions, coarse enough that
# the curriculum crosses a ~300-frame route in a few dozen moves.
DEMO_STRIDE = 8
FRAME_SKIP = 4

# SMB1 RAM "float state": 0 while Mario stands on solid ground. Frontier
# snapshots are only taken then, so a restart never begins in a doomed fall.
SMB1_FLOAT_STATE = 0x001D

# Raw screen rows of the SMB1 status bar; the playfield starts at row 32.
HUD_ROWS = 32

# Observation pipeline and episode limits shared by training, evaluation and recording. A saved
# checkpoint expects exactly this observation: changing the stack, the size or the action set
# makes existing checkpoints unloadable.
FRAME_STACK = 4
OBS_SIZE = 84

# Random idle frames after a reset keep a deterministic policy from replaying one trajectory.
NOOP_MAX = 30
# Agent steps without a new furthest x before an episode is cut (truncated, so PPO bootstraps).
STALL_STEPS = 150
MAX_EPISODE_STEPS = 2000
# `mario record` plays a longer episode than training allows, to film a whole run.
RECORD_MAX_STEPS = 5000
SEED = 42

# PPO. Changing any of these changes what a training run learns.
LEARNING_RATE = 1e-4
CLIP_RANGE = 0.2
N_STEPS = 512
GAMMA = 0.9
GAE_LAMBDA = 1.0
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
