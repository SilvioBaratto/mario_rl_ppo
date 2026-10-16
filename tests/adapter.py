"""The only module that knows where the code under test lives.

The golden traces in ``tests/golden`` were captured from the code as it was before the package
restructure. Every test and probe reaches the code through this file, so a move of a module
changes this file and nothing else; a golden that needs to change is a behaviour change and
belongs in its own commit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import evaluate as _evaluate
import find_route as _find_route
import play_and_record as _play_and_record
import train as _train
from agents.demo_cloning import DemoCloningCallback, build_demo_dataset, build_game_demo_dataset
from agents.ppo_agent import MarioCNN, load_mario_model, mario_policy_kwargs, resolve_model_path
from environments.env_factory import (
    DEFAULT_ENV_ID,
    DEMO_ROUTES_DIR,
    DEMO_STRIDE,
    FRAME_SKIP,
    HUD_ROWS,
    NOOP_ACTION,
    SMB1_FLOAT_STATE,
    CustomReward,
    FrontierStart,
    GameProgress,
    GrayResizeObservation,
    LevelStartFrameStack,
    NoopReset,
    ReverseCurriculum,
    SkipFrame,
    StallTruncation,
    assign_levels,
    create_eval_env,
    create_mario_env,
    create_vec_env,
    demo_start_frames,
    demo_warmup_frames,
    episode_end_reason,
    is_single_stage,
    level_key,
    load_demo_route,
    parse_levels,
    replay_demo_route,
    route_level,
)
from utils.logger import setup_logger

ROOT = Path(__file__).resolve().parents[1]
ROUTES_DIR = DEMO_ROUTES_DIR

# Trained on this stack (see the run's system_info.txt); untracked, so tests that need it skip.
CHECKPOINT = ROOT / "runs" / "world1_game3" / "checkpoints" / "checkpoint_17894136.zip"

LEVELS = ("1-1", "1-2", "1-3", "1-4")

SYMBOLS: dict[str, Any] = {
    "CustomReward": CustomReward,
    "DemoCloningCallback": DemoCloningCallback,
    "FrontierStart": FrontierStart,
    "GameProgress": GameProgress,
    "GrayResizeObservation": GrayResizeObservation,
    "LevelStartFrameStack": LevelStartFrameStack,
    "MarioCNN": MarioCNN,
    "NoopReset": NoopReset,
    "ReverseCurriculum": ReverseCurriculum,
    "SkipFrame": SkipFrame,
    "StallTruncation": StallTruncation,
    "assign_levels": assign_levels,
    "build_demo_dataset": build_demo_dataset,
    "build_game_demo_dataset": build_game_demo_dataset,
    "create_eval_env": create_eval_env,
    "create_mario_env": create_mario_env,
    "create_vec_env": create_vec_env,
    "demo_start_frames": demo_start_frames,
    "demo_warmup_frames": demo_warmup_frames,
    "episode_end_reason": episode_end_reason,
    "is_single_stage": is_single_stage,
    "level_key": level_key,
    "load_demo_route": load_demo_route,
    "load_mario_model": load_mario_model,
    "mario_policy_kwargs": mario_policy_kwargs,
    "parse_levels": parse_levels,
    "replay_demo_route": replay_demo_route,
    "resolve_model_path": resolve_model_path,
    "route_level": route_level,
    "setup_logger": setup_logger,
}

CONSTANTS: dict[str, Any] = {
    "CLIP_RANGE": _train.CLIP_RANGE,
    "DEFAULT_ENV_ID": DEFAULT_ENV_ID,
    "DEMO_STRIDE": DEMO_STRIDE,
    "FRAME_SKIP": FRAME_SKIP,
    "HUD_ROWS": HUD_ROWS,
    "LEARNING_RATE": _train.LEARNING_RATE,
    "NOOP_ACTION": NOOP_ACTION,
    "SMB1_FLOAT_STATE": SMB1_FLOAT_STATE,
}

# Smallest argv that each command accepts, to read its argparse defaults.
_CLI: dict[str, tuple[Any, list[str]]] = {
    "train": (_train, []),
    "evaluate": (_evaluate, ["--model-path", "model"]),
    "record": (_play_and_record, ["--model-path", "model"]),
    "find-route": (_find_route, ["--level", "1-1"]),
}


def cli_defaults(command: str) -> dict[str, Any]:
    """Return the parsed namespace of ``command`` when given only its required flags."""
    module, required = _CLI[command]
    saved = sys.argv
    sys.argv = [command, *required]
    try:
        return vars(module.parse_args())
    finally:
        sys.argv = saved


def cli_command(command: str) -> list[str]:
    """Return the argv prefix that runs ``command`` as a subprocess."""
    script = {
        "train": "train.py",
        "evaluate": "evaluate.py",
        "record": "play_and_record.py",
        "find-route": "find_route.py",
    }[command]
    return [sys.executable, str(ROOT / script)]
