"""Deterministic probes of the code under test, shared by the golden capture and the tests.

Each probe returns ``{"exact": ..., "platform_bound": ...}``. ``exact`` holds values that only
depend on the emulator and on Python arithmetic, so they must match on every machine.
``platform_bound`` holds hashes of pixels and floating-point tensors, which depend on the OpenCV
and BLAS builds and are compared only on the platform that captured them.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import os
import platform
import subprocess
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from tests import adapter as a

INFO_KEYS = (
    "x_pos",
    "y_pos",
    "world",
    "stage",
    "life",
    "score",
    "coins",
    "time",
    "status",
    "flag_get",
)
VEC_INFO_KEYS = (
    "x_pos",
    "world",
    "stage",
    "life",
    "flag_get",
    "TimeLimit.truncated",
    "stalled",
    "level_start",
)
MARK_EVERY = 50
GAME_ID = "SuperMarioBros-v0"
VEC_LEVELS = (
    "SuperMarioBros-1-1-v0",
    "SuperMarioBros-1-2-v0",
    "SuperMarioBros-1-3-v0",
    GAME_ID,
)


def platform_tag() -> str:
    """Return the tag that decides whether platform-bound values are comparable."""
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def sha(data: bytes) -> str:
    """Return the hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def plain(value: Any) -> Any:
    """Convert numpy scalars and containers into JSON-native values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def result(exact: Any, platform_bound: Any = None) -> dict[str, Any]:
    """Wrap a probe's two value groups."""
    return {"exact": plain(exact), "platform_bound": plain(platform_bound or {})}


def _step_row(action: Any, reward: Any, terminated: bool, truncated: bool, info: dict) -> list:
    extras = sorted(k for k in info if k not in INFO_KEYS)
    return [
        plain(action),
        float(reward).hex(),
        bool(terminated),
        bool(truncated),
        *[plain(info.get(k)) for k in INFO_KEYS],
        extras,
    ]


class _Chain:
    """Rolling hash with a checkpoint every ``MARK_EVERY`` updates, to bisect a divergence."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self.marks: list[str] = []
        self._count = 0

    def update(self, data: bytes) -> None:
        """Fold ``data`` into the hash and keep a mark every ``MARK_EVERY`` updates."""
        self._hash.update(data)
        self._count += 1
        if self._count % MARK_EVERY == 0:
            self.marks.append(self._hash.hexdigest())

    def final(self) -> list[str]:
        """Return the periodic marks followed by the hash of everything seen."""
        return [*self.marks, self._hash.hexdigest()]


def api_surface() -> dict[str, Any]:
    """Public names with parameter names and defaults, public methods, and constants."""
    surface: dict[str, Any] = {}
    for name, obj in sorted(a.SYMBOLS.items()):
        is_class = inspect.isclass(obj)
        target = obj.__init__ if is_class else obj
        params = [
            [p.name, p.kind.name, "<required>" if p.default is p.empty else repr(p.default)]
            for p in inspect.signature(target).parameters.values()
        ]
        entry: dict[str, Any] = {"kind": "class" if is_class else "function", "params": params}
        if is_class:
            entry["methods"] = sorted(
                n for n, m in vars(obj).items() if callable(m) and not n.startswith("_")
            )
            entry["bases"] = [b.__name__ for b in obj.__mro__[1:]]
        surface[name] = entry
    constants = {name: repr(value) for name, value in sorted(a.CONSTANTS.items())}
    return result({"symbols": surface, "constants": constants})


def cli_surface() -> dict[str, Any]:
    """Every command's argparse defaults when only its required flags are given."""
    return result(
        {
            cmd: {k: repr(v) for k, v in sorted(a.cli_defaults(cmd).items())}
            for cmd in ("train", "evaluate", "record", "find-route")
        }
    )


def route_files() -> dict[str, Any]:
    """Byte hashes and shape of the shipped demonstration routes."""
    files = {}
    for path in sorted(a.ROUTES_DIR.glob("*.json")):
        route = json.loads(path.read_text())
        files[path.name] = {
            "sha256": sha(path.read_bytes()),
            "env_id": route["env_id"],
            "frames": len(route["frames"]),
            "agent_actions": len(route.get("agent_actions", [])),
            "waypoints": [[w["frame"], w["x"]] for w in route["waypoints"]],
        }
    return result(files)


def _scripted_progress() -> dict[str, Any]:
    scripts = {
        "clean_two_levels": [
            ({"world": 1, "stage": 1, "life": 2}, False),
            ({"world": 1, "stage": 1, "life": 2, "flag_get": True}, False),
            ({"world": 1, "stage": 2, "life": 2}, False),
            ({"world": 1, "stage": 2, "life": 2, "flag_get": True}, True),
        ],
        "death_then_stall": [
            ({"world": 1, "stage": 1, "life": 2}, False),
            ({"world": 1, "stage": 1, "life": 1}, False),
            ({"world": 1, "stage": 1, "life": 1, "stalled": True}, True),
        ],
        "final_death_wraps_life": [
            ({"world": 1, "stage": 1, "life": 0}, False),
            ({"world": 1, "stage": 1, "life": 255}, True),
        ],
        "time_limit": [
            ({"world": 1, "stage": 1, "life": 2}, False),
            ({"world": 1, "stage": 1, "life": 2, "TimeLimit.truncated": True}, True),
        ],
    }
    out = {}
    for name, steps in scripts.items():
        progress = a.GameProgress()
        for info, done in steps:
            progress.update(info, done)
        out[name] = {
            "levels": progress.levels,
            "lives_lost": progress.lives_lost,
            "clean_levels": progress.clean_levels,
            "levels_cleared": progress.levels_cleared,
            "summary": progress.summary(),
        }
    return out


def _curriculum_walk(warmup: int) -> dict[str, Any]:
    routes = [a.load_demo_route(f"SuperMarioBros-{lv[0]}-{lv[2]}-v0") for lv in a.LEVELS]
    curriculum = a.ReverseCurriculum(routes, warmup_frames=warmup)
    out: dict[str, Any] = {"required": curriculum.required, "start": curriculum.cursors()}
    for level in sorted(curriculum.cursors()):
        moves = 0
        counts = []
        while curriculum.cursors()[level] is not None and moves < 5000:
            cursor = curriculum.cursors()[level]
            moved = curriculum.record({"level": level, "frame": cursor, "success": True})
            if moved:
                moves += 1
                counts.append(list(moved))
        out[level] = {"moves": moves, "counts_head": counts[:3], "end": curriculum.cursors()[level]}
    return out


def pure_tables() -> dict[str, Any]:
    """Outputs of the pure functions and state machines for fixed inputs."""
    errors = {}
    for bad in ("", "9-1", "0-1", "1-5", "1_1", "1-1,", "x"):
        try:
            a.parse_levels(bad)
        except ValueError as exc:
            errors[bad] = str(exc)
    routes = {lv: a.load_demo_route(f"SuperMarioBros-{lv[0]}-{lv[2]}-v0") for lv in a.LEVELS}
    starts = {}
    for lv, route in routes.items():
        for warmup in (0, a.demo_warmup_frames(4)):
            frames = a.demo_start_frames(route, a.DEMO_STRIDE, warmup)
            starts[f"{lv}/w{warmup}"] = [frames[0], frames[-1], len(frames)]
    table = {
        "parse_levels": {
            spec: a.parse_levels(spec)
            for spec in ("1-1", "1-1,1-2,1-3", "game", " 1-1 , game ", "8-4")
        },
        "parse_levels_errors": errors,
        "assign_levels": {
            "mixed_8": a.assign_levels(["a", "b", "c"], 8),
            "single": a.assign_levels("x", 3),
            "repeat": a.assign_levels(["a", "a", "b"], 5),
        },
        "level_key": a.level_key(1, 3),
        "route_level": {lv: a.route_level(r) for lv, r in routes.items()},
        "is_single_stage": {
            env_id: a.is_single_stage(env_id)
            for env_id in ("SuperMarioBros-1-1-v0", GAME_ID, "SuperMarioBros-8-4-v3", "x")
        },
        "episode_end_reason": [
            a.episode_end_reason(info)
            for info in (
                {"stalled": True},
                {"TimeLimit.truncated": True},
                {},
                {"stalled": True, "TimeLimit.truncated": True},
            )
        ],
        "demo_warmup_frames": {str(n): a.demo_warmup_frames(n) for n in (1, 2, 4, 5)},
        "demo_start_frames": starts,
        "game_progress": _scripted_progress(),
        "curriculum": {"w0": _curriculum_walk(0), "w12": _curriculum_walk(12)},
    }
    return result(table)


def raw_route_replay() -> dict[str, Any]:
    """Replay each route's joypad bytes on the bare emulator and record where it ends up."""
    import gym_super_mario_bros

    out: dict[str, Any] = {}
    shots: dict[str, Any] = {}
    for lv in a.LEVELS:
        env_id = f"SuperMarioBros-{lv[0]}-{lv[2]}-v0"
        route = a.load_demo_route(env_id)
        nes = gym_super_mario_bros.make(env_id).unwrapped
        try:
            nes.reset()
            expected = {w["frame"]: w["x"] for w in route["waypoints"]}
            waypoint_x, ram_marks = {}, []
            for frame, byte in enumerate(route["frames"], start=1):
                nes.step(byte)
                if frame in expected:
                    waypoint_x[frame] = int(nes._x_position)
                if frame % 60 == 0:
                    ram_marks.append(sha(nes.ram.tobytes()))
            final_screen = sha(nes.screen.tobytes())
            replays = {}
            for warmup in (0, a.demo_warmup_frames(4)):
                states = a.replay_demo_route(nes, route, warmup_frames=warmup)
                replays[str(warmup)] = {
                    "frames": sorted(states),
                    "x": [states[f][2] for f in sorted(states)],
                }
            out[lv] = {"waypoint_x": waypoint_x, "ram_marks": ram_marks, "replays": replays}
            shots[lv] = final_screen
        finally:
            nes.close()
    return result(out, {"final_screens": shots})


def wrapped_trajectory(level: str, offset: int) -> dict[str, Any]:
    """Play a route's agent actions through the training env after ``offset`` idle frames."""
    env_id = f"SuperMarioBros-{level[0]}-{level[2]}-v0"
    route = a.load_demo_route(env_id)
    env = a.create_mario_env(env_id, seed=42, noop_max=0)
    try:
        frame, _ = env.reset()
        nes = env.unwrapped
        for _ in range(offset):
            nes.step(0)
        if offset:
            frame = env.observation(nes.screen)
        info_chain, obs_chain = _Chain(), _Chain()
        obs_chain.update(frame.tobytes())
        steps, max_x, terminated, truncated, info = 0, 0, False, False, {}
        for action in route["agent_actions"]:
            frame, reward, terminated, truncated, info = env.step(action)
            steps += 1
            max_x = max(max_x, int(info["x_pos"]))
            info_chain.update(
                json.dumps(_step_row(action, reward, terminated, truncated, info)).encode()
            )
            obs_chain.update(frame.tobytes())
            if terminated or truncated:
                break
        goal = route["waypoints"][-1]["x"]
        exact = {
            "steps": steps,
            "max_x": max_x,
            "reached_goal": max_x >= goal,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "flag_get": bool(info.get("flag_get")),
            "info_chain": info_chain.final(),
        }
        return result(exact, {"obs_chain": obs_chain.final()})
    finally:
        env.close()


def game_chain(offset: int) -> dict[str, Any]:
    """Chain the four routes through the whole game, idling through each level end."""
    routes = [a.load_demo_route(f"SuperMarioBros-1-{s}-v0") for s in (1, 2, 3, 4)]
    env = a.create_mario_env(
        GAME_ID,
        noop_max=0,
        stall_steps=0,
        use_custom_reward=False,
        max_steps=sum(len(r["agent_actions"]) for r in routes) + 2000,
    )
    try:
        frame, info = env.reset()
        nes = env.unwrapped
        for _ in range(offset):
            nes.step(0)
        if offset:
            frame = env.observation(nes.screen)
        info_chain, obs_chain = _Chain(), _Chain()
        obs_chain.update(frame.tobytes())
        life = info["life"]
        transcript = []
        for index, route in enumerate(routes):
            level = (info.get("world"), info.get("stage"))
            done, taken, moved_on = False, 0, False
            for action in route["agent_actions"]:
                frame, reward, terminated, truncated, info = env.step(action)
                taken += 1
                info_chain.update(
                    json.dumps(_step_row(action, reward, terminated, truncated, info)).encode()
                )
                obs_chain.update(frame.tobytes())
                done = terminated or truncated or info["life"] < life
                moved_on = (info.get("world"), info.get("stage")) != level
                if info.get("flag_get") or moved_on or done:
                    break
            ok = (bool(info.get("flag_get")) or moved_on) and not info["life"] < life
            idle = 0
            while ok and not done and (info.get("world"), info.get("stage")) == level:
                frame, reward, terminated, truncated, info = env.step(0)
                idle += 1
                info_chain.update(
                    json.dumps(_step_row(0, reward, terminated, truncated, info)).encode()
                )
                obs_chain.update(frame.tobytes())
                done = terminated or truncated
            transcript.append(
                {
                    "route": route["env_id"],
                    "taken": taken,
                    "idle": idle,
                    "ok": ok,
                    "done": done,
                    "world": plain(info.get("world")),
                    "stage": plain(info.get("stage")),
                    "life": plain(info.get("life")),
                    "x_pos": plain(info.get("x_pos")),
                    "level_start": bool(info.get("level_start")),
                }
            )
            if not ok or (done and index + 1 < len(routes)):
                break
        return result(
            {"transcript": transcript, "info_chain": info_chain.final()},
            {"obs_chain": obs_chain.final()},
        )
    finally:
        env.close()


def _wrapper_chain(env: Any) -> list[str]:
    names = []
    while True:
        names.append(type(env).__name__)
        inner = getattr(env, "venv", None) or getattr(env, "env", None)
        if inner is None:
            break
        env = inner
    return names


def vec_rollout(use_subproc: bool, steps: int = 300) -> dict[str, Any]:
    """Step a mixed-level vec env with scripted actions and hash everything it returns."""
    env = a.create_vec_env(
        num_envs=len(VEC_LEVELS),
        env_id=list(VEC_LEVELS),
        seed=42,
        noop_max=30,
        use_subproc=use_subproc,
    )
    try:
        actions = np.random.default_rng(0).integers(0, 7, size=(steps, len(VEC_LEVELS)))
        obs = env.reset()
        obs_chain, reward_chain, info_chain = _Chain(), _Chain(), _Chain()
        obs_chain.update(obs.tobytes())
        dones_at, episodes = [], []
        for t in range(steps):
            obs, rewards, dones, infos = env.step(actions[t])
            obs_chain.update(obs.tobytes())
            reward_chain.update(np.asarray(rewards, dtype=np.float32).tobytes())
            info_chain.update(
                json.dumps([[plain(info.get(k)) for k in VEC_INFO_KEYS] for info in infos]).encode()
            )
            for i, done in enumerate(dones):
                if done:
                    dones_at.append([t, i])
                    episode = infos[i].get("episode")
                    if episode:
                        episodes.append([t, i, float(episode["r"]).hex(), int(episode["l"])])
        exact = {
            "reward_chain": reward_chain.final(),
            "info_chain": info_chain.final(),
            "dones_at": dones_at,
            "episodes": episodes,
            "obs_space": repr(env.observation_space),
        }
        if not use_subproc:
            exact["wrapper_chain"] = _wrapper_chain(env)
        return result(exact, {"obs_chain": obs_chain.final()})
    finally:
        env.close()


def same_rollout(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Tell whether two vec rollouts agree, allowing float32 rounding in the episode returns.

    VecMonitor adds rewards into a float32 buffer; DummyVecEnv hands it float32 rewards and
    SubprocVecEnv float64 ones, so the summed return differs in the last bit between the two.
    """
    one, two = dict(left["exact"]), dict(right["exact"])
    for exact in (one, two):
        exact.pop("wrapper_chain", None)
    episodes_one, episodes_two = one.pop("episodes"), two.pop("episodes")
    if one != two or left["platform_bound"] != right["platform_bound"]:
        return False
    return len(episodes_one) == len(episodes_two) and all(
        x[:2] == y[:2]
        and x[3] == y[3]
        and math.isclose(float.fromhex(x[2]), float.fromhex(y[2]), rel_tol=1e-6)
        for x, y in zip(episodes_one, episodes_two, strict=True)
    )


def frontier_demo_restarts(steps: int = 200) -> dict[str, Any]:
    """Run frontier and demo restarts in-process with a fixed curriculum cursor."""
    r13 = a.load_demo_route("SuperMarioBros-1-3-v0")
    r11 = a.load_demo_route("SuperMarioBros-1-1-v0")
    warmup = a.demo_warmup_frames(4)
    env = a.create_vec_env(
        num_envs=2,
        env_id=["SuperMarioBros-1-3-v0", "SuperMarioBros-1-1-v0"],
        seed=42,
        frontier_prob=0.5,
        demo_starts=True,
        use_subproc=False,
    )
    try:
        env.env_method(
            "set_demo_cursor", "1-3", a.demo_start_frames(r13, a.DEMO_STRIDE, warmup)[10]
        )
        env.env_method(
            "set_demo_cursor", "1-1", a.demo_start_frames(r11, a.DEMO_STRIDE, warmup)[-1]
        )
        base = env
        while hasattr(base, "venv"):
            base = base.venv
        actions = np.random.default_rng(0).integers(0, 7, size=(steps, 2))
        obs = env.reset()
        obs_chain, info_chain = _Chain(), _Chain()
        obs_chain.update(obs.tobytes())
        for t in range(steps):
            obs, rewards, dones, infos = env.step(actions[t])
            obs_chain.update(obs.tobytes())
            row = []
            for i, info in enumerate(infos):
                restart = base.reset_infos[i]
                row.append(
                    [
                        plain(info.get("x_pos")),
                        bool(dones[i]),
                        plain(restart.get("frontier_start")),
                        plain(restart.get("demo_start")),
                        plain(restart.get("demo_outcome")),
                    ]
                )
            info_chain.update(json.dumps(row).encode())
        return result({"info_chain": info_chain.final()}, {"obs_chain": obs_chain.final()})
    finally:
        env.close()


def bc_datasets(max_noop: int = 2) -> dict[str, Any]:
    """Behaviour-cloning datasets built from the shipped routes."""
    routes = [a.load_demo_route(f"SuperMarioBros-1-{s}-v0") for s in (1, 2, 3, 4)]
    exact, bound = {}, {}
    for name, builder in (("single", a.build_demo_dataset), ("game", a.build_game_demo_dataset)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obs, actions = builder(routes, frame_stack=4, max_noop=max_noop)
        exact[name] = {
            "pairs": int(len(actions)),
            "obs_shape": list(obs.shape),
            "actions": sha(actions.tobytes()),
            "warnings": sorted(
                str(w.message) for w in caught if issubclass(w.category, UserWarning)
            ),
        }
        bound[name] = {"obs": sha(obs.tobytes())}
    return result(exact, bound)


def _stub_vec_env() -> Any:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    class _Stub(gym.Env):
        observation_space = spaces.Box(0, 255, (84, 84, 1), np.uint8)
        action_space = spaces.Discrete(7)

        def reset(self, *, seed=None, options=None):
            """Return an all-black frame."""
            return np.zeros((84, 84, 1), np.uint8), {}

        def step(self, action):
            """Return an all-black frame with no reward and no end."""
            return np.zeros((84, 84, 1), np.uint8), 0.0, False, False, {}

    return VecFrameStack(DummyVecEnv([_Stub]), 4, channels_order="last")


def _state_dict_digest(state: dict) -> dict[str, Any]:
    keys = sorted(state)
    layout = [[k, list(state[k].shape)] for k in keys]
    weights = hashlib.sha256()
    for k in keys:
        weights.update(k.encode())
        weights.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return {
        "keys": len(keys),
        "elements": int(sum(state[k].numel() for k in keys)),
        "layout": sha(json.dumps(layout).encode()),
        "weights": weights.hexdigest(),
    }


def checkpoint_contract() -> dict[str, Any]:
    """The reference checkpoint's layout, how it loads, and what it predicts for fixed input."""
    import torch
    from stable_baselines3 import PPO

    with zipfile.ZipFile(a.CHECKPOINT) as zf:
        saved = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)
    venv = _stub_vec_env()
    model = a.load_mario_model(a.CHECKPOINT, env=venv, device="cpu")
    probe_obs = np.random.default_rng(0).integers(0, 256, size=(8, 4, 84, 84), dtype=np.uint8)

    def logits(policy: Any) -> list[list[float]]:
        """Return the action logits of ``policy`` for the fixed probe batch."""
        with torch.no_grad():
            dist = policy.get_distribution(torch.as_tensor(probe_obs))
            return dist.distribution.logits.cpu().numpy().astype(np.float64).tolist()

    fresh = PPO("CnnPolicy", venv, policy_kwargs=a.mario_policy_kwargs(), seed=42, device="cpu")
    kwargs = a.mario_policy_kwargs()
    exact = {
        "zip_sha256": sha(a.CHECKPOINT.read_bytes()),
        "saved": {k: v for k, v in _state_dict_digest(saved).items() if k != "weights"},
        "policy_kwargs": {
            "extractor": kwargs["features_extractor_class"].__name__,
            "extractor_kwargs": kwargs["features_extractor_kwargs"],
            "normalize_images": kwargs["normalize_images"],
        },
        "fresh_layout": _state_dict_digest(fresh.policy.state_dict())["layout"],
        "saved_layout": _state_dict_digest(saved)["layout"],
        "loaded_matches_saved": _state_dict_digest(model.policy.state_dict())["weights"]
        == _state_dict_digest(saved)["weights"],
    }
    bound = {
        "saved_weights": _state_dict_digest(saved)["weights"],
        "fresh_weights": _state_dict_digest(fresh.policy.state_dict())["weights"],
        "loaded_logits": logits(model.policy),
        "fresh_logits": logits(fresh.policy),
    }
    return result(exact, bound)


def policy_behaviour(offsets: tuple[int, ...] = (0, 1, 2, 7, 15, 30)) -> dict[str, Any]:
    """Greedy rollouts of the reference checkpoint on every level for several NOOP offsets."""
    model = a.load_mario_model(a.CHECKPOINT, env=None, device="cpu")
    out: dict[str, Any] = {}
    for lv in a.LEVELS:
        env_id = f"SuperMarioBros-{lv[0]}-{lv[2]}-v0"
        for offset in offsets:
            env = a.create_mario_env(env_id, noop_max=0)
            try:
                frame, _ = env.reset()
                nes = env.unwrapped
                for _ in range(offset):
                    nes.step(0)
                if offset:
                    frame = env.observation(nes.screen)
                stack = np.zeros(frame.shape[:2] + (4,), dtype=np.uint8)
                stack[..., -1:] = frame
                chosen, max_x, steps, flag, ended = [], 0, 0, False, ""
                while True:
                    action, _ = model.predict(stack.transpose(2, 0, 1)[None], deterministic=True)
                    chosen.append(int(action[0]))
                    frame, _, terminated, truncated, info = env.step(int(action[0]))
                    stack = np.roll(stack, -1, axis=-1)
                    stack[..., -1:] = frame
                    steps += 1
                    max_x = max(max_x, int(info["x_pos"]))
                    flag = bool(info.get("flag_get"))
                    if terminated or truncated:
                        ended = "terminated" if terminated else "truncated"
                        break
                out[f"{lv}/{offset}"] = {
                    "steps": steps,
                    "max_x": max_x,
                    "flag": flag,
                    "ended": ended,
                    "actions": sha(bytes(chosen)),
                }
            finally:
                env.close()
    return result(out)


def _run_cli(
    command: str, args: list[str], workdir: Path, timeout: int = 900
) -> subprocess.CompletedProcess:
    env = {**os.environ, **a.cli_env(), "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
    return subprocess.run(
        [*a.cli_command(command), *args],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _final_model_digest(path: Path) -> tuple[int | None, dict[str, Any]]:
    import torch

    with zipfile.ZipFile(path) as zf:
        saved = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)
        steps = json.loads(zf.read("data")).get("num_timesteps")
    return steps, _state_dict_digest(saved)


def train_smoke() -> dict[str, Any]:
    """Train for 2048 steps on two workers, then resume the checkpoint to 4096 steps."""
    base = [
        "--total-timesteps",
        "2048",
        "--num-envs",
        "2",
        "--eval-freq",
        "0",
        "--device",
        "cpu",
        "--seed",
        "42",
    ]
    variants = {
        "plain": ["--exp-name", "smoke_plain"],
        "curriculum": [
            "--exp-name",
            "smoke_curriculum",
            "--levels",
            "1-1,1-3",
            "--frontier-starts",
            "0.5",
            "--demo-starts",
            "--demo-bc",
            "5",
            "--noop-max",
            "2",
        ],
    }
    exact: dict[str, Any] = {}
    bound: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for name, extra in variants.items():
            proc = _run_cli("train", [*base, *extra], workdir)
            run = workdir / "runs" / extra[1] / "checkpoints"
            steps, digest = _final_model_digest(run / "final_model.zip")
            exact[name] = {
                "returncode": proc.returncode,
                "num_timesteps": steps,
                "checkpoints": sorted(p.name for p in run.iterdir()),
                "layout": digest["layout"],
            }
            bound[name] = {"weights": digest["weights"]}
        resumed = _run_cli(
            "train",
            [
                "--resume",
                str(workdir / "runs" / "smoke_plain" / "checkpoints" / "final_model"),
                "--total-timesteps",
                "4096",
                "--num-envs",
                "2",
                "--eval-freq",
                "0",
                "--device",
                "cpu",
                "--seed",
                "42",
                "--exp-name",
                "smoke_resumed",
            ],
            workdir,
        )
        run = workdir / "runs" / "smoke_resumed" / "checkpoints"
        steps, digest = _final_model_digest(run / "final_model.zip")
        exact["resumed"] = {
            "returncode": resumed.returncode,
            "num_timesteps": steps,
            "layout": digest["layout"],
        }
        bound["resumed"] = {"weights": digest["weights"]}
    return result(exact, bound)


def evaluate_cli() -> dict[str, Any]:
    """Evaluate the reference checkpoint on 1-1 for two episodes through the command line."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        proc = _run_cli(
            "evaluate",
            [
                "--model-path",
                str(a.CHECKPOINT),
                "--levels",
                "1-1",
                "--n-episodes",
                "2",
                "--device",
                "cpu",
                "--seed",
                "42",
            ],
            workdir,
        )
        (report,) = workdir.glob("eval_results_ppo_*.json")
        data = json.loads(report.read_text())
    data.pop("timestamp")
    return result({"returncode": proc.returncode, "report": data})
