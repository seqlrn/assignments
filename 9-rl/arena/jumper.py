"""
JumperEnv — a Chrome-Dino-style jump-and-run environment.

The agent runs at a fixed x position; the world scrolls toward it and obstacles
must be jumped over. The action space is discrete:
    0 = NOOP, 1 = JUMP

Two observation modes:
    "features" -> 5-dim float vector (default; suitable for tabular Q-learning
                  after discretization, and for MLP-based DQN)
    "rgb"     -> (H, W, 3) uint8 array (suitable for CNN-based DQN)

The env intentionally has no external dependencies beyond NumPy. Rendering is
optional and lives in arena.render (pygame).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class Action(IntEnum):
    NOOP = 0
    JUMP = 1


# ---------- world constants ----------
WORLD_W = 160          # logical world width in pixels
WORLD_H = 60           # logical world height in pixels
GROUND_Y = 50          # y coordinate of the ground line
PLAYER_X = 20          # fixed x of the player
PLAYER_W = 6
PLAYER_H = 10
GRAVITY = 0.8
JUMP_VELOCITY = -6.0
BASE_SPEED = 2.0       # initial scroll speed (px / step)
SPEED_GROWTH = 0.0008  # speed gain per step (capped)
MAX_SPEED = 5.0
MIN_GAP = 35           # min horizontal gap between obstacles (px)
MAX_GAP = 90
OBS_MIN_W = 4
OBS_MAX_W = 10
OBS_H = 12             # all obstacles are this tall (low cacti)


@dataclass
class _Obstacle:
    x: float           # left edge
    w: int


class JumperEnv:
    """
    A minimal jump-and-run environment.

    Parameters
    ----------
    obs_mode : "features" or "rgb"
    max_steps : episode timeout (truncation)
    seed : RNG seed
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    action_space_n = 2  # NOOP, JUMP

    # bounds for feature observations (used for discretization / normalization)
    FEATURE_LOW = np.array([0.0, JUMP_VELOCITY, 0.0, OBS_MIN_W, BASE_SPEED], dtype=np.float32)
    FEATURE_HIGH = np.array([PLAYER_H + 30.0, -JUMP_VELOCITY, float(WORLD_W), OBS_MAX_W, MAX_SPEED], dtype=np.float32)
    FEATURE_NAMES = ("player_height", "player_vy", "next_obs_dx", "next_obs_w", "speed")

    def __init__(self, obs_mode: str = "features", max_steps: int = 2000, seed: int | None = None):
        if obs_mode not in ("features", "rgb"):
            raise ValueError(f"obs_mode must be 'features' or 'rgb', got {obs_mode!r}")
        self.obs_mode = obs_mode
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # state
        self._player_y = float(GROUND_Y - PLAYER_H)
        self._player_vy = 0.0
        self._on_ground = True
        self._obstacles: list[_Obstacle] = []
        self._speed = BASE_SPEED
        self._steps = 0
        self._score = 0
        self._renderer = None  # lazy-initialized in render()

    # ------------------------------------------------------------------ API

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the env. Returns (observation, info)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._player_y = float(GROUND_Y - PLAYER_H)
        self._player_vy = 0.0
        self._on_ground = True
        self._obstacles = []
        self._speed = BASE_SPEED
        self._steps = 0
        self._score = 0
        self._spawn_obstacle(initial=True)
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one step. Returns (obs, reward, terminated, truncated, info)."""
        if action not in (0, 1):
            raise ValueError(f"invalid action {action}; expected 0 (NOOP) or 1 (JUMP)")

        # --- player physics
        if action == Action.JUMP and self._on_ground:
            self._player_vy = JUMP_VELOCITY
            self._on_ground = False

        if not self._on_ground:
            self._player_vy += GRAVITY
            self._player_y += self._player_vy
            if self._player_y >= GROUND_Y - PLAYER_H:
                self._player_y = float(GROUND_Y - PLAYER_H)
                self._player_vy = 0.0
                self._on_ground = True

        # --- world scrolls left
        self._speed = min(MAX_SPEED, self._speed + SPEED_GROWTH)
        cleared = 0
        for o in self._obstacles:
            o.x -= self._speed
        # remove obstacles that went off-screen on the left, count them as cleared
        kept = []
        for o in self._obstacles:
            if o.x + o.w >= 0:
                kept.append(o)
            else:
                cleared += 1
        self._obstacles = kept

        # spawn new obstacle if the rightmost one is far enough left
        if not self._obstacles or (WORLD_W - (self._obstacles[-1].x + self._obstacles[-1].w)) >= self._rand_gap():
            self._spawn_obstacle()

        # --- reward & termination
        terminated = self._collides()
        reward = -10.0 if terminated else (0.1 + 1.0 * cleared)
        self._score += cleared

        self._steps += 1
        truncated = (self._steps >= self.max_steps) and not terminated
        return self._observation(), float(reward), bool(terminated), bool(truncated), self._info()

    def render(self, mode: str = "human") -> np.ndarray | None:
        """Render the env. Lazy-imports pygame; returns rgb array if mode='rgb_array'."""
        from arena.render import Renderer  # noqa: WPS433  (lazy import)
        if self._renderer is None:
            self._renderer = Renderer(WORLD_W, WORLD_H, scale=4, headless=(mode == "rgb_array"))
        frame = self._renderer.draw(self._snapshot())
        if mode == "human":
            self._renderer.show()
            return None
        return frame

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ helpers

    def _spawn_obstacle(self, initial: bool = False) -> None:
        w = int(self.rng.integers(OBS_MIN_W, OBS_MAX_W + 1))
        if initial:
            x = float(WORLD_W + 10)  # give the agent some breathing room
        else:
            x = float(WORLD_W)
        self._obstacles.append(_Obstacle(x=x, w=w))

    def _rand_gap(self) -> int:
        return int(self.rng.integers(MIN_GAP, MAX_GAP + 1))

    def _next_obstacle(self) -> _Obstacle | None:
        for o in self._obstacles:
            if o.x + o.w > PLAYER_X:  # still ahead of (or under) the player
                return o
        return None

    def _collides(self) -> bool:
        px0, py0 = PLAYER_X, self._player_y
        px1, py1 = PLAYER_X + PLAYER_W, self._player_y + PLAYER_H
        for o in self._obstacles:
            ox0, oy0 = o.x, GROUND_Y - OBS_H
            ox1, oy1 = o.x + o.w, GROUND_Y
            if px0 < ox1 and px1 > ox0 and py0 < oy1 and py1 > oy0:
                return True
        return False

    def _observation(self) -> np.ndarray:
        if self.obs_mode == "features":
            nxt = self._next_obstacle()
            if nxt is None:
                dx, w = float(WORLD_W), float(OBS_MIN_W)
            else:
                dx = max(0.0, nxt.x - (PLAYER_X + PLAYER_W))
                w = float(nxt.w)
            height_above_ground = (GROUND_Y - PLAYER_H) - self._player_y
            return np.array([height_above_ground, self._player_vy, dx, w, self._speed], dtype=np.float32)
        # rgb
        return self.render(mode="rgb_array")  # type: ignore[return-value]

    def _info(self) -> dict[str, Any]:
        return {"score": self._score, "speed": self._speed, "steps": self._steps}

    def _snapshot(self) -> dict[str, Any]:
        return {
            "player": (PLAYER_X, self._player_y, PLAYER_W, PLAYER_H),
            "obstacles": [(o.x, GROUND_Y - OBS_H, o.w, OBS_H) for o in self._obstacles],
            "ground_y": GROUND_Y,
            "score": self._score,
        }
