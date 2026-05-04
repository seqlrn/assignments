"""Pygame renderer for JumperEnv. Imported lazily so the env stays headless-friendly."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


class Renderer:
    def __init__(self, world_w: int, world_h: int, scale: int = 4, headless: bool = False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame  # local import; only required when rendering
        pygame.init()
        self.pygame = pygame
        self.scale = scale
        self.size = (world_w * scale, world_h * scale)
        if headless:
            self.surface = pygame.Surface(self.size)
            self.window = None
        else:
            self.window = pygame.display.set_mode(self.size)
            pygame.display.set_caption("Arena – Jumper")
            self.surface = self.window
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("monospace", 12 * scale // 2)
        except (NotImplementedError, Exception):
            # pygame.font is broken on some Python builds (e.g. circular-import bug
            # in pygame 2.6.x on Python 3.14). The score overlay is cosmetic; skip it.
            self.font = None

    def draw(self, snap: dict[str, Any]) -> np.ndarray:
        s = self.scale
        self.surface.fill((247, 247, 247))
        # ground line
        self.pygame.draw.line(
            self.surface, (40, 40, 40),
            (0, snap["ground_y"] * s), (self.size[0], snap["ground_y"] * s), 2,
        )
        # obstacles
        for (x, y, w, h) in snap["obstacles"]:
            self.pygame.draw.rect(self.surface, (30, 130, 30),
                                  self.pygame.Rect(int(x * s), int(y * s), int(w * s), int(h * s)))
        # player
        px, py, pw, ph = snap["player"]
        self.pygame.draw.rect(self.surface, (200, 60, 60),
                              self.pygame.Rect(int(px * s), int(py * s), int(pw * s), int(ph * s)))
        # score (only if a font is available; the window title bar carries it too)
        if self.font is not None:
            text = self.font.render(f"score: {snap['score']}", True, (40, 40, 40))
            self.surface.blit(text, (8, 8))
        # to numpy (H, W, 3)
        arr = self.pygame.surfarray.array3d(self.surface)
        return np.transpose(arr, (1, 0, 2))

    def show(self) -> None:
        if self.window is None:
            return
        # pump (don't consume) events so the OS sees the app as responsive but
        # play.py / external loops can still read keyboard input from the queue.
        self.pygame.event.pump()
        self.pygame.display.flip()
        self.clock.tick(30)

    def close(self) -> None:
        self.pygame.quit()
