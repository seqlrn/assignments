"""
Play JumperEnv with the keyboard. Run before the assignment to build intuition
for what the agent has to learn.

    python -m arena.play

Controls:
    SPACE / UP   jump
    R            restart after death
    ESC / Q      quit

The window prints score, total return, and survived steps in the title bar.
"""

from __future__ import annotations

import sys

from arena import Action, JumperEnv


def main() -> None:
    try:
        import pygame
    except ImportError:
        sys.exit("pygame not installed. Run: pip install pygame")

    env = JumperEnv(seed=None)  # fresh seed each run
    env.render(mode="human")    # opens the window

    obs, _ = env.reset()
    total = 0.0
    steps = 0
    last_score = 0
    best = 0
    dead = False

    while True:
        action = Action.NOOP
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                _bye(best)
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    _bye(best)
                    return
                if ev.key in (pygame.K_SPACE, pygame.K_UP) and not dead:
                    action = Action.JUMP
                if ev.key == pygame.K_r and dead:
                    obs, _ = env.reset()
                    total = 0.0
                    steps = 0
                    last_score = 0
                    dead = False

        if not dead:
            obs, r, term, trunc, info = env.step(action)
            total += r
            steps += 1
            last_score = info["score"]
            if term or trunc:
                dead = True
                best = max(best, last_score)

        env.render(mode="human")
        pygame.display.set_caption(
            f"Arena – Jumper   |   score {last_score}   return {total:6.1f}   "
            f"steps {steps:4d}   best {best}" + ("   [DEAD — press R]" if dead else "")
        )


def _bye(best: int) -> None:
    print(f"thanks for playing — best score this session: {best}")


if __name__ == "__main__":
    main()
