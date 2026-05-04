"""Render a single random-policy episode of JumperEnv. Needs pygame."""

from __future__ import annotations

import numpy as np

from arena import JumperEnv


def main() -> None:
    env = JumperEnv(seed=0)
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    total = 0.0
    steps = 0
    while True:
        env.render(mode="human")
        a = int(rng.integers(0, 2))
        obs, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
        if term or trunc:
            break
    print(f"random rollout: steps={steps} return={total:.2f} score={info['score']}")
    env.close()


if __name__ == "__main__":
    main()
