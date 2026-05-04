import math

import numpy as np
from opensimplex import OpenSimplex

from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


def _fbm(
    gen: OpenSimplex, x: float, y: float, octaves: int, persistence: float, lacunarity: float
) -> float:
    value, amplitude, frequency, max_val = 0.0, 1.0, 1.0, 0.0
    for _ in range(octaves):
        value += gen.noise2(x * frequency, y * frequency) * amplitude
        max_val += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return value / max_val


class ElevationStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        cfg = self.config
        w, h = state.width, state.height

        seed_a = int(self.rng.integers(0, 2**32))
        seed_b = int(self.rng.integers(0, 2**32))
        gen = OpenSimplex(seed_a)
        warp_gen = OpenSimplex(seed_b)

        arr = np.zeros((w, h))

        for q in range(w):
            for r in range(h):
                nx = q / w * cfg.noise_scale
                ny = r / h * cfg.noise_scale

                warp_x = warp_gen.noise2(nx, ny) * cfg.domain_warp_strength
                warp_y = warp_gen.noise2(nx + 100.0, ny + 100.0) * cfg.domain_warp_strength

                value = _fbm(
                    gen,
                    nx + warp_x,
                    ny + warp_y,
                    cfg.noise_octaves,
                    cfg.noise_persistence,
                    cfg.noise_lacunarity,
                )

                if cfg.continent_falloff:
                    dist = math.sqrt(((q / w - 0.5) * 2) ** 2 + ((r / h - 0.5) * 2) ** 2)
                    falloff = max(0.0, 1.0 - dist)
                    value *= falloff

                arr[q, r] = value

        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)

        for q in range(w):
            for r in range(h):
                state.hexes[(q, r)].elevation = float(arr[q, r])

        return state
