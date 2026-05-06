import numpy as np
from opensimplex import OpenSimplex

from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState


class ElevationStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        cfg = self.config
        w, h = state.width, state.height

        seed_a = int(self.rng.integers(0, 2**32))
        seed_b = int(self.rng.integers(0, 2**32))
        gen = OpenSimplex(seed_a)
        warp_gen = OpenSimplex(seed_b)

        # Base coordinate axes
        q_1d = np.arange(w) / w * cfg.noise_scale  # (w,)
        r_1d = np.arange(h) / h * cfg.noise_scale  # (h,)

        # Domain warp: batch evaluate on the cartesian grid (2 calls instead of 2*w*h)
        warp_x = warp_gen.noise2array(q_1d, r_1d).T * cfg.domain_warp_strength  # (w, h)
        warp_y = warp_gen.noise2array(q_1d + 100.0, r_1d + 100.0).T * cfg.domain_warp_strength

        # Warped coordinates as flat arrays for single-loop FBM
        nx_flat = (q_1d[:, np.newaxis] + warp_x).ravel()  # (w*h,)
        ny_flat = (r_1d[np.newaxis, :] + warp_y).ravel()

        # FBM accumulation via single flat loop (avoids Python double-loop overhead)
        octaves = cfg.noise_octaves
        amps = np.array([cfg.noise_persistence**i for i in range(octaves)])
        freqs = np.array([cfg.noise_lacunarity**i for i in range(octaves)])
        max_val = float(amps.sum())

        noise2 = gen.noise2
        result = np.empty(w * h)
        for i in range(w * h):
            nx, ny = nx_flat[i], ny_flat[i]
            v = 0.0
            for j in range(octaves):
                v += noise2(nx * freqs[j], ny * freqs[j]) * amps[j]
            result[i] = v / max_val
        arr = result.reshape(w, h)

        if cfg.continent_falloff:
            qf = (np.arange(w)[:, np.newaxis] / w - 0.5) * 2
            rf = (np.arange(h)[np.newaxis, :] / h - 0.5) * 2
            dist = np.sqrt(qf**2 + rf**2)
            arr *= np.maximum(0.0, 1.0 - dist)

        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)

        for q in range(w):
            for r in range(h):
                state.hexes[(q, r)].elevation = float(arr[q, r])

        return state
