import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState

_MAX_STEPS = 64
_EVAPORATION = 0.99


class ErosionStage(GeneratorStage):
    def run(self, state: WorldState) -> WorldState:
        cfg = self.config
        w, h = state.width, state.height

        arr = np.zeros((w, h))
        for q in range(w):
            for r in range(h):
                arr[q, r] = state.hexes[(q, r)].elevation

        land_coords = [(q, r) for q in range(w) for r in range(h) if arr[q, r] >= cfg.sea_level]

        if land_coords:
            land_arr = np.array(land_coords)
            indices = self.rng.integers(0, len(land_coords), size=cfg.erosion_iterations)

            for idx in indices:
                sq, sr = land_arr[idx]
                self._drop_particle(arr, float(sq), float(sr), w, h, cfg)

        arr = gaussian_filter(arr, sigma=0.5)

        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)

        for q in range(w):
            for r in range(h):
                state.hexes[(q, r)].elevation = float(arr[q, r])

        return state

    def _drop_particle(self, arr: np.ndarray, px: float, py: float, w: int, h: int, cfg) -> None:
        dir_x, dir_y = 0.0, 0.0
        speed = 1.0
        water = 1.0
        sediment = 0.0

        for _ in range(_MAX_STEPS):
            ci, cj = int(px), int(py)

            if ci < 0 or ci >= w or cj < 0 or cj >= h:
                break
            if arr[ci, cj] < cfg.sea_level:
                arr[ci, cj] += sediment
                break

            # Gradient from 4 neighbors (clamp at edges)
            left = arr[max(ci - 1, 0), cj]
            right = arr[min(ci + 1, w - 1), cj]
            up = arr[ci, max(cj - 1, 0)]
            down = arr[ci, min(cj + 1, h - 1)]
            gx = (right - left) * 0.5
            gy = (down - up) * 0.5

            dir_x = cfg.erosion_inertia * dir_x - (1.0 - cfg.erosion_inertia) * gx
            dir_y = cfg.erosion_inertia * dir_y - (1.0 - cfg.erosion_inertia) * gy

            length = (dir_x**2 + dir_y**2) ** 0.5
            if length < 1e-8:
                break
            dir_x /= length
            dir_y /= length

            new_px = px + dir_x
            new_py = py + dir_y
            ni, nj = int(new_px), int(new_py)

            if ni < 0 or ni >= w or nj < 0 or nj >= h:
                break

            dh = arr[ni, nj] - arr[ci, cj]
            capacity = max(-dh, 0.01) * speed * water * cfg.erosion_capacity

            if sediment > capacity:
                deposit = cfg.erosion_deposition * (sediment - capacity)
                arr[ci, cj] += deposit
                sediment -= deposit
            else:
                erode = min(
                    cfg.erosion_erosion_rate * (capacity - sediment), abs(dh) if dh < 0 else 0.0
                )
                arr[ci, cj] -= erode
                sediment += erode

            speed = max(speed + dh, 0.01)
            water *= _EVAPORATION
            px, py = new_px, new_py
