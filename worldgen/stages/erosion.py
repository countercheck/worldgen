import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState

try:
    import numba as _numba

    _jit = _numba.njit
except ImportError:  # numba optional — fall back to pure Python
    _numba = None  # type: ignore[assignment]

    def _jit(fn):  # type: ignore[misc]
        return fn

_MAX_STEPS = 64
_EVAPORATION = 0.99


@_jit
def _drop_particle(
    arr: np.ndarray,
    channel_affinity: np.ndarray,
    px: float,
    py: float,
    w: int,
    h: int,
    sea_level: float,
    inertia: float,
    capacity: float,
    deposition: float,
    erosion_rate: float,
    affinity_gain: float,
) -> None:
    dir_x, dir_y = 0.0, 0.0
    speed = 1.0
    water = 1.0
    sediment = 0.0

    for _ in range(_MAX_STEPS):
        ci, cj = int(px), int(py)

        if ci < 0 or ci >= w or cj < 0 or cj >= h:
            break
        if arr[ci, cj] < sea_level:
            arr[ci, cj] += sediment
            break

        # Gradient from 4 neighbors (clamp at edges)
        left = arr[max(ci - 1, 0), cj]
        right = arr[min(ci + 1, w - 1), cj]
        up = arr[ci, max(cj - 1, 0)]
        down = arr[ci, min(cj + 1, h - 1)]
        gx = (right - left) * 0.5
        gy = (down - up) * 0.5

        dir_x = inertia * dir_x - (1.0 - inertia) * gx
        dir_y = inertia * dir_y - (1.0 - inertia) * gy

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
        cap = max(-dh, 0.01) * speed * water * capacity

        if sediment > cap:
            deposit = deposition * (sediment - cap)
            arr[ci, cj] += deposit
            sediment -= deposit
        else:
            erode = min(erosion_rate * (cap - sediment), abs(dh) if dh < 0 else 0.0)
            arr[ci, cj] -= erode
            sediment += erode
            if erode > 0.0:
                channel_affinity[ci, cj] += affinity_gain

        speed = max(speed + dh, 0.01)
        water *= _EVAPORATION
        px, py = new_px, new_py


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
            n_land = len(land_coords)
            n_iter = cfg.erosion_iterations
            affinity_interval = cfg.erosion_affinity_update_interval

            # Channel affinity: starts uniform, biases later particles toward established channels
            channel_affinity = np.ones((w, h))

            # Initial sample indices (uniform random)
            indices = self.rng.integers(0, n_land, size=n_iter)

            for step in range(n_iter):
                sq, sr = int(land_arr[indices[step], 0]), int(land_arr[indices[step], 1])
                _drop_particle(
                    arr,
                    channel_affinity,
                    float(sq),
                    float(sr),
                    w,
                    h,
                    cfg.sea_level,
                    cfg.erosion_inertia,
                    cfg.erosion_capacity,
                    cfg.erosion_deposition,
                    cfg.erosion_erosion_rate,
                    cfg.erosion_channel_affinity_gain,
                )

                # Periodically re-weight remaining indices toward established channels
                if affinity_interval > 0 and step > 0 and step % affinity_interval == 0:
                    remaining = n_iter - step - 1
                    if remaining > 0:
                        land_weights = channel_affinity[land_arr[:, 0], land_arr[:, 1]]
                        land_weights = land_weights / land_weights.sum()
                        indices[step + 1 :] = self.rng.choice(
                            n_land, size=remaining, p=land_weights
                        )

        arr = gaussian_filter(arr, sigma=0.5)

        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)

        for q in range(w):
            for r in range(h):
                state.hexes[(q, r)].elevation = float(arr[q, r])

        return state
