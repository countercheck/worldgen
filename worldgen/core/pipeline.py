import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .world_state import WorldState

if TYPE_CHECKING:
    from .config import WorldConfig

# (index, total, stage class name, seconds taken or None if the stage is just starting)
StageReporter = Callable[[int, int, str, float | None], None]


class GeneratorStage(ABC):
    """Base class for pipeline stages."""

    def __init__(self, config: "WorldConfig", rng: np.random.Generator):
        self.config = config
        self.rng = rng

    @abstractmethod
    def run(self, state: WorldState) -> WorldState:
        """Transform world state. Return modified state."""
        pass


class GeneratorPipeline:
    """Orchestrates a sequence of generation stages."""

    def __init__(self, seed: int, config: "WorldConfig"):
        self.seed = seed
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.stages: list[tuple[type[GeneratorStage], dict]] = []

    def add_stage(
        self, stage_cls: type[GeneratorStage], stage_config: dict | None = None
    ) -> "GeneratorPipeline":
        """Add a stage to the pipeline."""
        self.stages.append((stage_cls, stage_config or {}))
        return self

    def run(self, on_stage: "StageReporter | None" = None) -> WorldState:
        """Run all stages in sequence.

        *on_stage*, if given, is called twice per stage — once as it starts with
        `elapsed=None`, and once when it returns with the seconds it took. A pipeline this
        slow looks hung without it: a 96x96 organic run spends most of its time inside two
        stages, and until one of them printed something there was no way to tell a long
        stage from a wedged one.

        It is a callback rather than a print because `core/` does no I/O — what the
        progress *looks* like is the caller's business, and `cli.py` is the only caller
        that wants it on. Timing is measured here anyway; it costs nothing and it is the
        only place that can see a stage boundary.
        """
        from .world_state import WorldState

        state = WorldState.empty(
            self.seed, self.config.width, self.config.height, self.config.grid_layout
        )
        state.metadata["seed"] = self.seed
        state.metadata["config"] = self.config.__dict__

        total = len(self.stages)
        for index, (stage_cls, _stage_config) in enumerate(self.stages, start=1):
            # Drawn before anything else in the loop, as it always has been. The child
            # seed is a function of how many times this generator has been called, so
            # anything that moved a draw would change every world made since.
            child_rng = np.random.default_rng(self.rng.integers(0, 2**32))
            stage = stage_cls(self.config, child_rng)
            if on_stage is not None:
                on_stage(index, total, stage_cls.__name__, None)
            started = time.perf_counter()
            state = stage.run(state)
            if on_stage is not None:
                on_stage(index, total, stage_cls.__name__, time.perf_counter() - started)

        return state
