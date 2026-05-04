from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .world_state import WorldState

if TYPE_CHECKING:
    from .config import WorldConfig


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

    def run(self) -> WorldState:
        """Run all stages in sequence."""
        from .world_state import WorldState

        state = WorldState.empty(self.seed, self.config.width, self.config.height)
        state.metadata["seed"] = self.seed
        state.metadata["config"] = self.config.__dict__

        for stage_cls, _stage_config in self.stages:
            child_rng = np.random.default_rng(self.rng.integers(0, 2**32))
            stage = stage_cls(self.config, child_rng)
            state = stage.run(state)

        return state
