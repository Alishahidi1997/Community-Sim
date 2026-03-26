from __future__ import annotations

import random

from population_sim.config import EnvironmentConfig


class Environment:
    def __init__(self, config: EnvironmentConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.food_multiplier = 1.0

    def update(self) -> None:
        swing = self.rng.uniform(
            -self.config.food_variability,
            self.config.food_variability,
        )
        self.food_multiplier = max(0.1, 1.0 + swing - self.config.environmental_stress)

        if self.rng.random() < self.config.shock_probability:
            self.food_multiplier = max(
                0.05,
                self.food_multiplier * (1.0 - self.config.shock_severity),
            )

    def available_food(self, population_size: int) -> float:
        return population_size * self.config.base_food_per_capita * self.food_multiplier

