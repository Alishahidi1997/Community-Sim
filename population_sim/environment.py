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
        if population_size <= 0:
            return 0.0
        # Sub-linear scaling creates a soft carrying-capacity effect so food does
        # not grow linearly forever with population.
        effective_population = population_size ** 0.9
        congestion = 1.0 / (1.0 + (population_size / 1200.0) ** 0.65)
        return (
            effective_population
            * self.config.base_food_per_capita
            * self.food_multiplier
            * (0.65 + 0.35 * congestion)
        )

