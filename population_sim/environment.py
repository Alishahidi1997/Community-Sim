from __future__ import annotations

import random

from population_sim.config import EnvironmentConfig


class Environment:
    def __init__(self, config: EnvironmentConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng
        self.food_multiplier = 1.0
        self.resource_richness = {
            "water": rng.uniform(0.35, 1.0),
            "fertile_land": rng.uniform(0.3, 1.0),
            "timber": rng.uniform(0.2, 1.0),
            "ore": rng.uniform(0.15, 1.0),
        }
        self.territory_size = rng.uniform(0.75, 1.45)

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

        # Gradual recovery of timber/ore/fertility after extraction (regrowth, shallow deposits).
        regen = 0.0024 + self.rng.uniform(0, 0.0018)
        for k in self.resource_richness:
            cur = self.resource_richness[k]
            headroom = max(0.0, 1.0 - cur)
            self.resource_richness[k] = min(1.0, max(0.08, cur + regen * (0.35 + 0.65 * headroom)))

    def available_food(self, population_size: int) -> float:
        if population_size <= 0:
            return 0.0
        # Sub-linear scaling creates a soft carrying-capacity effect so food does
        # not grow linearly forever with population.
        effective_population = population_size ** 0.9
        congestion = 1.0 / (1.0 + (population_size / 1200.0) ** 0.65)
        fertile_factor = 0.6 + self.resource_richness["fertile_land"] * 0.4
        water_factor = 0.75 + self.resource_richness["water"] * 0.25
        return (
            effective_population
            * self.config.base_food_per_capita
            * self.food_multiplier
            * self.territory_size
            * fertile_factor
            * water_factor
            * (0.65 + 0.35 * congestion)
        )

    def resource_score(self) -> float:
        return (
            self.resource_richness["water"] * 0.32
            + self.resource_richness["fertile_land"] * 0.3
            + self.resource_richness["timber"] * 0.2
            + self.resource_richness["ore"] * 0.18
        ) * self.territory_size

