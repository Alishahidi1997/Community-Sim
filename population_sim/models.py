from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Dict


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


class DiseaseState(str, Enum):
    SUSCEPTIBLE = "susceptible"
    INFECTED = "infected"
    RECOVERED = "recovered"


@dataclass
class Individual:
    person_id: int
    age: int
    gender: Gender
    region_id: int
    alive: bool
    health: float
    disease_susceptibility: float
    genetic_traits: Dict[str, float]
    vaccinated: bool
    disease_states: Dict[str, DiseaseState]
    immunity_levels: Dict[str, float]

    def age_one_year(self) -> None:
        self.age += 1


def random_traits(rng: random.Random) -> Dict[str, float]:
    return {
        "resilience": rng.uniform(0.25, 0.95),
        "fertility": rng.uniform(0.25, 0.95),
        "immunity": rng.uniform(0.25, 0.95),
    }


def inherit_traits(
    mother: Individual,
    father: Individual,
    rng: random.Random,
    mutation_std: float = 0.05,
) -> Dict[str, float]:
    child_traits: Dict[str, float] = {}
    for key in mother.genetic_traits:
        mean_val = (mother.genetic_traits[key] + father.genetic_traits.get(key, 0.5)) / 2.0
        value = rng.gauss(mean_val, mutation_std)
        child_traits[key] = max(0.0, min(1.0, value))
    return child_traits

