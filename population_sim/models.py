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
    knowledge: float
    tool_skill: float
    spiritual_tendency: float
    belief_group: str
    happiness: float
    stress: float
    aggression: float
    ambition: float
    faction: str
    language: str
    personal_tools: int
    books_authored: int
    riding_skill: float  # 0–1; mounts / wheels improve mobility once unlocked culturally
    inventions_made: int  # count of breakthroughs this person originated
    mutation_burden: float
    political_power: float  # 0–1; wealth, skill, age, and office compound over time
    # Inherited trait; combined with config.cognition.world_iq for softmax decision temperature
    cognitive_iq: float
    # Stored value in abstract currency (production, trade, theft, taxes)
    wealth: float
    # Local standing: affects theft outcomes, status goal, institutional trust
    reputation: float
    # Yearly replanned; migration/social use these for trend-aware, goal-directed behavior
    primary_goal: str
    observed_food_ema: float
    observed_food_trend: float
    # Incarceration: years left after end-of-year tick; blocks migration and cuts income.
    jail_years_remaining: int = 0
    # Mutual romantic bond (contact graph); cleared if partner dies or bond breaks.
    love_partner_id: int | None = None
    # Founded a revelatory movement (belief_group becomes way_of_<id>).
    is_prophet: bool = False
    # Caught theft / violent crime; repeated theft can trigger jail.
    crimes_caught_count: int = 0
    # Formal bond with love_partner_id (mutual); cleared on breakup or death of partner.
    is_married: bool = False
    # Where this person lives within their region: hinterland vs denser tiers (coexists across the world).
    # Allowed values: rural, village, town, city (city-tier only when regional settlement is a city).
    living_context: str = "rural"
    # Smoothed diet adequacy (tracks regional food ratio over years); drives chronic stress / health.
    nutrition_ema: float = 1.0

    def age_one_year(self) -> None:
        self.age += 1


def random_traits(rng: random.Random) -> Dict[str, float]:
    return {
        "resilience": rng.uniform(0.25, 0.95),
        "fertility": rng.uniform(0.25, 0.95),
        "immunity": rng.uniform(0.25, 0.95),
    }


def random_social_profile(rng: random.Random) -> tuple[float, float, float, str]:
    beliefs = ["ancestor", "sun", "river", "nature"]
    return (
        rng.uniform(0.1, 0.35),  # knowledge
        rng.uniform(0.1, 0.35),  # tool skill
        rng.uniform(0.2, 0.8),   # spiritual tendency
        rng.choice(beliefs),
    )


def random_emotional_profile(rng: random.Random) -> tuple[float, float, float, float]:
    return (
        rng.uniform(0.45, 0.7),   # happiness
        rng.uniform(0.2, 0.5),    # stress
        rng.uniform(0.1, 0.4),    # aggression
        rng.uniform(0.15, 0.75),  # ambition
    )


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


def inherit_social_profile(
    mother: Individual,
    father: Individual,
    rng: random.Random,
) -> tuple[float, float, float, str]:
    knowledge = max(0.0, min(1.0, rng.gauss((mother.knowledge + father.knowledge) / 2.0, 0.04)))
    tool_skill = max(0.0, min(1.0, rng.gauss((mother.tool_skill + father.tool_skill) / 2.0, 0.04)))
    spiritual = max(
        0.0,
        min(1.0, rng.gauss((mother.spiritual_tendency + father.spiritual_tendency) / 2.0, 0.06)),
    )
    belief_group = mother.belief_group if rng.random() < 0.5 else father.belief_group
    return knowledge, tool_skill, spiritual, belief_group


def inherit_riding(mother: Individual, father: Individual, rng: random.Random) -> float:
    base = (mother.riding_skill + father.riding_skill) / 2.0
    return max(0.0, min(1.0, rng.gauss(base, 0.07)))


def inherit_cognitive_iq(
    mother: Individual,
    father: Individual,
    rng: random.Random,
    diversity: float = 1.0,
) -> float:
    base = (mother.cognitive_iq + father.cognitive_iq) / 2.0
    d = max(0.0, min(1.0, diversity))
    std = 0.035 + 0.085 * d
    return max(0.08, min(0.98, rng.gauss(base, std)))


def inherit_wealth(mother: Individual, father: Individual, rng: random.Random) -> float:
    base = max(0.05, (mother.wealth + father.wealth) / 2.0)
    return max(0.0, rng.gauss(base, base * 0.12 + 0.08))


def inherit_reputation(mother: Individual, father: Individual, rng: random.Random) -> float:
    base = (mother.reputation + father.reputation) / 2.0
    return max(0.0, min(1.0, rng.gauss(base, 0.07)))


def inherit_emotions(
    mother: Individual,
    father: Individual,
    rng: random.Random,
) -> tuple[float, float, float, float]:
    happiness = max(0.0, min(1.0, rng.gauss((mother.happiness + father.happiness) / 2.0, 0.07)))
    stress = max(0.0, min(1.0, rng.gauss((mother.stress + father.stress) / 2.0, 0.07)))
    aggression = max(0.0, min(1.0, rng.gauss((mother.aggression + father.aggression) / 2.0, 0.07)))
    ambition = max(0.0, min(1.0, rng.gauss((mother.ambition + father.ambition) / 2.0, 0.08)))
    return happiness, stress, aggression, ambition

