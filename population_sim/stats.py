from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
from pathlib import Path
from typing import Dict, List

from population_sim.models import DiseaseState, Individual


@dataclass
class YearStats:
    year: int
    population: int
    births: int
    deaths: int
    avg_age: float
    avg_health: float
    infected: int
    recovered: int
    susceptible: int
    vaccinated: int
    multi_infected: int
    mean_resilience: float
    mean_fertility: float
    mean_immunity: float
    genetic_diversity: float
    region_0: int
    region_1: int
    region_2: int
    food_per_capita: float


class StatsTracker:
    def __init__(self) -> None:
        self.history: List[YearStats] = []

    def record(
        self,
        year: int,
        population: list[Individual],
        births: int,
        deaths: int,
        available_food: float,
    ) -> None:
        alive = [p for p in population if p.alive]
        count = len(alive)
        if count == 0:
            self.history.append(
                YearStats(
                    year=year,
                    population=0,
                    births=births,
                    deaths=deaths,
                    avg_age=0.0,
                    avg_health=0.0,
                    infected=0,
                    recovered=0,
                    susceptible=0,
                    vaccinated=0,
                    multi_infected=0,
                    mean_resilience=0.0,
                    mean_fertility=0.0,
                    mean_immunity=0.0,
                    genetic_diversity=0.0,
                    region_0=0,
                    region_1=0,
                    region_2=0,
                    food_per_capita=0.0,
                )
            )
            return

        def _state_counts(person: Individual) -> tuple[int, int, int]:
            states = list(person.disease_states.values())
            infected_local = sum(1 for s in states if s == DiseaseState.INFECTED)
            recovered_local = sum(1 for s in states if s == DiseaseState.RECOVERED)
            susceptible_local = sum(1 for s in states if s == DiseaseState.SUSCEPTIBLE)
            return infected_local, recovered_local, susceptible_local

        infected = sum(1 for p in alive if _state_counts(p)[0] > 0)
        recovered = sum(1 for p in alive if _state_counts(p)[1] > 0)
        susceptible = sum(1 for p in alive if _state_counts(p)[2] == len(p.disease_states))
        vaccinated = sum(1 for p in alive if p.vaccinated)
        multi_infected = sum(1 for p in alive if _state_counts(p)[0] > 1)

        avg_age = sum(p.age for p in alive) / count
        avg_health = sum(p.health for p in alive) / count
        mean_resilience = sum(p.genetic_traits.get("resilience", 0.5) for p in alive) / count
        mean_fertility = sum(p.genetic_traits.get("fertility", 0.5) for p in alive) / count
        mean_immunity = sum(p.genetic_traits.get("immunity", 0.5) for p in alive) / count
        region_0 = sum(1 for p in alive if p.region_id == 0)
        region_1 = sum(1 for p in alive if p.region_id == 1)
        region_2 = sum(1 for p in alive if p.region_id == 2)

        diversity_components = []
        for trait in ("resilience", "fertility", "immunity"):
            values = [p.genetic_traits.get(trait, 0.5) for p in alive]
            mean_val = sum(values) / count
            var_val = sum((v - mean_val) ** 2 for v in values) / count
            diversity_components.append(var_val)
        genetic_diversity = sum(diversity_components) / len(diversity_components)

        self.history.append(
            YearStats(
                year=year,
                population=count,
                births=births,
                deaths=deaths,
                avg_age=avg_age,
                avg_health=avg_health,
                infected=infected,
                recovered=recovered,
                susceptible=susceptible,
                vaccinated=vaccinated,
                multi_infected=multi_infected,
                mean_resilience=mean_resilience,
                mean_fertility=mean_fertility,
                mean_immunity=mean_immunity,
                genetic_diversity=genetic_diversity,
                region_0=region_0,
                region_1=region_1,
                region_2=region_2,
                food_per_capita=available_food / count if count else 0.0,
            )
        )

    def to_rows(self) -> List[Dict[str, float]]:
        return [asdict(item) for item in self.history]

    def export_csv(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        if not rows:
            return
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

