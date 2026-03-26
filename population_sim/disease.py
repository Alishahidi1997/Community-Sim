from __future__ import annotations

import random
from collections import defaultdict

from population_sim.config import PathogenConfig
from population_sim.models import DiseaseState, Individual


class MultiDiseaseModel:
    def __init__(self, configs: list[PathogenConfig], rng: random.Random) -> None:
        self.configs = configs
        self.rng = rng
        self.pathogen_index = {cfg.name: cfg for cfg in configs}

    def seed_initial_infections(self, population: list[Individual]) -> None:
        for person in population:
            for cfg in self.configs:
                if person.alive and self.rng.random() < cfg.initial_infected_fraction:
                    person.disease_states[cfg.name] = DiseaseState.INFECTED

    def apply_transmission_from_contacts(
        self,
        population: list[Individual],
        contacts: dict[int, list[int]],
        cross_region_contact_rate: float,
    ) -> None:
        alive_map = {p.person_id: p for p in population if p.alive}
        to_infect: dict[str, set[int]] = defaultdict(set)

        for person in alive_map.values():
            for neighbor_id in contacts.get(person.person_id, []):
                neighbor = alive_map.get(neighbor_id)
                if neighbor is None:
                    continue
                weight = 1.0 if neighbor.region_id == person.region_id else cross_region_contact_rate
                if weight <= 0:
                    continue

                for pathogen_name, cfg in self.pathogen_index.items():
                    if neighbor.disease_states.get(pathogen_name) != DiseaseState.INFECTED:
                        continue
                    if person.disease_states.get(pathogen_name) != DiseaseState.SUSCEPTIBLE:
                        continue
                    immunity = person.immunity_levels.get(pathogen_name, 0.0)
                    genetic_immunity = person.genetic_traits.get("immunity", 0.5)
                    vaccination_factor = 1.0 - (0.45 if person.vaccinated else 0.0)
                    risk = (
                        cfg.infection_rate
                        * person.disease_susceptibility
                        * (1.05 - genetic_immunity)
                        * (1.0 - immunity)
                        * vaccination_factor
                        * weight
                    )
                    if self.rng.random() < max(0.0, min(1.0, risk)):
                        to_infect[pathogen_name].add(person.person_id)

        for pathogen_name, ids in to_infect.items():
            for pid in ids:
                alive_map[pid].disease_states[pathogen_name] = DiseaseState.INFECTED

    def apply_progression_and_mutation(self, person: Individual) -> int:
        if not person.alive:
            return 0
        deaths = 0
        for cfg in self.configs:
            state = person.disease_states.get(cfg.name, DiseaseState.SUSCEPTIBLE)
            if state != DiseaseState.INFECTED:
                continue

            death_risk = cfg.mortality_rate * (1.2 - person.genetic_traits.get("resilience", 0.5))
            if self.rng.random() < max(0.0, death_risk):
                person.alive = False
                deaths += 1
                break

            if self.rng.random() < cfg.recovery_rate:
                person.disease_states[cfg.name] = DiseaseState.RECOVERED
                person.immunity_levels[cfg.name] = min(1.0, person.immunity_levels.get(cfg.name, 0.0) + 0.35)
                person.health = min(1.0, person.health + 0.03)
            else:
                person.health = max(0.0, person.health - 0.06)

            if self.rng.random() < cfg.mutation_rate:
                cfg.infection_rate = max(0.01, min(0.95, cfg.infection_rate + self.rng.uniform(-cfg.mutation_strength, cfg.mutation_strength)))
                cfg.mortality_rate = max(0.001, min(0.7, cfg.mortality_rate + self.rng.uniform(-cfg.mutation_strength * 0.5, cfg.mutation_strength * 0.5)))
                cfg.recovery_rate = max(0.01, min(0.95, cfg.recovery_rate + self.rng.uniform(-cfg.mutation_strength, cfg.mutation_strength)))
        return deaths

    def apply_immunity_loss(self, person: Individual) -> None:
        if not person.alive:
            return
        for cfg in self.configs:
            name = cfg.name
            if person.disease_states.get(name) == DiseaseState.RECOVERED:
                person.immunity_levels[name] = max(0.0, person.immunity_levels.get(name, 0.0) - cfg.immunity_loss_rate)
                if person.immunity_levels.get(name, 0.0) <= 0.05:
                    person.disease_states[name] = DiseaseState.SUSCEPTIBLE

