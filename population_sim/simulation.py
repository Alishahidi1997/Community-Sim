from __future__ import annotations

import random
from collections import defaultdict

from population_sim.config import SimulationConfig
from population_sim.disease import MultiDiseaseModel
from population_sim.environment import Environment
from population_sim.models import DiseaseState, Gender, Individual, inherit_traits, random_traits
from population_sim.stats import StatsTracker


class SimulationEngine:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.environments = [Environment(config.environment, self.rng) for _ in range(config.demographics.region_count)]
        self.disease_model = MultiDiseaseModel(config.pathogens, self.rng)
        self.stats = StatsTracker()

        self.population: list[Individual] = []
        self.contact_graph: dict[int, list[int]] = {}
        self.current_era = "hunter-gatherer"
        self.last_step_events: dict[str, int] = {"births": 0, "deaths": 0, "new_infections": 0}
        self.next_person_id = 1
        self._initialize_population()

    def _initialize_population(self) -> None:
        pathogen_names = [p.name for p in self.config.pathogens]
        if self.config.demographics.initial_population == 2:
            founders = [Gender.MALE, Gender.FEMALE]
            for gender in founders:
                traits = random_traits(self.rng)
                person = Individual(
                    person_id=self.next_person_id,
                    age=self.rng.randint(18, 24),
                    gender=gender,
                    region_id=0,
                    alive=True,
                    health=self.rng.uniform(0.8, 1.0),
                    disease_susceptibility=self.rng.uniform(0.8, 1.2),
                    genetic_traits=traits,
                    vaccinated=False,
                    disease_states={name: DiseaseState.SUSCEPTIBLE for name in pathogen_names},
                    immunity_levels={name: 0.0 for name in pathogen_names},
                )
                self.population.append(person)
                self.next_person_id += 1
        else:
            for _ in range(self.config.demographics.initial_population):
                age = self.rng.randint(0, 70)
                gender = Gender.FEMALE if self.rng.random() < 0.5 else Gender.MALE
                traits = random_traits(self.rng)
                person = Individual(
                    person_id=self.next_person_id,
                    age=age,
                    gender=gender,
                    region_id=self.rng.randint(0, self.config.demographics.region_count - 1),
                    alive=True,
                    health=self.rng.uniform(0.6, 1.0),
                    disease_susceptibility=self.rng.uniform(0.6, 1.4),
                    genetic_traits=traits,
                    vaccinated=False,
                    disease_states={name: DiseaseState.SUSCEPTIBLE for name in pathogen_names},
                    immunity_levels={name: 0.0 for name in pathogen_names},
                )
                self.population.append(person)
                self.next_person_id += 1

        self.disease_model.seed_initial_infections(self.population)
        self.contact_graph = self._build_contact_graph([p for p in self.population if p.alive])

    def run(self) -> StatsTracker:
        for year in range(self.config.years):
            births, deaths, available_food = self.step(year)
            self.stats.record(year, self.population, births, deaths, available_food)

            if not any(p.alive for p in self.population):
                break
        return self.stats

    def step(self, year: int) -> tuple[int, int, float]:
        births = 0
        deaths = 0
        new_infections_before = self._count_any_infected()
        alive = [p for p in self.population if p.alive]
        if not alive:
            return 0, 0, 0.0

        era = self._era_profile(year)
        self.current_era = era["name"]
        for env in self.environments:
            env.update()
        available_food = self._available_food_total(alive)
        food_ratio_by_region = self._food_ratio_by_region(alive)

        self._apply_migration(alive)
        self._apply_vaccination_policy(alive, year)
        self.contact_graph = self._rewire_contact_graph(alive, self.contact_graph)
        self.disease_model.apply_transmission_from_contacts(
            alive,
            self.contact_graph,
            self.config.migration.cross_region_contact_rate,
        )

        for person in alive:
            person.age_one_year()

            nutrition_delta = (food_ratio_by_region.get(person.region_id, 1.0) - 1.0) * 0.12 * era["food_effect"]
            resilience = person.genetic_traits.get("resilience", 0.5)
            person.health = max(0.0, min(1.0, person.health + nutrition_delta * (0.8 + resilience)))

            disease_deaths = self.disease_model.apply_progression_and_mutation(person)
            if disease_deaths:
                deaths += disease_deaths
                continue
            self.disease_model.apply_immunity_loss(person)

            natural_mortality = (
                self.config.demographics.natural_mortality_base
                + max(0, person.age - 40) * self.config.demographics.natural_mortality_age_factor
            )
            natural_mortality *= era["mortality_multiplier"]
            if person.age > self.config.demographics.max_age:
                natural_mortality = 1.0

            if person.health < 0.25:
                natural_mortality += (0.25 - person.health) * 0.9

            if self.rng.random() < natural_mortality:
                person.alive = False
                deaths += 1

        newborns = self._generate_births(era["birth_multiplier"])
        births = len(newborns)
        self.population.extend(newborns)

        alive_now = [p for p in self.population if p.alive]
        self.contact_graph = self._build_contact_graph(alive_now)
        new_infections_after = self._count_any_infected()
        self.last_step_events = {
            "births": births,
            "deaths": deaths,
            "new_infections": max(0, new_infections_after - new_infections_before),
        }
        return births, deaths, available_food

    def _generate_births(self, era_birth_multiplier: float = 1.0) -> list[Individual]:
        cfg = self.config.demographics
        pathogen_names = [p.name for p in self.config.pathogens]
        food_ratio_by_region = self._food_ratio_by_region([p for p in self.population if p.alive])
        infection_ratio_by_region = self._infection_ratio_by_region([p for p in self.population if p.alive])
        bcfg = self.config.behavior
        females = [
            p
            for p in self.population
            if p.alive
            and p.gender == Gender.FEMALE
            and cfg.reproductive_age_min <= p.age <= cfg.reproductive_age_max
            and p.health > 0.35
        ]
        males = [
            p
            for p in self.population
            if p.alive
            and p.gender == Gender.MALE
            and cfg.reproductive_age_min <= p.age <= cfg.reproductive_age_max
            and p.health > 0.35
        ]

        if not females or not males:
            return []

        newborns: list[Individual] = []
        for mother in females:
            if self.rng.random() > cfg.partner_match_rate:
                continue

            father = self.rng.choice(males)
            fertility_score = (
                cfg.base_birth_rate
                * (0.6 + mother.genetic_traits.get("fertility", 0.5) * cfg.fertility_trait_weight)
                * (0.6 + father.genetic_traits.get("fertility", 0.5) * cfg.fertility_trait_weight)
                * mother.health
            )
            if bcfg.enabled:
                food_pressure = max(0.0, 1.0 - food_ratio_by_region.get(mother.region_id, 1.0))
                disease_pressure = infection_ratio_by_region.get(mother.region_id, 0.0)
                stress_penalty = (food_pressure + disease_pressure) * bcfg.stress_birth_penalty_weight
                fertility_score *= max(0.15, 1.0 - stress_penalty)
            fertility_score *= era_birth_multiplier
            fertility_score = max(0.0, min(1.0, fertility_score))
            if self.rng.random() >= fertility_score:
                continue

            traits = inherit_traits(mother, father, self.rng)
            child = Individual(
                person_id=self.next_person_id,
                age=0,
                gender=Gender.FEMALE if self.rng.random() < 0.5 else Gender.MALE,
                region_id=mother.region_id,
                alive=True,
                health=self.rng.uniform(0.7, 1.0),
                disease_susceptibility=max(0.2, min(1.8, self.rng.gauss(1.0, 0.2))),
                genetic_traits=traits,
                vaccinated=False,
                disease_states={name: DiseaseState.SUSCEPTIBLE for name in pathogen_names},
                immunity_levels={name: 0.0 for name in pathogen_names},
            )
            self.next_person_id += 1

            if self.rng.random() < cfg.child_mortality:
                child.alive = False
            else:
                newborns.append(child)
        return newborns

    def _available_food_total(self, alive: list[Individual]) -> float:
        by_region = defaultdict(int)
        for person in alive:
            by_region[person.region_id] += 1
        total = 0.0
        for region_id, count in by_region.items():
            total += self.environments[region_id].available_food(count)
        return total

    def _food_ratio_by_region(self, alive: list[Individual]) -> dict[int, float]:
        by_region = defaultdict(int)
        for person in alive:
            by_region[person.region_id] += 1
        ratios: dict[int, float] = {}
        for region_id, count in by_region.items():
            available = self.environments[region_id].available_food(count)
            ratios[region_id] = available / count if count else 0.0
        return ratios

    def _apply_migration(self, alive: list[Individual]) -> None:
        mcfg = self.config.migration
        if not mcfg.enabled or self.config.demographics.region_count <= 1:
            return
        infection_by_region = self._infection_ratio_by_region(alive)
        food_ratio_by_region = self._food_ratio_by_region(alive)
        bcfg = self.config.behavior
        for person in alive:
            migrate_prob = mcfg.migration_rate
            target_region = person.region_id

            if bcfg.enabled:
                best_score = -10.0
                for region_id in range(self.config.demographics.region_count):
                    food_score = food_ratio_by_region.get(region_id, 1.0)
                    inf_score = 1.0 - infection_by_region.get(region_id, 0.0)
                    score = food_score * bcfg.migration_food_weight + inf_score * bcfg.migration_infection_weight
                    if score > best_score:
                        best_score = score
                        target_region = region_id

                current_score = (
                    food_ratio_by_region.get(person.region_id, 1.0) * bcfg.migration_food_weight
                    + (1.0 - infection_by_region.get(person.region_id, 0.0)) * bcfg.migration_infection_weight
                )
                if target_region != person.region_id and best_score > current_score:
                    migrate_prob = min(1.0, migrate_prob * 1.8)

            if self.rng.random() < migrate_prob:
                if bcfg.enabled and target_region != person.region_id:
                    person.region_id = target_region
                else:
                    options = [r for r in range(self.config.demographics.region_count) if r != person.region_id]
                    person.region_id = self.rng.choice(options)

    def _apply_vaccination_policy(self, alive: list[Individual], year: int) -> None:
        vcfg = self.config.vaccination
        if not vcfg.enabled or year < vcfg.start_year:
            return
        eligible = [p for p in alive if (not p.vaccinated) and p.age >= vcfg.min_age]
        self.rng.shuffle(eligible)
        limit = min(vcfg.max_per_year, int(len(alive) * vcfg.annual_coverage_fraction), len(eligible))
        for person in eligible[:limit]:
            person.vaccinated = True
            for pathogen in self.config.pathogens:
                person.immunity_levels[pathogen.name] = max(
                    person.immunity_levels.get(pathogen.name, 0.0),
                    vcfg.effectiveness,
                )

    def _build_contact_graph(self, alive: list[Individual]) -> dict[int, list[int]]:
        graph: dict[int, set[int]] = defaultdict(set)
        avg = max(1, self.config.contact_network.avg_contacts)
        people = alive[:]
        self.rng.shuffle(people)
        bcfg = self.config.behavior
        for i, person in enumerate(people):
            candidates = people[max(0, i - avg * 2): i] + people[i + 1: i + 1 + avg * 2]
            if not candidates:
                continue
            self.rng.shuffle(candidates)
            selected: list[Individual] = []
            if bcfg.enabled:
                safe = [
                    c
                    for c in candidates
                    if not any(state == DiseaseState.INFECTED for state in c.disease_states.values())
                ]
                risky = [c for c in candidates if c not in safe]
                safe_quota = int(avg * bcfg.contact_avoid_infected_bias)
                selected.extend(safe[:safe_quota])
                remaining = avg - len(selected)
                if remaining > 0:
                    selected.extend(risky[:remaining])
                if len(selected) < avg:
                    selected.extend(safe[safe_quota : safe_quota + (avg - len(selected))])
            else:
                selected = candidates[:avg]

            for neighbor in selected[:avg]:
                if neighbor.person_id == person.person_id:
                    continue
                graph[person.person_id].add(neighbor.person_id)
                graph[neighbor.person_id].add(person.person_id)
        return {k: list(v) for k, v in graph.items()}

    def _rewire_contact_graph(
        self,
        alive: list[Individual],
        graph: dict[int, list[int]],
    ) -> dict[int, list[int]]:
        rewire = self.config.contact_network.rewiring_rate
        if rewire <= 0:
            return graph
        ids = [p.person_id for p in alive]
        if not ids:
            return {}
        graph_sets: dict[int, set[int]] = {pid: set(neighbors) for pid, neighbors in graph.items()}
        for pid in ids:
            graph_sets.setdefault(pid, set())
            if self.rng.random() < rewire:
                replacement = self.rng.choice(ids)
                if replacement != pid:
                    graph_sets[pid].add(replacement)
                    graph_sets.setdefault(replacement, set()).add(pid)
        return {k: list(v) for k, v in graph_sets.items()}

    def _infection_ratio_by_region(self, alive: list[Individual]) -> dict[int, float]:
        counts = defaultdict(int)
        infected = defaultdict(int)
        for person in alive:
            counts[person.region_id] += 1
            if any(state == DiseaseState.INFECTED for state in person.disease_states.values()):
                infected[person.region_id] += 1
        return {
            region_id: (infected[region_id] / count if count else 0.0)
            for region_id, count in counts.items()
        }

    def _count_any_infected(self) -> int:
        return sum(
            1
            for p in self.population
            if p.alive and any(state == DiseaseState.INFECTED for state in p.disease_states.values())
        )

    def _era_profile(self, year: int) -> dict[str, float | str]:
        if year < 60:
            return {
                "name": "hunter-gatherer",
                "food_effect": 1.15,
                "birth_multiplier": 1.2,
                "mortality_multiplier": 1.1,
            }
        if year < 120:
            return {
                "name": "agrarian",
                "food_effect": 1.0,
                "birth_multiplier": 1.0,
                "mortality_multiplier": 1.0,
            }
        if year < 170:
            return {
                "name": "industrial",
                "food_effect": 0.9,
                "birth_multiplier": 1.15,
                "mortality_multiplier": 0.85,
            }
        return {
            "name": "modern",
            "food_effect": 0.85,
            "birth_multiplier": 0.92,
            "mortality_multiplier": 0.72,
        }

