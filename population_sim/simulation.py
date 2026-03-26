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
        self.next_person_id = 1
        self._initialize_population()

    def _initialize_population(self) -> None:
        for _ in range(self.config.demographics.initial_population):
            age = self.rng.randint(0, 70)
            gender = Gender.FEMALE if self.rng.random() < 0.5 else Gender.MALE
            traits = random_traits(self.rng)
            pathogen_names = [p.name for p in self.config.pathogens]
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
        alive = [p for p in self.population if p.alive]
        if not alive:
            return 0, 0, 0.0

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

            nutrition_delta = (food_ratio_by_region.get(person.region_id, 1.0) - 1.0) * 0.2
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
            if person.age > self.config.demographics.max_age:
                natural_mortality = 1.0

            if person.health < 0.25:
                natural_mortality += (0.25 - person.health) * 0.9

            if self.rng.random() < natural_mortality:
                person.alive = False
                deaths += 1

        newborns = self._generate_births()
        births = len(newborns)
        self.population.extend(newborns)

        alive_now = [p for p in self.population if p.alive]
        self.contact_graph = self._build_contact_graph(alive_now)
        return births, deaths, available_food

    def _generate_births(self) -> list[Individual]:
        cfg = self.config.demographics
        pathogen_names = [p.name for p in self.config.pathogens]
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
        for person in alive:
            if self.rng.random() < mcfg.migration_rate:
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
        for i, person in enumerate(people):
            candidates = people[max(0, i - avg * 2): i] + people[i + 1: i + 1 + avg * 2]
            if not candidates:
                continue
            self.rng.shuffle(candidates)
            for neighbor in candidates[:avg]:
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

