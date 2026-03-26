from __future__ import annotations

import random
from collections import defaultdict

from population_sim.config import SimulationConfig
from population_sim.disease import MultiDiseaseModel
from population_sim.environment import Environment
from population_sim.models import (
    DiseaseState,
    Gender,
    Individual,
    inherit_emotions,
    inherit_social_profile,
    inherit_traits,
    random_emotional_profile,
    random_social_profile,
    random_traits,
)
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
        self.last_step_events: dict[str, int | list[str]] = {
            "births": 0,
            "deaths": 0,
            "new_infections": 0,
            "stories": [],
            "adjustments": [],
        }
        self.civilization_index = 0.0
        self.cult_count = 0
        self.world_structures: list[dict[str, int | str | float]] = []
        self.agriculture_unlocked = False
        self.timeline_events: list[dict[str, str | int]] = []
        self.city_summaries: list[dict[str, str | int]] = []
        self.major_events: list[dict[str, str | int]] = []
        self.unlocked_milestones: set[str] = set()
        self.food_system_bonus = 0.0
        self.mortality_reduction_bonus = 0.0
        self.knowledge_boost = 0.0
        self.temp_food_penalty = 0.0
        self.temp_birth_penalty = 0.0
        self.temp_mortality_penalty = 0.0
        self.temp_effect_years_left = 0
        self.friendships: set[tuple[int, int]] = set()
        self.enmities: set[tuple[int, int]] = set()
        self.alliances: set[tuple[str, str]] = set()
        self.faction_names = ["River Clan", "Sun Pact", "Iron League", "Oak Circle", "Sky Union"]
        self.language_names = ["Proto", "Asteric", "Vardenic", "Lunari", "Nordic"]
        self.total_tools_crafted = 0
        self.total_books_written = 0
        self.next_person_id = 1
        self._initialize_population()

    def _initialize_population(self) -> None:
        pathogen_names = [p.name for p in self.config.pathogens]
        if self.config.demographics.initial_population == 2:
            founders = [Gender.MALE, Gender.FEMALE]
            for gender in founders:
                traits = random_traits(self.rng)
                knowledge, tool_skill, spiritual, belief_group = random_social_profile(self.rng)
                happiness, stress, aggression = random_emotional_profile(self.rng)
                faction = self.rng.choice(self.faction_names[:2])
                language = self.language_names[0]
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
                    knowledge=knowledge,
                    tool_skill=tool_skill,
                    spiritual_tendency=spiritual,
                    belief_group=belief_group,
                    happiness=happiness,
                    stress=stress,
                    aggression=aggression,
                    faction=faction,
                    language=language,
                    personal_tools=0,
                    books_authored=0,
                    mutation_burden=0.0,
                )
                self.population.append(person)
                self.next_person_id += 1
        else:
            for _ in range(self.config.demographics.initial_population):
                age = self.rng.randint(0, 70)
                gender = Gender.FEMALE if self.rng.random() < 0.5 else Gender.MALE
                traits = random_traits(self.rng)
                knowledge, tool_skill, spiritual, belief_group = random_social_profile(self.rng)
                happiness, stress, aggression = random_emotional_profile(self.rng)
                faction = self.rng.choice(self.faction_names)
                language = self.rng.choice(self.language_names[:3])
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
                    knowledge=knowledge,
                    tool_skill=tool_skill,
                    spiritual_tendency=spiritual,
                    belief_group=belief_group,
                    happiness=happiness,
                    stress=stress,
                    aggression=aggression,
                    faction=faction,
                    language=language,
                    personal_tools=0,
                    books_authored=0,
                    mutation_burden=0.0,
                )
                self.population.append(person)
                self.next_person_id += 1

        self.disease_model.seed_initial_infections(self.population)
        self.contact_graph = self._build_contact_graph([p for p in self.population if p.alive])
        self._initialize_world_structures()

    def run(self) -> StatsTracker:
        for year in range(self.config.years):
            births, deaths, available_food = self.step(year)
            self.stats.record(
                year,
                self.population,
                births,
                deaths,
                available_food,
                friendships=len(self.friendships),
                enmities=len(self.enmities),
            )

            if not any(p.alive for p in self.population):
                break
        return self.stats

    def step(self, year: int) -> tuple[int, int, float]:
        births = 0
        deaths = 0
        stories: list[str] = []
        new_infections_before = self._count_any_infected()
        alive = [p for p in self.population if p.alive]
        if not alive:
            return 0, 0, 0.0

        era = self._era_profile(year)
        self.current_era = era["name"]
        for env in self.environments:
            env.update()
        self._decay_temporal_effects()
        available_food = self._available_food_total(alive)
        available_food *= max(0.2, 1.0 + self.food_system_bonus - self.temp_food_penalty)
        food_ratio_by_region = self._food_ratio_by_region(alive)

        self._apply_migration(alive)
        self._apply_vaccination_policy(alive, year)
        self.contact_graph = self._rewire_contact_graph(alive, self.contact_graph)
        self._prune_social_edges(alive)
        self._simulate_social_dynamics(alive)
        self._apply_social_learning(alive)
        self._craft_tools_and_books(alive, year)
        alliance_stories, alliance_deaths = self._simulate_alliances_and_war(alive, year)
        stories.extend(alliance_stories)
        deaths += alliance_deaths
        self.disease_model.apply_transmission_from_contacts(
            alive,
            self.contact_graph,
            self.config.migration.cross_region_contact_rate,
        )

        for person in alive:
            person.age_one_year()

            nutrition_delta = (food_ratio_by_region.get(person.region_id, 1.0) - 1.0) * 0.12 * era["food_effect"]
            resilience = person.genetic_traits.get("resilience", 0.5)
            productivity = self._individual_productivity(person)
            tech_bonus = 1.0 + (person.tool_skill * 0.25) + productivity * 0.2
            person.health = max(0.0, min(1.0, person.health + nutrition_delta * (0.8 + resilience) * tech_bonus))
            self._apply_biological_mutation(person, year)

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
            natural_mortality *= max(0.6, 1.0 - (person.knowledge * 0.18 + person.tool_skill * 0.12))
            natural_mortality *= 1.0 + self.temp_mortality_penalty - self.mortality_reduction_bonus

            if self.rng.random() < natural_mortality:
                person.alive = False
                deaths += 1

        newborns = self._generate_births(era["birth_multiplier"] * (1.0 - self.temp_birth_penalty))
        births = len(newborns)
        self.population.extend(newborns)

        alive_now = [p for p in self.population if p.alive]
        self.contact_graph = self._build_contact_graph(alive_now)
        self._update_civilization_metrics(alive_now)
        transition_stories = self._update_world_structures(year, alive_now)
        stories.extend(transition_stories)
        adjustments = self._auto_adjust_parameters(alive_now, available_food)
        event_deaths, event_stories = self._process_world_events(year, alive_now)
        deaths += event_deaths
        stories.extend(event_stories)
        new_infections_after = self._count_any_infected()
        self.last_step_events = {
            "births": births,
            "deaths": deaths,
            "new_infections": max(0, new_infections_after - new_infections_before),
            "stories": stories,
            "adjustments": adjustments,
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
            fertility_score *= 1.0 + ((mother.knowledge + father.knowledge) * 0.08)
            fertility_score *= max(0.4, 0.8 + (self._individual_productivity(mother) + self._individual_productivity(father)) * 0.15)
            fertility_score = max(0.0, min(1.0, fertility_score))
            if self.rng.random() >= fertility_score:
                continue

            traits = inherit_traits(mother, father, self.rng)
            knowledge, tool_skill, spiritual, belief_group = inherit_social_profile(mother, father, self.rng)
            happiness, stress, aggression = inherit_emotions(mother, father, self.rng)
            faction = mother.faction if self.rng.random() < 0.5 else father.faction
            language = mother.language if self.rng.random() < 0.65 else father.language
            if "digital_age" in self.unlocked_milestones and self.rng.random() < 0.08:
                language = self.rng.choice(self.language_names)
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
                knowledge=knowledge,
                tool_skill=tool_skill,
                spiritual_tendency=spiritual,
                belief_group=belief_group,
                happiness=happiness,
                stress=stress,
                aggression=aggression,
                faction=faction,
                language=language,
                personal_tools=0,
                books_authored=0,
                mutation_burden=0.0,
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
            modifier = max(0.2, 1.0 + self.food_system_bonus - self.temp_food_penalty)
            ratios[region_id] = (available * modifier) / count if count else 0.0
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
        if year < 120:
            return {
                "name": "hunter-gatherer",
                "food_effect": 1.15,
                "birth_multiplier": 1.2,
                "mortality_multiplier": 1.1,
            }
        if year < 280:
            return {
                "name": "agrarian",
                "food_effect": 1.0,
                "birth_multiplier": 1.0,
                "mortality_multiplier": 1.0,
            }
        if year < 480:
            return {
                "name": "industrial",
                "food_effect": 0.9,
                "birth_multiplier": 1.15,
                "mortality_multiplier": 0.85,
            }
        if year < 760:
            return {
                "name": "modern",
                "food_effect": 0.85,
                "birth_multiplier": 0.92,
                "mortality_multiplier": 0.72,
            }
        return {
            "name": "information age",
            "food_effect": 0.85,
            "birth_multiplier": 0.84,
            "mortality_multiplier": 0.64,
        }

    def _apply_social_learning(self, alive: list[Individual]) -> None:
        if not alive:
            return
        id_map = {p.person_id: p for p in alive}
        for person in alive:
            neighbors = [id_map[n] for n in self.contact_graph.get(person.person_id, []) if n in id_map]
            if not neighbors:
                continue
            teacher = max(neighbors, key=lambda n: n.knowledge + n.tool_skill)
            person.knowledge = min(1.0, person.knowledge + 0.012 * teacher.knowledge + self.knowledge_boost)
            person.tool_skill = min(1.0, person.tool_skill + 0.01 * teacher.tool_skill)

            # Belief diffusion: high-spiritual groups can form stable cults/religions.
            if self.rng.random() < 0.03 * person.spiritual_tendency:
                person.belief_group = teacher.belief_group
            if self.rng.random() < 0.002 and person.spiritual_tendency > 0.7:
                person.belief_group = f"cult_{person.person_id}"

    def _update_civilization_metrics(self, alive: list[Individual]) -> None:
        if not alive:
            self.civilization_index = 0.0
            self.cult_count = 0
            return
        avg_knowledge = sum(p.knowledge for p in alive) / len(alive)
        avg_tools = sum(p.tool_skill for p in alive) / len(alive)
        self.civilization_index = (avg_knowledge * 0.55) + (avg_tools * 0.45)
        cult_names = {p.belief_group for p in alive if p.belief_group.startswith("cult_")}
        self.cult_count = len(cult_names)

    def _simulate_social_dynamics(self, alive: list[Individual]) -> None:
        cfg = self._conflict_params()
        if self.config.conflict.preset.lower().strip() == "high_conflict":
            for p in alive:
                p.aggression = min(1.0, p.aggression + 0.004)
                p.stress = min(1.0, p.stress + 0.002)
        id_map = {p.person_id: p for p in alive}
        for person in alive:
            for nid in self.contact_graph.get(person.person_id, []):
                if person.person_id >= nid:
                    continue
                other = id_map.get(nid)
                if other is None:
                    continue
                pair = (person.person_id, other.person_id)
                similarity = 1.0 - abs(person.spiritual_tendency - other.spiritual_tendency)
                trust_signal = (
                    similarity * 0.45
                    + (1.0 - abs(person.happiness - other.happiness)) * 0.25
                    + (1.0 - abs(person.aggression - other.aggression)) * 0.3
                )
                conflict_signal = (person.aggression + other.aggression) * 0.5 + abs(person.stress - other.stress) * 0.5

                faction_tension = cfg["faction_tension"] if person.faction != other.faction else -0.02
                language_tension = cfg["language_tension"] if person.language != other.language else -0.01
                conflict_signal += faction_tension + language_tension

                if trust_signal > cfg["friend_trust_threshold"] and self.rng.random() < cfg["friend_prob"]:
                    self.friendships.add(pair)
                    if pair in self.enmities:
                        self.enmities.discard(pair)
                elif conflict_signal > cfg["enemy_conflict_threshold"] and self.rng.random() < cfg["enemy_prob"]:
                    self.enmities.add(pair)
                    if pair in self.friendships:
                        self.friendships.discard(pair)

                if pair in self.friendships:
                    person.happiness = min(1.0, person.happiness + 0.01)
                    other.happiness = min(1.0, other.happiness + 0.01)
                    person.stress = max(0.0, person.stress - 0.008)
                    other.stress = max(0.0, other.stress - 0.008)
                if pair in self.enmities:
                    person.stress = min(1.0, person.stress + 0.012)
                    other.stress = min(1.0, other.stress + 0.012)
                    person.aggression = min(1.0, person.aggression + 0.005)
                    other.aggression = min(1.0, other.aggression + 0.005)

    def _prune_social_edges(self, alive: list[Individual]) -> None:
        alive_ids = {p.person_id for p in alive}
        self.friendships = {pair for pair in self.friendships if pair[0] in alive_ids and pair[1] in alive_ids}
        self.enmities = {pair for pair in self.enmities if pair[0] in alive_ids and pair[1] in alive_ids}

    def _individual_productivity(self, person: Individual) -> float:
        friend_count = sum(1 for a, b in self.friendships if a == person.person_id or b == person.person_id)
        enemy_count = sum(1 for a, b in self.enmities if a == person.person_id or b == person.person_id)
        social_bonus = min(0.2, friend_count * 0.01) - min(0.18, enemy_count * 0.015)
        emotional = person.happiness * 0.45 - person.stress * 0.35 - person.aggression * 0.15
        skill = person.knowledge * 0.22 + person.tool_skill * 0.28
        assets = min(0.15, person.personal_tools * 0.01 + person.books_authored * 0.015)
        mutation_penalty = min(0.25, person.mutation_burden * 0.3)
        return max(-0.45, min(0.9, social_bonus + emotional + skill + assets - mutation_penalty))

    def _process_world_events(self, year: int, alive: list[Individual]) -> tuple[int, list[str]]:
        stories: list[str] = []
        event_deaths = 0
        pop = len(alive)
        if pop == 0:
            return 0, stories

        avg_knowledge = sum(p.knowledge for p in alive) / pop
        avg_tools = sum(p.tool_skill for p in alive) / pop
        belief_groups = len({p.belief_group for p in alive})
        stories.extend(self._historical_milestones(year, pop, avg_knowledge, avg_tools))

        if "fire" not in self.unlocked_milestones and avg_tools > 0.25 and pop >= 3:
            self.unlocked_milestones.add("fire")
            self.food_system_bonus += 0.06
            stories.append(self._register_event(year, "Fire mastered", "Cooking and warmth improve survival."))
        if "tools" not in self.unlocked_milestones and avg_tools > 0.38 and pop >= 8:
            self.unlocked_milestones.add("tools")
            self.food_system_bonus += 0.05
            stories.append(self._register_event(year, "Advanced tools", "Tool use boosts food gathering efficiency."))
        if "agriculture" not in self.unlocked_milestones and self.current_era in ("agrarian", "industrial", "modern"):
            self.unlocked_milestones.add("agriculture")
            self.food_system_bonus += 0.09
            self.agriculture_unlocked = True
            stories.append(self._register_event(year, "Agriculture emerges", "Farming stabilizes settlements."))
        if "writing" not in self.unlocked_milestones and avg_knowledge > 0.48 and pop >= 20:
            self.unlocked_milestones.add("writing")
            self.knowledge_boost += 0.002
            stories.append(self._register_event(year, "Writing invented", "Knowledge now spreads faster across generations."))
        if "state" not in self.unlocked_milestones and pop >= 90 and self.civilization_index > 0.42:
            self.unlocked_milestones.add("state")
            self.mortality_reduction_bonus += 0.05
            stories.append(self._register_event(year, "City-state formed", "Organized governance improves social stability."))
        if "country" not in self.unlocked_milestones and pop >= 220 and self.civilization_index > 0.58:
            self.unlocked_milestones.add("country")
            self.mortality_reduction_bonus += 0.07
            stories.append(self._register_event(year, "Country founded", "Institutions reduce preventable deaths."))

        # Natural disasters with real effects.
        if self.rng.random() < 0.018:
            self.temp_food_penalty += 0.25
            self.temp_effect_years_left = max(self.temp_effect_years_left, 5)
            stories.append(self._register_event(year, "Great drought", "Harvest collapse reduces food for several years."))
        if self.rng.random() < 0.01:
            self.temp_mortality_penalty += 0.12
            self.temp_effect_years_left = max(self.temp_effect_years_left, 3)
            direct = self._apply_direct_deaths(alive, 0.03, 1)
            event_deaths += direct
            stories.append(self._register_event(year, "Major flood", f"Flooding causes {direct} direct deaths."))
        if self.rng.random() < 0.008:
            self.temp_birth_penalty += 0.2
            self.temp_effect_years_left = max(self.temp_effect_years_left, 4)
            stories.append(self._register_event(year, "Volcanic winter", "Cold years reduce fertility and food."))

        # Conflict events, only after society gets larger/fragmented.
        if pop >= 120 and belief_groups >= 4 and year > 80 and self.rng.random() < 0.025:
            direct = self._apply_direct_deaths(alive, self.rng.uniform(0.04, 0.1), 2)
            event_deaths += direct
            self.temp_birth_penalty += 0.1
            self.temp_effect_years_left = max(self.temp_effect_years_left, 3)
            stories.append(self._register_event(year, "Regional war", f"Conflict breaks out and kills {direct} people."))

        return event_deaths, stories

    def _historical_milestones(
        self,
        year: int,
        population: int,
        avg_knowledge: float,
        avg_tools: float,
    ) -> list[str]:
        stories: list[str] = []

        def trigger(key: str, cond: bool, title: str, details: str, *,
                    food: float = 0.0, mortality: float = 0.0, knowledge: float = 0.0) -> None:
            if key in self.unlocked_milestones or not cond:
                return
            self.unlocked_milestones.add(key)
            self.food_system_bonus += food
            self.mortality_reduction_bonus += mortality
            self.knowledge_boost += knowledge
            stories.append(self._register_event(year, title, details))

        trigger(
            "bronze_age",
            year >= 180 and avg_tools > 0.42 and population >= 40,
            "Bronze craft era",
            "Metal tools improve farming and construction.",
            food=0.03,
        )
        trigger(
            "iron_age",
            year >= 260 and avg_tools > 0.5 and population >= 70,
            "Iron tools spread",
            "Stronger tools increase productivity and defense.",
            food=0.035,
        )
        trigger(
            "road_network",
            year >= 320 and population >= 100 and self.civilization_index > 0.45,
            "Road network expanded",
            "Trade and coordination improve between settlements.",
            knowledge=0.0006,
        )
        trigger(
            "paper_record",
            year >= 430 and avg_knowledge > 0.56 and population >= 120,
            "Record-keeping expands",
            "Administrative memory improves continuity.",
            knowledge=0.001,
        )
        trigger(
            "printing",
            year >= 540 and avg_knowledge > 0.62 and self.total_books_written >= 8,
            "Printing tradition emerges",
            "Books multiply and literacy accelerates.",
            knowledge=0.0018,
        )
        trigger(
            "public_sanitation",
            year >= 620 and population >= 160 and self.civilization_index > 0.62,
            "Public sanitation reforms",
            "Urban health measures reduce preventable disease.",
            mortality=0.05,
        )
        trigger(
            "modern_medicine",
            year >= 710 and avg_knowledge > 0.68 and population >= 220,
            "Modern medicine established",
            "Clinical care lowers mortality and mutation burden impact.",
            mortality=0.08,
        )
        trigger(
            "electric_grid",
            year >= 760 and self.civilization_index > 0.7 and population >= 260,
            "Electric infrastructure built",
            "Powered production and communication boost growth.",
            food=0.02,
            knowledge=0.0012,
        )
        trigger(
            "digital_age",
            year >= 860 and avg_knowledge > 0.76 and self.total_books_written >= 30,
            "Digital communication age",
            "Information access scales across the society.",
            knowledge=0.0022,
        )
        trigger(
            "space_age",
            year >= 940 and self.civilization_index > 0.82 and population >= 320,
            "Early space age",
            "Long-horizon planning improves resilience.",
            mortality=0.04,
        )

        return stories

    def _apply_direct_deaths(self, alive: list[Individual], fraction: float, minimum: int) -> int:
        candidates = [p for p in alive if p.alive]
        if not candidates:
            return 0
        kill_count = min(len(candidates), max(minimum, int(len(candidates) * fraction)))
        self.rng.shuffle(candidates)
        for person in candidates[:kill_count]:
            person.alive = False
        return kill_count

    def _register_event(self, year: int, title: str, details: str) -> str:
        self.major_events.append({"year": year + 1, "title": title, "details": details})
        return f"Y{year + 1}: {title}"

    def _decay_temporal_effects(self) -> None:
        if self.temp_effect_years_left <= 0:
            return
        self.temp_effect_years_left -= 1
        self.temp_food_penalty *= 0.85
        self.temp_birth_penalty *= 0.85
        self.temp_mortality_penalty *= 0.85

    def _initialize_world_structures(self) -> None:
        for region_id in range(self.config.demographics.region_count):
            self.world_structures.append(
                {
                    "id": f"settlement_{region_id}",
                    "name": self._generate_settlement_name(region_id),
                    "kind": "settlement",
                    "level": "camp",
                    "region_id": region_id,
                    "slot": 0.5,
                    "culture": "tribal",
                    "religion": "ancestor",
                }
            )

    def _update_world_structures(self, year: int, alive: list[Individual]) -> list[str]:
        stories: list[str] = []
        pop = len(alive)
        if pop == 0:
            return stories
        avg_knowledge = sum(p.knowledge for p in alive) / pop
        avg_tools = sum(p.tool_skill for p in alive) / pop
        belief_groups = len({p.belief_group for p in alive})
        avg_spiritual = sum(p.spiritual_tendency for p in alive) / pop

        if (not self.agriculture_unlocked) and (self.current_era != "hunter-gatherer" or avg_tools > 0.42):
            self.agriculture_unlocked = True
            stories.append(
                self._register_timeline_transition(
                    year,
                    "Agriculture practice adopted",
                    "Food production shifted from gathering to cultivation.",
                )
            )

        # Settlement progression as concrete state transitions.
        settlement = self._get_settlement_structure(0)
        if settlement is not None:
            level = settlement["level"]
            next_level = None
            if level == "camp" and pop >= 18 and self.agriculture_unlocked:
                next_level = "village"
            elif level == "village" and pop >= 70 and self.civilization_index > 0.32:
                next_level = "town"
            elif level == "town" and pop >= 170 and self.civilization_index > 0.5:
                next_level = "city"
            if next_level:
                settlement["level"] = next_level
                if next_level == "city":
                    settlement["name"] = f"{settlement['name']} City"
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Settlement evolved: {settlement['name']} {level} -> {next_level}",
                        f"Population and organization enabled a {next_level}.",
                    )
                )

        # Agriculture fields become visible structures after agriculture unlock.
        if self.agriculture_unlocked:
            existing_fields = len([s for s in self.world_structures if s["kind"] == "field"])
            target_fields = min(14, max(2, int(pop / 18)))
            while existing_fields < target_fields:
                slot = 0.12 + (existing_fields % 8) * 0.1
                self.world_structures.append(
                    {
                        "id": f"field_{existing_fields}",
                        "kind": "field",
                        "level": 1,
                        "region_id": 0,
                        "slot": max(0.08, min(0.92, slot)),
                    }
                )
                existing_fields += 1
            if existing_fields == target_fields and target_fields >= 2:
                # Register first-time field establishment only once.
                if not any(e["title"] == "Agricultural fields established" for e in self.timeline_events):
                    stories.append(
                        self._register_timeline_transition(
                            year,
                            "Agricultural fields established",
                            "Farms appeared around the settlement.",
                        )
                    )

        # Institutions based on social practice.
        if avg_knowledge > 0.55 and pop >= 80 and not self._has_structure("school"):
            self.world_structures.append(
                {"id": "school_0", "kind": "school", "level": 1, "region_id": 0, "slot": 0.3}
            )
            stories.append(
                self._register_timeline_transition(
                    year,
                    "School founded",
                    "Knowledge density supports formal learning.",
                )
            )
        if avg_tools > 0.58 and pop >= 70 and not self._has_structure("workshop"):
            self.world_structures.append(
                {"id": "workshop_0", "kind": "workshop", "level": 1, "region_id": 0, "slot": 0.68}
            )
            stories.append(
                self._register_timeline_transition(
                    year,
                    "Workshop district formed",
                    "Craft specialization creates workshops.",
                )
            )
        if (avg_spiritual > 0.55 or belief_groups >= 3) and pop >= 65 and not self._has_structure("temple"):
            self.world_structures.append(
                {"id": "temple_0", "kind": "temple", "level": 1, "region_id": 0, "slot": 0.5}
            )
            stories.append(
                self._register_timeline_transition(
                    year,
                    "Temple built",
                    "Shared ritual practices establish a temple.",
                )
            )

        self._refresh_settlement_identities(alive)
        return stories

    def _register_timeline_transition(self, year: int, title: str, details: str) -> str:
        record = {"year": year + 1, "title": title, "details": details}
        self.timeline_events.append(record)
        self.major_events.append(record)
        return f"Y{year + 1}: {title}"

    def _has_structure(self, kind: str) -> bool:
        return any(s["kind"] == kind for s in self.world_structures)

    def _get_settlement_structure(self, region_id: int) -> dict[str, int | str | float] | None:
        for structure in self.world_structures:
            if structure["kind"] == "settlement" and structure["region_id"] == region_id:
                return structure
        return None

    def _generate_settlement_name(self, region_id: int) -> str:
        base = [
            "Aster",
            "Rivermark",
            "Sunhold",
            "Varden",
            "Oakhaven",
            "Northmere",
            "Grayfield",
            "Lunaris",
        ]
        if region_id < len(base):
            return base[region_id]
        return f"Frontier-{region_id}"

    def _refresh_settlement_identities(self, alive: list[Individual]) -> None:
        self.city_summaries = []
        by_region: dict[int, list[Individual]] = defaultdict(list)
        for person in alive:
            by_region[person.region_id].append(person)

        for structure in self.world_structures:
            if structure.get("kind") != "settlement":
                continue
            region_id = int(structure.get("region_id", 0))
            residents = by_region.get(region_id, [])
            if not residents:
                continue
            pop = len(residents)
            avg_knowledge = sum(p.knowledge for p in residents) / pop
            avg_tools = sum(p.tool_skill for p in residents) / pop
            religion = self._dominant_belief(residents)
            faction = self._dominant_label(residents, "faction")
            language = self._dominant_label(residents, "language")
            culture = self._culture_label(avg_knowledge, avg_tools, structure.get("level", "camp"))
            structure["religion"] = religion
            structure["culture"] = culture
            structure["faction"] = faction
            structure["language"] = language
            if structure.get("level") == "city":
                self.city_summaries.append(
                    {
                        "name": str(structure.get("name", "Unknown")),
                        "culture": str(culture),
                        "religion": str(religion),
                        "faction": str(faction),
                        "language": str(language),
                        "population": pop,
                    }
                )

    def _dominant_belief(self, residents: list[Individual]) -> str:
        counts: dict[str, int] = defaultdict(int)
        for p in residents:
            counts[p.belief_group] += 1
        if not counts:
            return "mixed"
        return max(counts.items(), key=lambda x: x[1])[0]

    def _culture_label(self, avg_knowledge: float, avg_tools: float, level: str | int) -> str:
        if level == "camp":
            return "tribal"
        if avg_knowledge > 0.72:
            return "scholastic"
        if avg_tools > 0.72:
            return "industrial"
        if avg_knowledge > 0.55 and avg_tools > 0.55:
            return "civic"
        if level == "village":
            return "agrarian"
        if level == "town":
            return "mercantile"
        return "urban"

    def _dominant_label(self, residents: list[Individual], field: str) -> str:
        counts: dict[str, int] = defaultdict(int)
        for p in residents:
            counts[str(getattr(p, field, "unknown"))] += 1
        if not counts:
            return "mixed"
        return max(counts.items(), key=lambda x: x[1])[0]

    def _auto_adjust_parameters(self, alive: list[Individual], available_food: float) -> list[str]:
        """Autonomous parameter adaptation based on simulation state."""
        if not alive:
            return []

        adjustments: list[str] = []
        pop = len(alive)
        infected = sum(
            1
            for p in alive
            if any(state == DiseaseState.INFECTED for state in p.disease_states.values())
        )
        infection_ratio = infected / pop
        avg_health = sum(p.health for p in alive) / pop
        food_per_capita = available_food / pop if pop else 0.0
        civ = self.civilization_index

        def _nudge_attr(obj, key: str, delta: float, lo: float, hi: float) -> float:
            old = getattr(obj, key)
            new = max(lo, min(hi, old + delta))
            setattr(obj, key, new)
            return new - old

        # Food policy reacts to scarcity/surplus.
        if food_per_capita < 0.92:
            d = _nudge_attr(self.config.environment, "base_food_per_capita", 0.015, 0.2, 2.5)
            if abs(d) > 1e-9:
                adjustments.append("Food policy increases supply")
        elif food_per_capita > 1.35 and pop > 80:
            d = _nudge_attr(self.config.environment, "base_food_per_capita", -0.008, 0.2, 2.5)
            if abs(d) > 1e-9:
                adjustments.append("Food policy trims oversupply")

        # Birth policy responds to stress vs stability.
        if avg_health < 0.45 or food_per_capita < 0.9:
            d = _nudge_attr(self.config.demographics, "base_birth_rate", -0.01, 0.05, 0.95)
            if abs(d) > 1e-9:
                adjustments.append("Birth policy tightened")
        elif avg_health > 0.62 and food_per_capita > 1.0 and pop < 300:
            d = _nudge_attr(self.config.demographics, "base_birth_rate", 0.006, 0.05, 0.95)
            if abs(d) > 1e-9:
                adjustments.append("Birth policy eased")

        # Disease response policy.
        if self.config.pathogens:
            pathogen = self.config.pathogens[0]
            if infection_ratio > 0.12:
                d1 = _nudge_attr(pathogen, "infection_rate", -0.01, 0.01, 0.95)
                d2 = _nudge_attr(self.config.vaccination, "annual_coverage_fraction", 0.01, 0.0, 0.8)
                if abs(d1) > 1e-9 or abs(d2) > 1e-9:
                    adjustments.append("Public health intervention escalates")
            elif infection_ratio < 0.02 and civ > 0.45:
                d = _nudge_attr(self.config.vaccination, "annual_coverage_fraction", -0.004, 0.0, 0.8)
                if abs(d) > 1e-9:
                    adjustments.append("Vaccination pressure relaxed")

        # Migration policy reacts to crowding and civilization maturity.
        if pop > 220 and civ > 0.5:
            d = _nudge_attr(self.config.migration, "migration_rate", 0.003, 0.0, 0.35)
            if abs(d) > 1e-9:
                adjustments.append("Migration policy opens mobility")
        elif pop < 80:
            d = _nudge_attr(self.config.migration, "migration_rate", -0.002, 0.0, 0.35)
            if abs(d) > 1e-9:
                adjustments.append("Migration policy consolidates settlements")

        return adjustments[:3]

    def _apply_biological_mutation(self, person: Individual, year: int) -> None:
        pre_medicine = "country" not in self.unlocked_milestones
        base_rate = 0.006 if pre_medicine else 0.0015
        if self.rng.random() >= base_rate:
            return
        delta = self.rng.gauss(0.0, 0.06)
        person.mutation_burden = max(0.0, min(1.0, person.mutation_burden + abs(delta) * 0.35))
        # Most mutations are neutral/slightly harmful; some are beneficial.
        person.genetic_traits["resilience"] = max(0.0, min(1.0, person.genetic_traits.get("resilience", 0.5) + delta))
        person.genetic_traits["immunity"] = max(0.0, min(1.0, person.genetic_traits.get("immunity", 0.5) + delta * 0.7))
        person.disease_susceptibility = max(0.2, min(2.2, person.disease_susceptibility + (-delta * 0.35)))
        person.health = max(0.0, min(1.0, person.health - person.mutation_burden * 0.01))

        if delta > 0.04:
            self.major_events.append(
                {"year": year + 1, "title": "Beneficial mutation spread", "details": "A resilient lineage gained survival advantage."}
            )

    def _craft_tools_and_books(self, alive: list[Individual], year: int) -> None:
        for person in alive:
            if person.tool_skill > 0.45 and self.rng.random() < (0.01 + person.tool_skill * 0.02):
                person.personal_tools += 1
                self.total_tools_crafted += 1

            writing_unlocked = "writing" in self.unlocked_milestones
            if writing_unlocked and person.knowledge > 0.6 and self.rng.random() < (0.004 + person.knowledge * 0.01):
                person.books_authored += 1
                self.total_books_written += 1
                self.knowledge_boost = min(0.01, self.knowledge_boost + 0.00008)
                if person.books_authored == 1 and self.rng.random() < 0.25:
                    self.timeline_events.append(
                        {
                            "year": year + 1,
                            "title": "Book authored",
                            "details": "Recorded knowledge increases long-term learning capacity.",
                        }
                    )

    def _simulate_alliances_and_war(self, alive: list[Individual], year: int) -> tuple[list[str], int]:
        stories: list[str] = []
        deaths = 0
        cfg = self._conflict_params()
        if len(alive) < 40:
            return stories, deaths

        by_belief: dict[str, list[Individual]] = defaultdict(list)
        for p in alive:
            by_belief[p.belief_group].append(p)
        groups = sorted(by_belief.keys())
        if len(groups) < 2:
            return stories, deaths

        # Form alliances when groups share low aggression and similar stress.
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1 :]:
                key = tuple(sorted((g1, g2)))
                if key in self.alliances:
                    continue
                a1 = sum(p.aggression for p in by_belief[g1]) / len(by_belief[g1])
                a2 = sum(p.aggression for p in by_belief[g2]) / len(by_belief[g2])
                s1 = sum(p.stress for p in by_belief[g1]) / len(by_belief[g1])
                s2 = sum(p.stress for p in by_belief[g2]) / len(by_belief[g2])
                if (a1 + a2) < 0.72 and abs(s1 - s2) < 0.2 and self.rng.random() < cfg["alliance_prob"]:
                    self.alliances.add(key)
                    msg = self._register_timeline_transition(
                        year,
                        "Alliance formed",
                        f"Groups {g1} and {g2} formed a defensive pact.",
                    )
                    stories.append(msg)

        # Need-based war: high stress + low food + group fragmentation, reduced by alliances.
        pop = len(alive)
        avg_stress = sum(p.stress for p in alive) / pop
        food_pc = self._available_food_total(alive) / pop if pop else 1.0
        alliance_factor = max(0.0, 1.0 - len(self.alliances) * cfg["alliance_damp"])
        war_pressure = max(0.0, (avg_stress - 0.45) + max(0.0, 1.0 - food_pc) + (len(groups) - 2) * 0.08)
        war_chance = min(cfg["war_cap"], war_pressure * cfg["war_scale"] * alliance_factor)
        if year > 70 and self.rng.random() < war_chance:
            casualty_fraction = min(0.12, 0.02 + war_pressure * 0.04)
            deaths = self._apply_direct_deaths(alive, casualty_fraction, 1)
            self.temp_birth_penalty += 0.08
            self.temp_effect_years_left = max(self.temp_effect_years_left, 3)
            msg = self._register_timeline_transition(
                year,
                "Resource war",
                f"Conflict over scarcity causes {deaths} deaths.",
            )
            stories.append(msg)

        return stories, deaths

    def _conflict_params(self) -> dict[str, float]:
        preset = self.config.conflict.preset.lower().strip()
        if preset == "high_conflict":
            return {
                "friend_trust_threshold": 0.78,
                "friend_prob": 0.03,
                "enemy_conflict_threshold": 0.48,
                "enemy_prob": 0.22,
                "alliance_prob": 0.006,
                "alliance_damp": 0.03,
                "war_scale": 0.14,
                "war_cap": 0.36,
                "faction_tension": 0.16,
                "language_tension": 0.12,
            }
        return {
            "friend_trust_threshold": 0.65,
            "friend_prob": 0.08,
            "enemy_conflict_threshold": 0.62,
            "enemy_prob": 0.06,
            "alliance_prob": 0.02,
            "alliance_damp": 0.06,
            "war_scale": 0.06,
            "war_cap": 0.18,
            "faction_tension": 0.08,
            "language_tension": 0.06,
        }

