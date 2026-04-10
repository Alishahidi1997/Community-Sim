from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np

from population_sim.agent_cognition import (
    GOAL_VALUES,
    WorldGoalContext,
    _softmax_sample_index,
    brain_choose_migration_region,
    brain_choose_primary_goal,
    brain_choose_primary_goal_fields,
    effective_cognition_iq,
    goal_migration_multipliers,
    heuristic_goal_logits,
    normalize_pair,
)
from population_sim.learned_policy import (
    LearnedGoalMLP,
    GOAL_POLICY_INPUT_DIM,
    blend_goal_logits,
    build_goal_feature_vector,
)
from population_sim.config import SimulationConfig
from population_sim.economy import (
    apply_income_taxes,
    default_region_policy,
    inter_region_trade,
    pairwise_market_trade,
    region_wealth_medians,
    theft_attempt,
)
from population_sim.disease import MultiDiseaseModel
from population_sim.environment import Environment
from population_sim.models import (
    DiseaseState,
    Gender,
    Individual,
    inherit_cognitive_iq,
    inherit_emotions,
    inherit_reputation,
    inherit_wealth,
    inherit_riding,
    inherit_social_profile,
    inherit_traits,
    random_emotional_profile,
    random_social_profile,
    random_traits,
)
from population_sim.world_realism import sanitation_transmission_multiplier
from population_sim.social_life import (
    followers_by_region,
    has_worship_shrine,
    is_prophet_movement_belief,
    prophet_movement_id,
)
from population_sim.stats import StatsTracker
from population_sim.tech_society import (
    apply_invention_resource_drain,
    apply_tool_craft_drain,
    compute_era_profile,
    format_settlement_polity_name,
    invention_roll_multiplier,
    material_pressure,
    region_can_craft_tools,
    region_meets_invention_minimums,
    settlement_name_stem,
)
from population_sim.world_dynamics import WorldDynamics


class SimulationEngine:
    # Within-region place type: rural hinterland vs village / town / urban core (mixed populations).
    _LIVING_BANDS = ("rural", "village", "town", "city")

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
        # region_id -> government state (leader, elites, elections)
        self.politics_by_region: dict[int, dict[str, object]] = {}
        self._friend_degree_cache: dict[int, int] = {}
        self._enemy_degree_cache: dict[int, int] = {}
        self.region_trade_links: set[tuple[int, int]] = set()
        self.region_food_adjustments: dict[int, float] = {}
        self.world_dynamics = WorldDynamics()
        self._region_food_prev: dict[int, float] = {}
        self._region_food_trend: dict[int, float] = {}
        self._pair_trust: dict[tuple[int, int], float] = {}
        self.current_season_index: int = 0
        self.current_season_name: str = "—"
        self._season_food_mult: float = 1.0
        self._season_migration_mult: float = 1.0
        self._season_disease_mult: float = 1.0
        self._season_wildlife_mult: float = 1.0
        rc = config.demographics.region_count
        self.region_treasury: dict[int, float] = {i: 0.0 for i in range(rc)}
        self.region_harvest_factor: dict[int, float] = {i: 1.0 for i in range(rc)}
        self.region_disaster_years_left: dict[int, int] = {}
        self.region_disaster_food_mult: dict[int, float] = {}
        self.region_policies: dict[int, dict[str, float]] = {
            i: default_region_policy(0) for i in range(rc)
        }
        self.goal_policy: LearnedGoalMLP | None = None
        self._goal_policy_batch: list[dict[str, object]] = []
        if self.config.cognition.learned_goal_network:
            self.goal_policy = LearnedGoalMLP(
                seed=self.config.random_seed + 7919,
                in_dim=GOAL_POLICY_INPUT_DIM,
                hidden=max(8, int(self.config.cognition.learned_goal_hidden)),
            )
        # Cultural tech shared across the world; individuals contribute via invention events.
        self.world_inventions: set[str] = set()
        # Farming/herding/wildlife (abstract headcounts + indices); feeds food and the realtime view.
        self.livestock_by_region: dict[int, float] = {}
        self.wildlife_index: float = 1.0
        self._initialize_population()

    def _initialize_population(self) -> None:
        pathogen_names = [p.name for p in self.config.pathogens]
        if self.config.demographics.initial_population == 2:
            founders = [Gender.MALE, Gender.FEMALE]
            for gender in founders:
                traits = random_traits(self.rng)
                knowledge, tool_skill, spiritual, belief_group = random_social_profile(self.rng)
                happiness, stress, aggression, ambition = random_emotional_profile(self.rng)
                faction = self.rng.choice(self.faction_names[:2])
                language = self.language_names[0]
                h0 = self.rng.uniform(0.8, 1.0)
                age0 = self.rng.randint(18, 24)
                cog_iq = self._sample_birth_cognitive_iq()
                wealth0 = self.rng.uniform(0.45, 1.25)
                rep0 = self.rng.uniform(0.38, 0.68)
                gctx = self._bootstrap_goal_ctx(1.0)
                person = Individual(
                    person_id=self.next_person_id,
                    age=age0,
                    gender=gender,
                    region_id=0,
                    alive=True,
                    health=h0,
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
                    ambition=ambition,
                    faction=faction,
                    language=language,
                    personal_tools=0,
                    books_authored=0,
                    riding_skill=0.0,
                    inventions_made=0,
                    mutation_burden=0.0,
                    political_power=self.rng.uniform(0.14, 0.28),
                    cognitive_iq=cog_iq,
                    wealth=wealth0,
                    reputation=rep0,
                    primary_goal=brain_choose_primary_goal_fields(
                        self.config.cognition.world_iq,
                        cog_iq,
                        health=h0,
                        stress=stress,
                        ambition=ambition,
                        age=age0,
                        happiness=happiness,
                        knowledge=knowledge,
                        tool_skill=tool_skill,
                        observed_food_ema=1.0,
                        observed_food_trend=0.0,
                        wealth=wealth0,
                        reputation=rep0,
                        ctx=gctx,
                        rng=self.rng,
                    ),
                    observed_food_ema=1.0,
                    observed_food_trend=0.0,
                )
                self.population.append(person)
                self.next_person_id += 1
        else:
            for _ in range(self.config.demographics.initial_population):
                age = self.rng.randint(0, 70)
                gender = Gender.FEMALE if self.rng.random() < 0.5 else Gender.MALE
                traits = random_traits(self.rng)
                knowledge, tool_skill, spiritual, belief_group = random_social_profile(self.rng)
                happiness, stress, aggression, ambition = random_emotional_profile(self.rng)
                faction = self.rng.choice(self.faction_names)
                language = self.rng.choice(self.language_names[:3])
                h0 = self.rng.uniform(0.6, 1.0)
                cog_iq = self._sample_birth_cognitive_iq()
                wealth0 = self.rng.uniform(0.32, 1.55)
                rep0 = self.rng.uniform(0.32, 0.72)
                gctx = self._bootstrap_goal_ctx(1.0)
                person = Individual(
                    person_id=self.next_person_id,
                    age=age,
                    gender=gender,
                    region_id=self.rng.randint(0, self.config.demographics.region_count - 1),
                    alive=True,
                    health=h0,
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
                    ambition=ambition,
                    faction=faction,
                    language=language,
                    personal_tools=0,
                    books_authored=0,
                    riding_skill=0.0,
                    inventions_made=0,
                    mutation_burden=0.0,
                    political_power=self.rng.uniform(0.12, 0.32),
                    cognitive_iq=cog_iq,
                    wealth=wealth0,
                    reputation=rep0,
                    primary_goal=brain_choose_primary_goal_fields(
                        self.config.cognition.world_iq,
                        cog_iq,
                        health=h0,
                        stress=stress,
                        ambition=ambition,
                        age=age,
                        happiness=happiness,
                        knowledge=knowledge,
                        tool_skill=tool_skill,
                        observed_food_ema=1.0,
                        observed_food_trend=0.0,
                        wealth=wealth0,
                        reputation=rep0,
                        ctx=gctx,
                        rng=self.rng,
                    ),
                    observed_food_ema=1.0,
                    observed_food_trend=0.0,
                )
                self.population.append(person)
                self.next_person_id += 1

        self.disease_model.seed_initial_infections(self.population)
        self.contact_graph = self._build_contact_graph([p for p in self.population if p.alive])
        self._initialize_world_structures()
        self._bootstrap_living_contexts([p for p in self.population if p.alive])

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

            if self.stats.history and self.stats.history[-1].population == 0:
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

        self._sync_season(year)
        self._tick_regional_harvest_weather()
        stories.extend(self._roll_new_regional_disasters(year))
        pop_n = len(alive)
        avg_stress_w = sum(p.stress for p in alive) / pop_n
        avg_k_w = sum(p.knowledge for p in alive) / pop_n
        avg_t_w = sum(p.tool_skill for p in alive) / pop_n
        civ_w = avg_k_w * 0.55 + avg_t_w * 0.45
        era = self._era_profile(year, civ_w)
        self.current_era = era["name"]
        for env in self.environments:
            env.update()
        self._decay_temporal_effects()
        self.region_food_adjustments = {}
        food_ratios_list = list(self._food_ratio_by_region(alive).values())
        self.world_dynamics.update_global(
            pop_n, avg_stress_w, food_ratios_list, civ_w, self._world_aggression()
        )
        food_ratio_for_culture = self._food_ratio_by_region(alive)
        culture_stories = self._step_culture_inventions_and_riding(alive, year, civ_w, food_ratio_for_culture)
        stories.extend(culture_stories)
        diplomacy_stories, diplomacy_deaths = self._simulate_regional_diplomacy(alive, year)
        stories.extend(diplomacy_stories)
        deaths += diplomacy_deaths
        available_food = self._available_food_total(alive)
        available_food *= max(0.2, 1.0 + self.food_system_bonus - self.temp_food_penalty)
        food_ratio_by_region = self._food_ratio_by_region(alive)
        self._step_regional_food_trends(food_ratio_by_region)
        self._replan_agent_goals(alive, civ_w, food_ratio_by_region, year)
        stories.extend(self._step_civic_unrest(alive, year, food_ratio_by_region))

        self._apply_migration(alive)
        self._update_personal_food_memory(alive)
        self._apply_vaccination_policy(alive, year)
        self.contact_graph = self._rewire_contact_graph(alive, self.contact_graph)
        self._prune_social_edges(alive)
        food_rr_aid = self._food_ratio_by_region(alive)
        self._step_mutual_aid_scarcity(alive, food_rr_aid)
        self._simulate_social_dynamics(alive)
        self._rebuild_social_degree_cache()
        stories.extend(self._step_economy(alive, year, civ_w))
        self._apply_social_learning(alive)
        self._step_belief_evolution(alive)
        stories.extend(self._step_social_life(alive, year))
        self._craft_tools_and_books(alive, year)
        alliance_stories, alliance_deaths = self._simulate_alliances_and_war(alive, year)
        stories.extend(alliance_stories)
        deaths += alliance_deaths
        san_scale = self._sanitation_transmission_scale(alive)
        crowd_scale = self._urban_crowding_disease_mult(alive)
        self.disease_model.apply_transmission_from_contacts(
            alive,
            self.contact_graph,
            self.config.migration.cross_region_contact_rate,
            transmission_scale=self._season_disease_mult * san_scale * crowd_scale,
        )

        id_health = {p.person_id: p for p in alive}
        wrx = self.config.world_realism
        for person in alive:
            person.age_one_year()

            nutrition_delta = (food_ratio_by_region.get(person.region_id, 1.0) - 1.0) * 0.12 * era["food_effect"]
            resilience = person.genetic_traits.get("resilience", 0.5)
            productivity = self._individual_productivity(person)
            tech_bonus = 1.0 + (person.tool_skill * 0.25) + productivity * 0.2
            person.health = max(0.0, min(1.0, person.health + nutrition_delta * (0.8 + resilience) * tech_bonus))
            if (
                wrx.enabled
                and person.age >= wrx.elder_support_age_min
                and person.love_partner_id is not None
            ):
                partner = id_health.get(person.love_partner_id)
                if (
                    partner is not None
                    and partner.alive
                    and partner.region_id == person.region_id
                ):
                    person.health = min(1.0, person.health + wrx.elder_partner_health_bonus)
            if (
                wrx.enabled
                and wrx.work_fatigue_enabled
                and 16 <= person.age < 64
                and person.ambition > 0.52
            ):
                person.stress = min(
                    1.0,
                    person.stress + wrx.work_fatigue_stress * (person.ambition - 0.5) * 1.85,
                )
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

        self._apply_contact_bereavement(alive)

        alive_for_births = [p for p in alive if p.alive]
        newborns, maternal_deaths = self._generate_births(
            alive_for_births, era["birth_multiplier"] * (1.0 - self.temp_birth_penalty)
        )
        births = len(newborns)
        deaths += maternal_deaths
        self.population.extend(newborns)

        alive_now = [p for p in self.population if p.alive]
        self.contact_graph = self._build_contact_graph(alive_now)
        self._update_civilization_metrics(alive_now)
        transition_stories = self._update_world_structures(year, alive_now)
        stories.extend(transition_stories)
        food_for_living = self._food_ratio_by_region(alive_now)
        for p in alive_now:
            self._clamp_living_context(p)
        self._evolve_living_contexts(alive_now, food_for_living)
        self._update_ecology(alive_now)
        adjustments = self._auto_adjust_parameters(alive_now, available_food)
        event_deaths, event_stories = self._process_world_events(year, alive_now)
        deaths += event_deaths
        stories.extend(event_stories)
        self._advance_jail_sentences(alive_now)
        new_infections_after = self._count_any_infected()
        self.last_step_events = {
            "births": births,
            "deaths": deaths,
            "new_infections": max(0, new_infections_after - new_infections_before),
            "stories": stories,
            "adjustments": adjustments,
        }
        self._train_learned_goal_policy(year)
        self._decay_regional_disaster_years()
        return births, deaths, available_food

    def _pick_father_for_birth(self, mother: Individual, males: list[Individual]) -> Individual:
        """Prefer local, healthy, fertile partners with aligned skills and language (bounded rationality)."""
        same_region = [m for m in males if m.region_id == mother.region_id]
        local_bias = min(0.92, 0.38 + mother.knowledge * 0.32 + mother.tool_skill * 0.22)
        pool = same_region if same_region and self.rng.random() < local_bias else males

        def score(m: Individual) -> float:
            fert = m.genetic_traits.get("fertility", 0.5)
            sim = (
                1.0
                - abs(mother.knowledge - m.knowledge) * 0.3
                - abs(mother.tool_skill - m.tool_skill) * 0.2
            )
            lang = 0.12 if m.language == mother.language else 0.0
            reg = 0.1 if m.region_id == mother.region_id else 0.0
            pair = normalize_pair(mother.person_id, m.person_id)
            trust_bonus = (self._pair_trust.get(pair, 0.5) - 0.5) * 0.34
            return max(0.05, m.health * (0.42 + fert * 0.58) * max(0.22, sim) + lang + reg + trust_bonus)

        k = max(3, min(14, len(pool)))
        top = sorted(pool, key=score, reverse=True)[:k]
        w = [score(m) for m in top]
        return self.rng.choices(top, weights=w, k=1)[0]

    def _generate_births(
        self, alive: list[Individual], era_birth_multiplier: float = 1.0
    ) -> tuple[list[Individual], int]:
        cfg = self.config.demographics
        pathogen_names = [p.name for p in self.config.pathogens]
        alive_count = len(alive)
        crowding_penalty = 1.0 / (1.0 + max(0.0, (alive_count - 900.0) / 650.0))
        food_ratio_by_region = self._food_ratio_by_region(alive)
        infection_ratio_by_region = self._infection_ratio_by_region(alive)
        bcfg = self.config.behavior
        wrm = self.config.world_realism
        maternal_deaths = 0
        females = [
            p
            for p in alive
            if p.gender == Gender.FEMALE
            and cfg.reproductive_age_min <= p.age <= cfg.reproductive_age_max
            and p.health > 0.35
        ]
        males = [
            p
            for p in alive
            if p.gender == Gender.MALE
            and cfg.reproductive_age_min <= p.age <= cfg.reproductive_age_max
            and p.health > 0.35
        ]

        if not females or not males:
            return [], 0

        macro_birth = self._compute_macro_goal_cache(alive, food_ratio_by_region)
        civ_birth = (
            (sum(p.knowledge for p in alive) / alive_count) * 0.55
            + (sum(p.tool_skill for p in alive) / alive_count) * 0.45
            if alive_count
            else 0.2
        )

        newborns: list[Individual] = []
        for mother in females:
            if self.rng.random() > cfg.partner_match_rate:
                continue

            preferred = next((m for m in males if m.person_id == mother.love_partner_id), None)
            if preferred is not None and self.rng.random() < 0.74:
                father = preferred
            else:
                father = self._pick_father_for_birth(mother, males)
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
            fertility_score *= crowding_penalty
            if wrm.enabled and mother.is_married and mother.love_partner_id == father.person_id:
                fertility_score *= 1.0 + wrm.married_birth_fertility_bonus
            fertility_score = max(0.0, min(1.0, fertility_score))
            if self.rng.random() >= fertility_score:
                continue

            traits = inherit_traits(mother, father, self.rng)
            knowledge, tool_skill, spiritual, belief_group = inherit_social_profile(mother, father, self.rng)
            happiness, stress, aggression, ambition = inherit_emotions(mother, father, self.rng)
            cognitive_iq = inherit_cognitive_iq(
                mother,
                father,
                self.rng,
                self.config.cognition.birth_iq_diversity,
            )
            c_wealth = inherit_wealth(mother, father, self.rng)
            c_rep = inherit_reputation(mother, father, self.rng)
            riding_skill = inherit_riding(mother, father, self.rng)
            political_power = max(
                0.0,
                min(
                    1.0,
                    self.rng.gauss((mother.political_power + father.political_power) / 2.0, 0.06),
                ),
            )
            faction = mother.faction if self.rng.random() < 0.5 else father.faction
            language = mother.language if self.rng.random() < 0.65 else father.language
            if "digital_age" in self.unlocked_milestones and self.rng.random() < 0.08:
                language = self.rng.choice(self.language_names)
            ch = self.rng.uniform(0.7, 1.0)
            gctx = self._world_goal_context_from_demographics(
                region_id=mother.region_id,
                faction=faction,
                wealth=c_wealth,
                food_ratio_by_region=food_ratio_by_region,
                civ_w=civ_birth,
                macro=macro_birth,
            )
            child_living = (
                mother.living_context
                if self.rng.random() < 0.68
                else self._sample_living_context(mother.region_id)
            )
            child_nutrition = max(
                0.22,
                min(1.12, self.rng.gauss(mother.nutrition_ema, 0.055)),
            )
            child = Individual(
                person_id=self.next_person_id,
                age=0,
                gender=Gender.FEMALE if self.rng.random() < 0.5 else Gender.MALE,
                region_id=mother.region_id,
                living_context=child_living,
                nutrition_ema=child_nutrition,
                alive=True,
                health=ch,
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
                ambition=ambition,
                faction=faction,
                language=language,
                personal_tools=0,
                books_authored=0,
                riding_skill=riding_skill,
                inventions_made=0,
                mutation_burden=0.0,
                political_power=political_power,
                cognitive_iq=cognitive_iq,
                wealth=c_wealth,
                reputation=c_rep,
                primary_goal=brain_choose_primary_goal_fields(
                    self.config.cognition.world_iq,
                    cognitive_iq,
                    health=ch,
                    stress=stress,
                    ambition=ambition,
                    age=0,
                    happiness=happiness,
                    knowledge=knowledge,
                    tool_skill=tool_skill,
                    observed_food_ema=1.0,
                    observed_food_trend=0.0,
                    wealth=c_wealth,
                    reputation=c_rep,
                    ctx=gctx,
                    rng=self.rng,
                ),
                observed_food_ema=1.0,
                observed_food_trend=0.0,
            )
            self._clamp_living_context(child)
            self.next_person_id += 1

            if self.rng.random() < cfg.child_mortality:
                child.alive = False
            else:
                newborns.append(child)
                if wrm.enabled and wrm.maternal_mortality_enabled:
                    mm = wrm.maternal_mortality_base * (wrm.maternal_mortality_health_scale - mother.health * 0.52)
                    if self.current_era == "hunter-gatherer":
                        mm *= 1.32
                    if "country" in self.unlocked_milestones:
                        mm *= 0.74
                    mm = max(0.0, min(0.11, mm))
                    if self.rng.random() < mm:
                        mother.alive = False
                        maternal_deaths += 1
        return newborns, maternal_deaths

    def _sync_season(self, year: int) -> None:
        sc = self.config.seasons
        n = len(sc.names)
        if not sc.enabled or n == 0:
            self.current_season_index = 0
            self.current_season_name = "—"
            self._season_food_mult = 1.0
            self._season_migration_mult = 1.0
            self._season_disease_mult = 1.0
            self._season_wildlife_mult = 1.0
            return

        def _pick(t: tuple[float, ...], i: int) -> float:
            return float(t[i % len(t)])

        idx = (int(year) + int(sc.phase_offset)) % n
        self.current_season_index = idx
        self.current_season_name = str(sc.names[idx])
        self._season_food_mult = _pick(sc.food_multiplier, idx)
        self._season_migration_mult = _pick(sc.migration_multiplier, idx)
        self._season_disease_mult = _pick(sc.disease_transmission_multiplier, idx)
        self._season_wildlife_mult = _pick(sc.wildlife_food_multiplier, idx)

    def _ecology_food_factor(self, region_id: int) -> float:
        hunt = 0.015 * self.wildlife_index
        wm = max(0.05, min(2.5, self._season_wildlife_mult))
        if not self.agriculture_unlocked and "animal_husbandry" not in self.world_inventions:
            return 1.0 + min(0.06, hunt * 2.2 * wm)
        fields = sum(
            1
            for s in self.world_structures
            if s.get("kind") == "field" and int(s.get("region_id", -1)) == region_id
        )
        lv = float(self.livestock_by_region.get(region_id, 0.0))
        farm = min(0.15, fields * 0.024) if self.agriculture_unlocked else 0.0
        herd = min(0.13, lv * 0.0048)
        if "animal_husbandry" in self.world_inventions:
            herd *= 1.28
        wild = 0.006 * self.wildlife_index if self.agriculture_unlocked else 0.014 * self.wildlife_index
        base_core = 1.0 + farm + herd
        extra = wild + hunt * 0.35
        return base_core + extra * wm

    def total_livestock(self) -> float:
        return float(sum(self.livestock_by_region.values()))

    def _update_ecology(self, alive: list[Individual]) -> None:
        if not alive:
            return
        by_region: dict[int, list[Individual]] = defaultdict(list)
        for p in alive:
            by_region[p.region_id].append(p)
        field_n: dict[int, int] = defaultdict(int)
        for s in self.world_structures:
            if s.get("kind") == "field":
                field_n[int(s.get("region_id", 0))] += 1
        n_regions = self.config.demographics.region_count
        global_pop = len(alive)
        pressure = min(2.8, global_pop / max(120.0, 15.0 * n_regions))
        for rid in range(n_regions):
            pop = len(by_region.get(rid, []))
            fields = field_n.get(rid, 0)
            cur = float(self.livestock_by_region.get(rid, 0.0))
            target = 0.0
            if self.agriculture_unlocked or "animal_husbandry" in self.world_inventions:
                target += min(38.0, pop * 0.22 + fields * 3.4)
            if "animal_husbandry" in self.world_inventions:
                target *= 1.45
            if "wheel" in self.world_inventions:
                target *= 1.08
            target += self.rng.gauss(0, 0.55)
            target = max(0.0, target)
            self.livestock_by_region[rid] = max(0.0, min(55.0, cur * 0.88 + target * 0.12))
        stress_w = sum(p.stress for p in alive) / global_pop
        self.wildlife_index = max(
            0.18,
            min(
                1.85,
                self.wildlife_index * 0.985
                + 0.018 * (1.15 - pressure * 0.12)
                - 0.01 * stress_w
                - (0.008 * n_regions if self.agriculture_unlocked else 0.0)
                + self.rng.uniform(-0.012, 0.012),
            ),
        )

    def _available_food_total(self, alive: list[Individual]) -> float:
        by_region = defaultdict(int)
        for person in alive:
            by_region[person.region_id] += 1
        total = 0.0
        for region_id, count in by_region.items():
            regional_food = self.environments[region_id].available_food(count)
            regional_food *= max(0.75, 1.0 + self.region_food_adjustments.get(region_id, 0.0))
            regional_food *= self._ecology_food_factor(region_id)
            regional_food *= self._season_food_mult
            regional_food *= self._region_harvest_mult(region_id)
            regional_food *= self._region_disaster_food_mult(region_id)
            total += regional_food
        return total

    def _world_aggression(self) -> float:
        wa = float(getattr(self.config.conflict, "world_aggression", 1.0))
        return max(0.15, min(3.0, wa))

    def _food_ratio_by_region(self, alive: list[Individual]) -> dict[int, float]:
        by_region = defaultdict(int)
        for person in alive:
            by_region[person.region_id] += 1
        ratios: dict[int, float] = {}
        for region_id, count in by_region.items():
            available = self.environments[region_id].available_food(count)
            available *= max(0.75, 1.0 + self.region_food_adjustments.get(region_id, 0.0))
            available *= self._ecology_food_factor(region_id)
            available *= self._season_food_mult
            available *= self._region_harvest_mult(region_id)
            available *= self._region_disaster_food_mult(region_id)
            modifier = max(0.2, 1.0 + self.food_system_bonus - self.temp_food_penalty)
            ratios[region_id] = (available * modifier) / count if count else 0.0
        return ratios

    def _adjacent_region_pairs(self) -> list[tuple[int, int]]:
        n = self.config.demographics.region_count
        return [(i, i + 1) for i in range(max(0, n - 1))]

    def _region_war_power(self, people: list[Individual], food_ratio: float) -> float:
        if not people:
            return 0.0
        n = len(people)
        h = sum(p.health for p in people) / n
        a = sum(p.aggression for p in people) / n
        t = sum(p.tool_skill for p in people) / n
        r = sum(p.riding_skill for p in people) / n
        w = sum(min(2.5, p.wealth) for p in people) / n
        pt = sum(min(24, p.personal_tools) for p in people) / n
        mount = 0.06 * r if "animal_husbandry" in self.world_inventions else 0.0
        wheel = 0.04 * r if "wheel" in self.world_inventions else 0.0
        iron = 0.05 if "iron_working" in self.world_inventions else 0.0
        return (
            n
            * h
            * (0.82 + 0.18 * a)
            * (1.0 + 0.14 * t + 0.04 * pt + mount + wheel + iron)
            * (0.88 + 0.24 * food_ratio)
            * (1.0 + 0.06 * min(1.0, w))
        )

    def _apply_border_conquest(self, winner_rid: int, loser_rid: int, intensity: float) -> str:
        env_w = self.environments[winner_rid]
        env_l = self.environments[loser_rid]
        frac = 0.035 + 0.09 * max(0.0, min(1.0, intensity))
        t_take = env_l.territory_size * frac
        t_take = min(t_take, max(0.0, env_l.territory_size - 0.52))
        if t_take <= 0.008:
            return f"{self._region_name(winner_rid)} held the border; gains were marginal."
        env_l.territory_size = max(0.5, env_l.territory_size - t_take)
        env_w.territory_size = min(1.85, env_w.territory_size + t_take * 0.9)
        for k in env_l.resource_richness:
            slice_ = min(env_l.resource_richness[k] * (0.025 + 0.05 * intensity), env_l.resource_richness[k] * 0.14)
            env_l.resource_richness[k] = max(0.08, env_l.resource_richness[k] - slice_)
            env_w.resource_richness[k] = min(1.0, env_w.resource_richness[k] + slice_ * 0.88)
        return (
            f" {self._region_name(winner_rid)} annexed border territory and resources from "
            f"{self._region_name(loser_rid)}."
        )

    def _step_culture_inventions_and_riding(
        self,
        alive: list[Individual],
        year: int,
        civ_w: float,
        food_ratio_by_region: dict[int, float],
    ) -> list[str]:
        stories: list[str] = []
        if len(alive) < 25:
            return stories
        tcfg = self.config.technology
        drain_s = max(0.0, tcfg.resource_drain_scale)

        invention_chain: list[tuple[str, float, str, str]] = [
            (
                "animal_husbandry",
                0.2,
                "Domestication",
                "Beasts of burden and mounts spread; people learn to ride and haul.",
            ),
            ("wheel", 0.34, "The wheel", "Wheeled transport speeds trade and long-distance movement."),
            (
                "advanced_tools",
                0.46,
                "Advanced tools",
                "Better implements raise craft output and everyday productivity.",
            ),
            (
                "iron_working",
                0.58,
                "Iron working",
                "Harder metal improves tools and slightly reduces preventable deaths.",
            ),
        ]

        for key, threshold, title, blurb in invention_chain:
            if key in self.world_inventions:
                continue
            if civ_w < threshold:
                continue
            pool = [
                p
                for p in alive
                if p.age >= 17
                and p.age < 72
                and (p.knowledge * 0.48 + p.tool_skill * 0.52) > 0.24 + threshold * 0.2
            ]
            if tcfg.resource_gated_inventions:
                pool = [
                    p
                    for p in pool
                    if region_meets_invention_minimums(self.environments[p.region_id], key)
                ]
            if not pool:
                continue
            headroom = min(1.6, (civ_w - threshold) * 3.2 + 0.08)
            base_p = min(0.52, 0.055 + headroom * 0.22)
            inventor = self.rng.choice(pool)
            fr = food_ratio_by_region.get(inventor.region_id, 1.0)
            need_mul = invention_roll_multiplier(
                inventor.cognitive_iq,
                inventor.knowledge,
                inventor.tool_skill,
                inventor.stress,
                invention_key=key,
                regional_food_ratio=fr,
            )
            if self.rng.random() > min(0.55, base_p * need_mul):
                continue
            env_inv = self.environments[inventor.region_id]
            if tcfg.resource_gated_inventions and not region_meets_invention_minimums(env_inv, key):
                continue
            self.world_inventions.add(key)
            if tcfg.resource_gated_inventions:
                apply_invention_resource_drain(env_inv, key, scale=drain_s)
            inventor.inventions_made += 1
            inventor.knowledge = min(1.0, inventor.knowledge + 0.018 + self.rng.uniform(0, 0.012))
            inventor.tool_skill = min(1.0, inventor.tool_skill + 0.012)
            stories.append(self._register_timeline_transition(year, f"Invention: {title}", blurb))
            if key == "animal_husbandry":
                self.food_system_bonus = min(0.28, self.food_system_bonus + 0.014)
            elif key == "wheel":
                self.food_system_bonus = min(0.28, self.food_system_bonus + 0.008)
            elif key == "advanced_tools":
                self.food_system_bonus = min(0.28, self.food_system_bonus + 0.012)
            elif key == "iron_working":
                self.mortality_reduction_bonus = min(0.18, self.mortality_reduction_bonus + 0.009)

        if "animal_husbandry" in self.world_inventions:
            for p in alive:
                if not (8 <= p.age <= 62):
                    continue
                base = 0.028 + p.tool_skill * 0.038
                if "wheel" in self.world_inventions:
                    base += 0.014
                if self.rng.random() < base:
                    p.riding_skill = min(1.0, p.riding_skill + self.rng.uniform(0.012, 0.034))

        return stories

    def _simulate_regional_diplomacy(self, alive: list[Individual], year: int) -> tuple[list[str], int]:
        stories: list[str] = []
        deaths = 0
        if len(alive) < 20 or self.config.demographics.region_count < 2:
            return stories, deaths
        by_region: dict[int, list[Individual]] = defaultdict(list)
        for p in alive:
            by_region[p.region_id].append(p)

        food_ratio_by_region = self._food_ratio_by_region(alive)

        for ra, rb in self._adjacent_region_pairs():
            pa = by_region.get(ra, [])
            pb = by_region.get(rb, [])
            if not pa or not pb:
                continue
            pair = (ra, rb) if ra < rb else (rb, ra)
            has_trade = pair in self.region_trade_links

            avg_ambition = (
                (sum(p.ambition for p in pa) / len(pa)) + (sum(p.ambition for p in pb) / len(pb))
            ) * 0.5
            avg_aggr = ((sum(p.aggression for p in pa) / len(pa)) + (sum(p.aggression for p in pb) / len(pb))) * 0.5
            food_a = food_ratio_by_region.get(ra, 1.0)
            food_b = food_ratio_by_region.get(rb, 1.0)
            same_faction = self._dominant_label(pa, "faction") == self._dominant_label(pb, "faction")
            pw_a = self._region_war_power(pa, food_a)
            pw_b = self._region_war_power(pb, food_b)
            rs_a = self.environments[ra].resource_score()
            rs_b = self.environments[rb].resource_score()
            pr = max(pw_a, pw_b) / max(1e-6, min(pw_a, pw_b))

            event, war_intensity = self.world_dynamics.step_border(
                ra,
                rb,
                food_a,
                food_b,
                same_faction,
                avg_aggr,
                avg_ambition,
                has_trade,
                year,
                self._world_aggression(),
                self.config.cognition.world_iq,
                resource_score_a=rs_a,
                resource_score_b=rs_b,
                relative_power_ratio=pr,
                material_pressure_a=material_pressure(rs_a, food_a),
                material_pressure_b=material_pressure(rs_b, food_b),
            )

            if event == "trade_open":
                self.region_trade_links.add(pair)
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Trade pact: {self._region_name(ra)} ↔ {self._region_name(rb)}",
                        "Diplomacy opened trade routes and migration corridors.",
                    )
                )
            elif event == "war":
                casualties = self._regional_war_casualties(pa, pb, war_intensity)
                deaths += casualties
                if has_trade and self.world_dynamics.global_instability > 0.52:
                    self.region_trade_links.discard(pair)
                conquest_note = ""
                if pw_a > pw_b * 1.06:
                    conquest_note = self._apply_border_conquest(ra, rb, war_intensity)
                elif pw_b > pw_a * 1.06:
                    conquest_note = self._apply_border_conquest(rb, ra, war_intensity)
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Border war: {self._region_name(ra)} vs {self._region_name(rb)}",
                        f"Resource and territory conflict caused {casualties} deaths.{conquest_note}",
                    )
                )
            elif event == "trade_break":
                if pair in self.region_trade_links:
                    self.region_trade_links.discard(pair)
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Trade rupture: {self._region_name(ra)} — {self._region_name(rb)}",
                        "Diplomatic tension collapsed a trade agreement.",
                    )
                )

            if pair in self.region_trade_links:
                transfer = max(0.0, abs(food_a - food_b) * 0.06)
                if food_a > food_b:
                    self.region_food_adjustments[ra] = self.region_food_adjustments.get(ra, 0.0) - transfer * 0.45
                    self.region_food_adjustments[rb] = self.region_food_adjustments.get(rb, 0.0) + transfer
                else:
                    self.region_food_adjustments[rb] = self.region_food_adjustments.get(rb, 0.0) - transfer * 0.45
                    self.region_food_adjustments[ra] = self.region_food_adjustments.get(ra, 0.0) + transfer
        return stories, deaths

    def _regional_war_casualties(self, pa: list[Individual], pb: list[Individual], intensity: float) -> int:
        casualties = 0
        all_people = sorted(pa + pb, key=lambda p: p.person_id)
        frac = min(0.04, 0.003 + max(0.0, min(1.0, intensity)) * 0.032)
        kill_n = max(1, int(len(all_people) * frac))
        for person in all_people[:kill_n]:
            if person.alive:
                person.alive = False
                casualties += 1
        return casualties

    def _step_regional_food_trends(self, food_ratio_by_region: dict[int, float]) -> None:
        for region_id, fr in food_ratio_by_region.items():
            prev = self._region_food_prev.get(region_id, fr)
            tr = self._region_food_trend.get(region_id, 0.0)
            self._region_food_trend[region_id] = 0.76 * tr + 0.24 * (fr - prev)
            self._region_food_prev[region_id] = fr
        for region_id in range(self.config.demographics.region_count):
            if region_id not in food_ratio_by_region:
                self._region_food_prev.setdefault(region_id, 1.0)

    def _bootstrap_goal_ctx(self, mean_food: float = 1.0) -> WorldGoalContext:
        return WorldGoalContext(
            civ_index=max(0.06, min(1.0, self.civilization_index * 0.9 + 0.05)),
            global_instability=self.world_dynamics.global_instability,
            food_inequality=self.world_dynamics.food_inequality,
            mean_food_ratio=mean_food,
            settlement_tier=0,
            regional_wealth_poor=0.0,
            local_food_vs_world=0.0,
            resource_index_local=0.5,
            treasury_strength_local=0.5,
            policy_tax_burden=0.06,
            policy_security=0.32,
            policy_institutional_openness=0.48,
            region_trade_connected=0.0,
            faction_local_power=0.45,
            wealth_spread_local=0.22,
        )

    def _compute_macro_goal_cache(
        self,
        alive: list[Individual],
        food_ratio_by_region: dict[int, float],
    ) -> dict[str, object]:
        """Per-year aggregates: resources, treasuries, trade access, faction shares, local inequality."""
        rc = self.config.demographics.region_count
        by_region: dict[int, list[Individual]] = defaultdict(list)
        for p in alive:
            by_region[p.region_id].append(p)

        fr_list = list(food_ratio_by_region.values()) if food_ratio_by_region else [1.0]
        mean_f = sum(fr_list) / len(fr_list)

        r_raw = [self.environments[i].resource_score() for i in range(rc)]
        r_max = max(r_raw) if r_raw else 1.0
        if r_max < 1e-9:
            r_norm = {i: 0.5 for i in range(rc)}
        else:
            r_norm = {i: r_raw[i] / r_max for i in range(rc)}

        tpc = {
            rid: self.region_treasury.get(rid, 0.0) / max(1, len(by_region.get(rid, [])))
            for rid in range(rc)
        }
        t_vals = [tpc[i] for i in range(rc)]
        lo, hi = (min(t_vals), max(t_vals)) if t_vals else (0.0, 1.0)
        if hi - lo < 1e-9:
            t_norm = {i: 0.5 for i in range(rc)}
        else:
            t_norm = {i: (tpc[i] - lo) / (hi - lo) for i in range(rc)}

        trade_conn = {i: 0.0 for i in range(rc)}
        for ra, rb in self._adjacent_region_pairs():
            pair = (ra, rb) if ra < rb else (rb, ra)
            if pair in self.region_trade_links:
                trade_conn[ra] = 1.0
                trade_conn[rb] = 1.0

        faction_share: dict[int, dict[str, float]] = {i: {} for i in range(rc)}
        wealth_spread: dict[int, float] = {}
        med_by_r: dict[int, float] = {}
        for rid in range(rc):
            people = by_region.get(rid, [])
            n = len(people)
            if n == 0:
                med_by_r[rid] = 0.5
                wealth_spread[rid] = 0.2
                faction_share[rid] = {}
                continue
            ws = sorted(p.wealth for p in people)
            med_by_r[rid] = ws[n // 2]
            if n < 2:
                wealth_spread[rid] = 0.12
            else:
                mean_w = sum(ws) / n
                var = sum((x - mean_w) ** 2 for x in ws) / (n - 1)
                std = math.sqrt(max(0.0, var))
                wealth_spread[rid] = min(1.0, std / (mean_w + 0.12))
            fc: dict[str, int] = defaultdict(int)
            for p in people:
                fc[p.faction] += 1
            faction_share[rid] = {f: c / n for f, c in fc.items()}

        return {
            "mean_f": mean_f,
            "med_by_r": med_by_r,
            "r_norm": r_norm,
            "t_norm": t_norm,
            "trade_conn": trade_conn,
            "faction_share": faction_share,
            "wealth_spread": wealth_spread,
        }

    def _world_goal_context_from_demographics(
        self,
        *,
        region_id: int,
        faction: str,
        wealth: float,
        food_ratio_by_region: dict[int, float],
        civ_w: float,
        macro: dict[str, object],
    ) -> WorldGoalContext:
        mean_f = float(macro["mean_f"])
        med_by_r: dict[int, float] = macro["med_by_r"]  # type: ignore[assignment]
        med = max(0.08, med_by_r.get(region_id, 0.5))
        rel_poor = max(0.0, min(1.0, (med - wealth) / (med + 0.12)))
        lf = food_ratio_by_region.get(region_id, mean_f) - mean_f
        lf = max(-0.55, min(0.55, lf)) / 0.55
        pol = self.region_policies.get(region_id, {})
        fshare: dict[str, float] = macro["faction_share"][region_id]  # type: ignore[index]
        fp = fshare.get(faction, 1.0 / max(1, len(fshare)))
        return WorldGoalContext(
            civ_index=civ_w,
            global_instability=self.world_dynamics.global_instability,
            food_inequality=self.world_dynamics.food_inequality,
            mean_food_ratio=mean_f,
            settlement_tier=self._settlement_tier(region_id),
            regional_wealth_poor=rel_poor,
            local_food_vs_world=lf,
            resource_index_local=float(macro["r_norm"][region_id]),  # type: ignore[index]
            treasury_strength_local=float(macro["t_norm"][region_id]),  # type: ignore[index]
            policy_tax_burden=float(pol.get("income_tax", 0.06)),
            policy_security=float(pol.get("theft_enforcement", 0.35)),
            policy_institutional_openness=float(pol.get("public_gossip", 0.5)),
            region_trade_connected=float(macro["trade_conn"][region_id]),  # type: ignore[index]
            faction_local_power=min(1.0, max(0.0, fp)),
            wealth_spread_local=float(macro["wealth_spread"][region_id]),  # type: ignore[index]
        )

    def _world_goal_context_for_person(
        self,
        person: Individual,
        food_ratio_by_region: dict[int, float],
        civ_w: float,
        macro: dict[str, object],
    ) -> WorldGoalContext:
        return self._world_goal_context_from_demographics(
            region_id=person.region_id,
            faction=person.faction,
            wealth=person.wealth,
            food_ratio_by_region=food_ratio_by_region,
            civ_w=civ_w,
            macro=macro,
        )

    def _migration_institution_bonus(self, region_id: int, goal: str, macro: dict[str, object]) -> float:
        """Extra migration pull from treasuries, taxes, trade routes, openness, resources (goal-weighted)."""
        pol = self.region_policies.get(region_id, {})
        tax = float(pol.get("income_tax", 0.06))
        opn = float(pol.get("public_gossip", 0.5))
        t_stab = float(macro["t_norm"][region_id])  # type: ignore[index]
        trc = float(macro["trade_conn"][region_id])  # type: ignore[index]
        r_n = float(macro["r_norm"][region_id])  # type: ignore[index]
        tax_ease = 1.0 - min(1.0, tax / 0.2)
        w_trade = 1.0 if goal == "trade" else (0.55 if goal == "connect" else 0.35)
        w_survive = 1.0 if goal == "survive" else (0.62 if goal == "prosper" else 0.42)
        b = 0.11 * t_stab * w_survive
        b += 0.07 * tax_ease * (1.15 if goal == "accumulate" else 0.82)
        b += 0.12 * trc * w_trade
        b += 0.055 * opn * (1.0 if goal in ("connect", "trade") else 0.68)
        b += 0.052 * r_n * (1.0 if goal in ("accumulate", "status") else 0.78)
        return b

    def _sample_birth_cognitive_iq(self) -> float:
        d = max(0.0, min(1.0, float(self.config.cognition.birth_iq_diversity)))
        mid = 0.58
        half_w = 0.1 + 0.34 * d
        x = self.rng.uniform(mid - half_w, mid + half_w)
        x += self.rng.gauss(0.0, 0.018 + 0.11 * d)
        return max(0.08, min(0.98, x))

    def _settlement_tier(self, region_id: int) -> int:
        st = self._get_settlement_structure(region_id)
        if st is None:
            return 0
        lvl = str(st.get("level", "camp"))
        base = {"camp": 0, "village": 1, "town": 2, "city": 3}.get(lvl, 0)
        if lvl != "city":
            return base
        polity = str(st.get("polity", "city"))
        if polity == "empire":
            return 5
        if polity == "country":
            return 4
        return 3

    def _refresh_region_policies(self) -> None:
        for rid in range(self.config.demographics.region_count):
            tier = self._settlement_tier(rid)
            pol = default_region_policy(tier)
            gov = str(self.politics_by_region.get(rid, {}).get("government", "informal"))
            if gov in ("democracy", "republic"):
                pol["theft_enforcement"] = max(0.12, pol["theft_enforcement"] - 0.09)
                pol["public_gossip"] = min(0.98, pol["public_gossip"] + 0.08)
                pol["income_tax"] = min(0.15, pol["income_tax"] + 0.01)
            elif gov in ("autocracy", "monarchy"):
                pol["theft_enforcement"] = min(0.94, pol["theft_enforcement"] + 0.12)
                pol["public_gossip"] = max(0.22, pol["public_gossip"] - 0.06)
                pol["income_tax"] = min(0.18, pol["income_tax"] + 0.025)
            elif gov == "chiefdom":
                pol["theft_enforcement"] = min(0.88, pol["theft_enforcement"] + 0.04)
            self.region_policies[rid] = pol

    def _city_public_communication(
        self,
        alive: list[Individual],
        year: int,
        id_map: dict[int, Individual],
    ) -> list[str]:
        out: list[str] = []
        by_region: dict[int, list[Individual]] = defaultdict(list)
        for p in alive:
            by_region[p.region_id].append(p)
        any_broadcast = False
        for rid, residents in by_region.items():
            if self._settlement_tier(rid) < 2:
                continue
            st = self._get_settlement_structure(rid)
            if st is None:
                continue
            pol_state = self.politics_by_region.get(rid, {})
            lid = pol_state.get("leader_id")
            gossip = float(self.region_policies.get(rid, {}).get("public_gossip", 0.5))
            if lid is not None and int(lid) in id_map:
                leader = id_map[int(lid)]
                spread = 0.0035 * (0.55 + leader.knowledge) * (0.45 + gossip)
                for p in residents:
                    if p.person_id == leader.person_id:
                        continue
                    p.knowledge = min(1.0, p.knowledge + spread * self.rng.uniform(0.85, 1.15))
                    if self.rng.random() < 0.05 * gossip:
                        p.reputation = min(1.0, p.reputation + 0.0025)
                topic = self.rng.choice(
                    (
                        "market fair measures",
                        "shared well repair",
                        "guest rights",
                        "granary rules after scarcity",
                    )
                )
                st["last_edict"] = f"Y{year + 1}: leader #{leader.person_id} — {topic}."
            else:
                st["last_edict"] = f"Y{year + 1}: elders and guilds exchanged news on roads and storage."
            any_broadcast = True
            if gossip > 0.55 and len(residents) > 4 and self.rng.random() < 0.12:
                a = self.rng.choice(residents)
                b = self.rng.choice(residents)
                if a.person_id != b.person_id:
                    pr = normalize_pair(a.person_id, b.person_id)
                    self._pair_trust[pr] = min(1.0, self._pair_trust.get(pr, 0.5) + 0.012 * gossip)
        if any_broadcast and self.rng.random() < 0.18:
            out.append(f"Y{year + 1}: towns posted public rules and news between districts.")
        return out

    def _step_economy(self, alive: list[Individual], year: int, civ_w: float) -> list[str]:
        ecfg = self.config.economy
        out: list[str] = []
        if not ecfg.enabled or not alive:
            return out
        self._refresh_region_policies()
        food_rr = self._food_ratio_by_region(alive)
        slc = self.config.social_life
        jw = slc.jail_wealth_multiplier if slc.enabled else 1.0
        wrx = self.config.world_realism
        rent_drag = wrx.dense_housing_rent_drag if wrx.enabled else 0.0
        for p in alive:
            fr = food_rr.get(p.region_id, 1.0)
            prod = self._individual_productivity(p)
            wm = jw if slc.enabled and p.jail_years_remaining > 0 else 1.0
            p.wealth += 0.017 * prod * (0.55 + 0.45 * fr) * wm
            if rent_drag > 0 and p.living_context in ("town", "city") and fr < 0.92:
                scarcity = max(0.0, (0.92 - fr) / 0.92)
                p.wealth *= max(0.92, 1.0 - rent_drag * scarcity * (1.15 + p.stress * 0.35))
            p.wealth = min(40.0, max(0.0, p.wealth))
        id_map = {p.person_id: p for p in alive}
        for person in alive:
            if person.jail_years_remaining > 0:
                continue
            nbs = [id_map[n] for n in self.contact_graph.get(person.person_id, []) if n in id_map]
            if not nbs:
                continue
            for _ in range(ecfg.pairwise_trade_attempts):
                pairwise_market_trade(
                    self.rng,
                    person,
                    self.rng.choice(nbs),
                    self._pair_trust,
                    ecfg.trade_goal_bias,
                )
        theft_rounds = max(1, min(len(alive) // 60, 400)) * ecfg.theft_attempts
        theft_seen = 0
        for _ in range(theft_rounds):
            thief = self.rng.choice(alive)
            if thief.jail_years_remaining > 0:
                continue
            nbs = [id_map[n] for n in self.contact_graph.get(thief.person_id, []) if n in id_map]
            if not nbs:
                continue
            victim = self.rng.choice(nbs)
            if victim.person_id == thief.person_id:
                continue
            if victim.jail_years_remaining > 0:
                continue
            enf = float(self.region_policies.get(thief.region_id, {}).get("theft_enforcement", 0.3))
            r = theft_attempt(
                self.rng,
                thief,
                victim,
                enf,
                self._pair_trust,
                self.friendships,
                self.enmities,
            )
            if r == "theft_caught":
                theft_seen += 1
                thief.crimes_caught_count += 1
                sl = self.config.social_life
                if (
                    sl.enabled
                    and thief.crimes_caught_count >= sl.theft_jail_after_catches
                    and enf >= sl.theft_jail_min_enforcement
                ):
                    thief.jail_years_remaining = max(thief.jail_years_remaining, sl.jail_years_theft)
                self.region_treasury[thief.region_id] = self.region_treasury.get(thief.region_id, 0.0) + min(
                    0.22,
                    max(0.0, thief.wealth) * 0.06,
                )
            elif r == "theft_success":
                theft_seen += 1
        if theft_seen and self.rng.random() < 0.22:
            out.append(f"Y{year + 1}: {theft_seen} theft incidents (fines or quiet losses).")
        medians = region_wealth_medians(alive)
        apply_income_taxes(alive, medians, self.region_policies, self.region_treasury)
        rc = self.config.demographics.region_count
        rscores = [self.environments[i].resource_score() for i in range(rc)]
        inter_region_trade(
            self.rng,
            self.region_treasury,
            rc,
            rscores,
            ecfg.inter_region_trade_volume,
            civ_w,
        )
        if wrx.enabled and wrx.treasury_corruption_enabled:
            for rid in range(rc):
                enf = float(self.region_policies.get(rid, {}).get("theft_enforcement", 0.5))
                if enf >= wrx.treasury_corruption_enforcement_lt:
                    continue
                bal = self.region_treasury.get(rid, 0.0)
                if bal <= 0.08:
                    continue
                gap = max(0.0, wrx.treasury_corruption_enforcement_lt - enf + 0.1)
                skim = bal * self.rng.uniform(0.0028, wrx.treasury_corruption_max_fraction) * gap
                self.region_treasury[rid] = max(0.0, bal - skim)
        out.extend(self._city_public_communication(alive, year, id_map))
        return out

    def _assign_replanned_goal(
        self,
        p: Individual,
        ctx: WorldGoalContext,
        wiq: float,
        year: int,
    ) -> None:
        cc = self.config.cognition
        if self.goal_policy is None:
            p.primary_goal = brain_choose_primary_goal(p, wiq, self.rng, ctx)
            return
        h_vec = np.array(heuristic_goal_logits(p, ctx), dtype=np.float64)
        x = build_goal_feature_vector(p, ctx)
        l_vec = self.goal_policy.forward_logits(x)
        im = max(1, cc.learned_goal_imitation_years)
        mix = cc.learned_goal_mix * min(1.0, (year + 1) / float(im))
        blended = blend_goal_logits(h_vec, l_vec, mix)
        eff = effective_cognition_iq(wiq, p.cognitive_iq)
        temp = max(0.07, 0.64 * math.exp(2.35 * (1.0 - eff)))
        idx = _softmax_sample_index(blended.tolist(), temp, self.rng)
        p.primary_goal = GOAL_VALUES[idx]
        self._goal_policy_batch.append(
            {
                "pid": p.person_id,
                "x": x.copy(),
                "action": idx,
                "h_logits": h_vec.copy(),
                "h0": p.health,
                "hp0": p.happiness,
                "w0": p.wealth,
            }
        )

    def _train_learned_goal_policy(self, year: int) -> None:
        if self.goal_policy is None:
            self._goal_policy_batch.clear()
            return
        if not self._goal_policy_batch:
            return
        cc = self.config.cognition
        n_im = cc.learned_goal_imitation_years
        id_map = {p.person_id: p for p in self.population}
        for rec in self._goal_policy_batch:
            x = rec["x"]  # type: ignore[assignment]
            h_tgt = rec["h_logits"]  # type: ignore[assignment]
            assert isinstance(x, np.ndarray) and isinstance(h_tgt, np.ndarray)
            if year < n_im:
                self.goal_policy.backward_imitation(x, h_tgt, cc.learned_goal_lr_imitation)
            else:
                pid = int(rec["pid"])  # type: ignore[arg-type]
                p = id_map.get(pid)
                if p is None or not p.alive:
                    r = -1.25
                else:
                    r = (
                        1.75 * (p.health - float(rec["h0"]))
                        + 0.95 * (p.happiness - float(rec["hp0"]))
                        + 0.11 * (p.wealth - float(rec["w0"]))
                    )
                r = max(-2.0, min(2.0, r))
                self.goal_policy.backward_reinforce(x, int(rec["action"]), r, cc.learned_goal_lr)  # type: ignore[arg-type]
        self._goal_policy_batch.clear()

    def _replan_agent_goals(
        self,
        alive: list[Individual],
        civ_w: float,
        food_ratio_by_region: dict[int, float],
        year: int,
    ) -> None:
        wiq = self.config.cognition.world_iq
        macro = self._compute_macro_goal_cache(alive, food_ratio_by_region)
        for p in alive:
            ctx = self._world_goal_context_for_person(p, food_ratio_by_region, civ_w, macro)
            self._assign_replanned_goal(p, ctx, wiq, year)

    def _update_personal_food_memory(self, alive: list[Individual]) -> None:
        food_ratio_by_region = self._food_ratio_by_region(alive)
        for p in alive:
            fr = food_ratio_by_region.get(p.region_id, 1.0)
            delta = fr - p.observed_food_ema
            p.observed_food_ema = 0.82 * p.observed_food_ema + 0.18 * fr
            p.observed_food_trend = 0.76 * p.observed_food_trend + 0.24 * delta
        self._step_nutrition_tracking(alive, food_ratio_by_region)

    def _region_harvest_mult(self, region_id: int) -> float:
        wr = self.config.world_realism
        if not wr.enabled or not wr.harvest_weather_enabled:
            return 1.0
        return float(self.region_harvest_factor.get(region_id, 1.0))

    def _region_disaster_food_mult(self, region_id: int) -> float:
        wr = self.config.world_realism
        if not wr.enabled or not wr.regional_disaster_enabled:
            return 1.0
        if self.region_disaster_years_left.get(region_id, 0) <= 0:
            return 1.0
        return float(self.region_disaster_food_mult.get(region_id, 1.0))

    def _roll_new_regional_disasters(self, year: int) -> list[str]:
        wr = self.config.world_realism
        out: list[str] = []
        if not wr.enabled or not wr.regional_disaster_enabled:
            return out
        n_reg = self.config.demographics.region_count
        lo = max(0.35, min(0.95, wr.regional_disaster_food_mult_low))
        hi = max(lo + 0.02, min(0.98, wr.regional_disaster_food_mult_high))
        dmin = max(1, wr.regional_disaster_min_years)
        dmax = max(dmin, wr.regional_disaster_max_years)
        for rid in range(n_reg):
            if self.region_disaster_years_left.get(rid, 0) > 0:
                continue
            if self.rng.random() > wr.regional_disaster_probability:
                continue
            dur = self.rng.randint(dmin, dmax)
            mult = self.rng.uniform(lo, hi)
            self.region_disaster_years_left[rid] = dur
            self.region_disaster_food_mult[rid] = mult
            kind = self.rng.choice(
                ("drought", "flood", "crop blight", "hailstorm", "livestock loss", "river flood")
            )
            out.append(
                self._register_timeline_transition(
                    year,
                    f"{self._region_name(rid)}: {kind}",
                    f"Food production reduced for about {dur} year(s).",
                )
            )
        return out

    def _decay_regional_disaster_years(self) -> None:
        wr = self.config.world_realism
        if not wr.enabled or not wr.regional_disaster_enabled:
            return
        for rid in list(self.region_disaster_years_left.keys()):
            yl = int(self.region_disaster_years_left[rid]) - 1
            if yl <= 0:
                self.region_disaster_years_left.pop(rid, None)
                self.region_disaster_food_mult.pop(rid, None)
            else:
                self.region_disaster_years_left[rid] = yl

    def _apply_contact_bereavement(self, cohort: list[Individual]) -> None:
        wr = self.config.world_realism
        if not wr.enabled or not wr.bereavement_enabled:
            return
        if wr.bereavement_stress <= 0:
            return
        alive_map = {p.person_id: p for p in cohort if p.alive}
        cap = max(1, min(24, wr.bereavement_max_contacts))
        for dead in cohort:
            if dead.alive:
                continue
            for nid in self.contact_graph.get(dead.person_id, [])[:cap]:
                n = alive_map.get(nid)
                if n is None:
                    continue
                n.stress = min(1.0, n.stress + wr.bereavement_stress)
                n.happiness = max(0.0, n.happiness - wr.bereavement_happiness_hit)

    def _step_mutual_aid_scarcity(
        self, alive: list[Individual], food_ratio_by_region: dict[int, float]
    ) -> None:
        wr = self.config.world_realism
        if not wr.enabled or not wr.mutual_aid_enabled or not alive:
            return
        thr = max(0.5, min(0.95, wr.mutual_aid_food_below))
        p_try = max(0.0, min(0.55, wr.mutual_aid_attempt_probability))
        idm = {p.person_id: p for p in alive}
        for p in alive:
            if p.jail_years_remaining > 0:
                continue
            fr = food_ratio_by_region.get(p.region_id, 1.0)
            if fr >= thr:
                continue
            if self.rng.random() > p_try:
                continue
            for oid in self.contact_graph.get(p.person_id, []):
                o = idm.get(oid)
                if o is None or o.region_id != p.region_id or o.jail_years_remaining > 0:
                    continue
                pr = normalize_pair(p.person_id, o.person_id)
                tr = self._pair_trust.get(pr, 0.5)
                if tr < wr.mutual_aid_min_trust:
                    continue
                p.happiness = min(1.0, p.happiness + wr.mutual_aid_happiness)
                o.happiness = min(1.0, o.happiness + wr.mutual_aid_happiness * 0.85)
                self._pair_trust[pr] = min(1.0, tr + wr.mutual_aid_trust)
                break

    def _tick_regional_harvest_weather(self) -> None:
        wr = self.config.world_realism
        if not wr.enabled or not wr.harvest_weather_enabled:
            return
        for rid in range(self.config.demographics.region_count):
            shock = 1.0 + self.rng.gauss(0.0, wr.harvest_volatility)
            shock = max(wr.harvest_min_factor, min(wr.harvest_max_factor, shock))
            prev = float(self.region_harvest_factor.get(rid, 1.0))
            blended = wr.harvest_persistence * prev + (1.0 - wr.harvest_persistence) * shock
            self.region_harvest_factor[rid] = max(
                wr.harvest_min_factor, min(wr.harvest_max_factor, blended)
            )

    def _urban_crowding_disease_mult(self, alive: list[Individual]) -> float:
        wr = self.config.world_realism
        if not wr.enabled or wr.urban_crowding_disease_boost <= 0 or not alive:
            return 1.0
        dense = sum(1 for p in alive if p.living_context in ("town", "city"))
        frac = dense / len(alive)
        return 1.0 + wr.urban_crowding_disease_boost * frac

    def _step_nutrition_tracking(
        self, alive: list[Individual], food_ratio_by_region: dict[int, float]
    ) -> None:
        wr = self.config.world_realism
        if not wr.enabled or not wr.nutrition_tracking_enabled:
            return
        a = max(0.04, min(0.45, wr.nutrition_ema_alpha))
        thr = max(0.35, min(0.75, wr.nutrition_stress_threshold))
        for p in alive:
            fr = food_ratio_by_region.get(p.region_id, 1.0)
            p.nutrition_ema = (1.0 - a) * p.nutrition_ema + a * fr
            p.nutrition_ema = max(0.12, min(1.35, p.nutrition_ema))
            if p.nutrition_ema < thr:
                p.stress = min(1.0, p.stress + wr.nutrition_chronic_stress_per_year)
                p.health = max(0.0, p.health - wr.nutrition_chronic_health_penalty * (thr - p.nutrition_ema) / thr)
            elif p.nutrition_ema > 1.02:
                p.health = min(1.0, p.health + wr.nutrition_recovery_health_bonus * (p.nutrition_ema - 1.0))

    def _apply_migration(self, alive: list[Individual]) -> None:
        mcfg = self.config.migration
        if not mcfg.enabled or self.config.demographics.region_count <= 1:
            return
        infection_by_region = self._infection_ratio_by_region(alive)
        food_ratio_by_region = self._food_ratio_by_region(alive)
        bcfg = self.config.behavior
        macro_mig = (
            self._compute_macro_goal_cache(alive, food_ratio_by_region) if bcfg.enabled else {}
        )
        for person in alive:
            if person.jail_years_remaining > 0:
                continue
            migrate_prob = mcfg.migration_rate
            target_region = person.region_id

            if bcfg.enabled:
                gf, gi, gr = goal_migration_multipliers(person.primary_goal)
                iq = 0.5 * person.knowledge + 0.5 * person.tool_skill
                trend_scale = 0.28 + 0.55 * iq
                region_scores: list[float] = []
                for region_id in range(self.config.demographics.region_count):
                    food_score = food_ratio_by_region.get(region_id, 1.0)
                    reg_trend = self._region_food_trend.get(region_id, 0.0)
                    food_effective = max(0.12, min(1.45, food_score + trend_scale * reg_trend))
                    inf_score = 1.0 - infection_by_region.get(region_id, 0.0)
                    resource_score = self.environments[region_id].resource_score()
                    score = (
                        food_effective * bcfg.migration_food_weight * gf
                        + inf_score * bcfg.migration_infection_weight * gi
                        + resource_score * (0.18 + person.ambition * 0.16) * gr
                    )
                    score += self._migration_institution_bonus(
                        region_id, person.primary_goal, macro_mig
                    )
                    region_scores.append(score)
                best_score = max(region_scores)
                target_region = brain_choose_migration_region(
                    region_scores,
                    self.config.cognition.world_iq,
                    person.cognitive_iq,
                    self.rng,
                )

                cur_food = food_ratio_by_region.get(person.region_id, 1.0)
                cur_trend = self._region_food_trend.get(person.region_id, 0.0)
                cur_food_eff = max(0.12, min(1.45, cur_food + trend_scale * cur_trend))
                current_score = (
                    cur_food_eff * bcfg.migration_food_weight * gf
                    + (1.0 - infection_by_region.get(person.region_id, 0.0)) * bcfg.migration_infection_weight * gi
                    + self.environments[person.region_id].resource_score() * (0.18 + person.ambition * 0.16) * gr
                )
                current_score += self._migration_institution_bonus(
                    person.region_id, person.primary_goal, macro_mig
                )
                advantage = best_score - current_score
                ride_mobility = 1.0
                if "animal_husbandry" in self.world_inventions:
                    ride_mobility += person.riding_skill * (0.24 + (0.12 if "wheel" in self.world_inventions else 0.0))
                min_edge = max(
                    0.006,
                    0.11 * (1.02 - iq) / (1.0 + person.ambition * 0.35) / max(0.75, ride_mobility),
                )
                clear_gain = target_region != person.region_id and advantage > min_edge and best_score > current_score
                if clear_gain:
                    migrate_prob = min(1.0, migrate_prob * (1.45 + person.ambition * 0.9))
                elif target_region != person.region_id and best_score > current_score:
                    migrate_prob *= 0.32 + 0.55 * iq
                if person.observed_food_trend < -0.035:
                    migrate_prob *= 1.0 + min(
                        0.32, (-person.observed_food_trend) * 1.85 * (0.42 + person.ambition * 0.28)
                    )
                if "animal_husbandry" in self.world_inventions and bcfg.enabled:
                    migrate_prob = min(1.0, migrate_prob * (1.0 + 0.2 * person.riding_skill))

            sm = max(0.05, min(2.2, self._season_migration_mult))
            migrate_prob = min(1.0, migrate_prob * sm)

            if self.rng.random() < migrate_prob:
                prev_r = person.region_id
                if bcfg.enabled and target_region != person.region_id:
                    person.region_id = target_region
                else:
                    options = [r for r in range(self.config.demographics.region_count) if r != person.region_id]
                    person.region_id = self.rng.choice(options)
                if person.region_id != prev_r:
                    person.living_context = self._sample_living_context(person.region_id)
                    self._clamp_living_context(person)

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
        base_avg = max(1, self.config.contact_network.avg_contacts)
        pop = len(alive)
        # Adaptive contact throttling keeps very large populations tractable.
        if pop > 3000:
            avg = max(2, int(base_avg * 0.2))
        elif pop > 1600:
            avg = max(2, int(base_avg * 0.35))
        elif pop > 900:
            avg = max(3, int(base_avg * 0.55))
        else:
            avg = base_avg
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

    def _era_profile(self, year: int, civ_w: float) -> dict[str, float | str]:
        return compute_era_profile(
            year,
            civ_w,
            self.world_inventions,
            self.unlocked_milestones,
            self.agriculture_unlocked,
            dynamic=self.config.technology.dynamic_eras,
        )

    def _mentor_suitability(self, pupil: Individual, mentor: Individual) -> float:
        gain_k = max(0.0, mentor.knowledge - pupil.knowledge)
        gain_t = max(0.0, mentor.tool_skill - pupil.tool_skill)
        lang = 0.11 if mentor.language == pupil.language else 0.0
        if any(state == DiseaseState.INFECTED for state in mentor.disease_states.values()):
            return -0.2
        pair = normalize_pair(pupil.person_id, mentor.person_id)
        trust_adj = (self._pair_trust.get(pair, 0.5) - 0.5) * 0.16
        return gain_k * 1.2 + gain_t + mentor.knowledge * 0.07 + mentor.tool_skill * 0.05 + lang + trust_adj

    def _apply_social_learning(self, alive: list[Individual]) -> None:
        if not alive:
            return
        id_map = {p.person_id: p for p in alive}
        for person in alive:
            if person.jail_years_remaining > 0:
                continue
            neighbors = [id_map[n] for n in self.contact_graph.get(person.person_id, []) if n in id_map]
            if not neighbors:
                continue
            best_mentor = max(neighbors, key=lambda n: self._mentor_suitability(person, n))
            if self._mentor_suitability(person, best_mentor) < 0.02:
                teacher = max(neighbors, key=lambda n: n.knowledge + n.tool_skill)
            else:
                teacher = best_mentor
            head_k = max(0.07, 1.0 - person.knowledge)
            head_t = max(0.07, 1.0 - person.tool_skill)
            iq = 0.5 * person.knowledge + 0.5 * person.tool_skill
            wiq = max(0.12, min(1.0, float(self.config.cognition.world_iq)))
            learn_scale = 0.48 + 0.58 * wiq
            rate_k = learn_scale * 0.012 * (1.0 + 0.5 * iq) * min(1.2, head_k * 1.75)
            rate_t = learn_scale * 0.01 * (1.0 + 0.45 * iq) * min(1.2, head_t * 1.75)
            person.knowledge = min(1.0, person.knowledge + rate_k * teacher.knowledge + self.knowledge_boost)
            person.tool_skill = min(1.0, person.tool_skill + rate_t * teacher.tool_skill)
            wrlearn = self.config.world_realism
            if (
                wrlearn.enabled
                and self._has_structure_in_region("school", person.region_id)
                and person.age <= wrlearn.school_max_age
            ):
                person.knowledge = min(
                    1.0,
                    person.knowledge
                    + wrlearn.school_learning_bonus * (0.52 + 0.48 * person.cognitive_iq),
                )

            # Belief: spiritual learners adopt mentors; skeptics drift toward well-informed teachers; rare schisms.
            if teacher.belief_group != person.belief_group:
                conv = (
                    0.034 * person.spiritual_tendency
                    + 0.014
                    * (1.0 - person.spiritual_tendency * 0.38)
                    * (0.42 + teacher.knowledge * 0.58)
                )
                if self.rng.random() < conv:
                    person.belief_group = teacher.belief_group
            if self.rng.random() < 0.0022 and person.spiritual_tendency > 0.72:
                person.belief_group = f"cult_{person.person_id}"

    def _step_belief_evolution(self, alive: list[Individual]) -> None:
        if len(alive) < 10:
            return
        by_region: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for p in alive:
            if p.belief_group.startswith("cult_"):
                continue
            if is_prophet_movement_belief(p.belief_group):
                continue
            by_region[p.region_id][p.belief_group] += 1
        majority: dict[int, str | None] = {}
        for rid, counts in by_region.items():
            majority[rid] = max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
        for p in alive:
            if p.belief_group.startswith("cult_"):
                continue
            if is_prophet_movement_belief(p.belief_group):
                continue
            dom = majority.get(p.region_id)
            if not dom or dom == p.belief_group:
                continue
            if p.stress > 0.69 and self.rng.random() < 0.017 + (p.stress - 0.69) * 0.065:
                p.belief_group = dom
                continue
            if p.knowledge > 0.5 and self.rng.random() < 0.0088 * p.knowledge * (1.0 - p.spiritual_tendency * 0.42):
                p.belief_group = dom
                continue
            if p.happiness > 0.6 and self.rng.random() < 0.0048:
                p.belief_group = dom

    def _advance_jail_sentences(self, alive: list[Individual]) -> None:
        for p in alive:
            if p.jail_years_remaining > 0:
                p.jail_years_remaining -= 1

    def _sanitation_transmission_scale(self, alive: list[Individual]) -> float:
        wr = self.config.world_realism
        if not wr.enabled:
            return 1.0
        if not alive:
            return 1.0
        by_r: dict[int, int] = defaultdict(int)
        for p in alive:
            by_r[p.region_id] += 1
        tw = sum(self._settlement_tier(rid) * by_r[rid] for rid in by_r)
        mt = tw / len(alive)
        return sanitation_transmission_multiplier(
            mt,
            has_writing_milestone=("writing" in self.unlocked_milestones),
            min_mean_tier=wr.sanitation_min_mean_settlement_tier,
            max_reduction=wr.sanitation_transmission_reduction,
        )

    def _step_civic_unrest(
        self,
        alive: list[Individual],
        year: int,
        food_ratio_by_region: dict[int, float],
    ) -> list[str]:
        wr = self.config.world_realism
        if not wr.enabled or len(alive) < 35:
            return []
        by_r: dict[int, list[Individual]] = defaultdict(list)
        for p in alive:
            by_r[p.region_id].append(p)
        stories: list[str] = []
        hit = 0
        for rid, people in by_r.items():
            fr = food_ratio_by_region.get(rid, 1.0)
            ms = sum(p.stress for p in people) / len(people)
            if fr >= wr.unrest_food_ratio_below or ms <= wr.unrest_region_mean_stress_above:
                continue
            if self.rng.random() > wr.unrest_event_probability:
                continue
            hit += 1
            self.world_dynamics.global_instability = min(
                1.0,
                self.world_dynamics.global_instability + wr.unrest_instability_bump,
            )
            for p in people:
                if self.rng.random() < 0.17:
                    p.stress = min(1.0, p.stress + 0.042)
                    p.happiness = max(0.0, p.happiness - 0.028)
        if hit:
            stories.append(
                self._register_timeline_transition(
                    year,
                    "Civic unrest",
                    f"Hunger and strain sparked protests or riots in {hit} region(s).",
                )
            )
        return stories

    def _step_social_life(self, alive: list[Individual], year: int) -> list[str]:
        sl = self.config.social_life
        wrm = self.config.world_realism
        if len(alive) < 12:
            return []
        stories: list[str] = []
        id_map = {p.person_id: p for p in alive}

        if sl.enabled:
            for p in alive:
                if p.jail_years_remaining > 0:
                    p.stress = min(1.0, p.stress + 0.022)
                    p.happiness = max(0.0, p.happiness - 0.035)
                    p.reputation = max(0.0, p.reputation - 0.006)

        jp = wrm.partner_jail_stress if wrm.enabled else 0.0
        if jp > 0:
            for p in alive:
                if p.jail_years_remaining > 0:
                    continue
                lid = p.love_partner_id
                if lid is None:
                    continue
                o = id_map.get(lid)
                if o is not None and o.alive and o.jail_years_remaining > 0:
                    p.stress = min(1.0, p.stress + jp)
                    p.happiness = max(0.0, p.happiness - jp * 0.72)

        for p in alive:
            lid = p.love_partner_id
            if lid is None:
                p.is_married = False
                continue
            o = id_map.get(lid)
            if o is None or not o.alive:
                p.love_partner_id = None
                p.is_married = False
                continue
            if o.love_partner_id != p.person_id:
                p.love_partner_id = None
                p.is_married = False
        for p in alive:
            if p.love_partner_id is None:
                p.is_married = False
                continue
            o = id_map.get(p.love_partner_id)
            if o is None or o.love_partner_id != p.person_id:
                p.love_partner_id = None
                p.is_married = False

        if wrm.enabled:
            splits = 0
            for p in alive:
                lid = p.love_partner_id
                if lid is None:
                    continue
                o = id_map.get(lid)
                if o is None or not o.alive:
                    continue
                pair = normalize_pair(p.person_id, o.person_id)
                if pair in self.enmities:
                    p.love_partner_id = None
                    o.love_partner_id = None
                    p.is_married = False
                    o.is_married = False
                    splits += 1
                    continue
                if p.region_id != o.region_id and self.rng.random() < wrm.breakup_long_distance_probability:
                    p.love_partner_id = None
                    o.love_partner_id = None
                    p.is_married = False
                    o.is_married = False
                    splits += 1
                    continue
                if max(p.stress, o.stress) >= wrm.breakup_high_stress_threshold and self.rng.random() < 0.24:
                    p.love_partner_id = None
                    o.love_partner_id = None
                    p.is_married = False
                    o.is_married = False
                    splits += 1
                    continue
                tr = self._pair_trust.get(pair, 0.5)
                if tr < wrm.breakup_low_trust and self.rng.random() < wrm.breakup_low_trust_probability:
                    p.love_partner_id = None
                    o.love_partner_id = None
                    p.is_married = False
                    o.is_married = False
                    splits += 1
            if splits and self.rng.random() < 0.26:
                stories.append(f"Y{year + 1}: {splits} couples separated (stress, distance, or feud).")

        if not sl.enabled:
            return stories

        attempts = min(sl.love_pair_attempts_per_year, max(30, len(alive) * 4))
        singles = [
            p
            for p in alive
            if p.jail_years_remaining == 0 and p.love_partner_id is None and 18 <= p.age <= 52
        ]
        self.rng.shuffle(singles)
        formed_love = 0
        for p in singles[:attempts]:
            if p.love_partner_id is not None:
                continue
            nbs = [id_map[n] for n in self.contact_graph.get(p.person_id, []) if n in id_map]
            candidates = [
                o
                for o in nbs
                if o.jail_years_remaining == 0
                and o.love_partner_id is None
                and 18 <= o.age <= 55
                and o.region_id == p.region_id
            ]
            if not candidates:
                continue
            o = self.rng.choice(candidates)
            pair = normalize_pair(p.person_id, o.person_id)
            if pair in self.enmities:
                continue
            tr = self._pair_trust.get(pair, 0.5)
            if tr < 0.5:
                continue
            if p.happiness < 0.32 or o.happiness < 0.32:
                continue
            p_ = sl.love_base_probability * (0.6 + tr) * (0.55 + 0.5 * min(p.happiness, o.happiness))
            if self.rng.random() > min(0.14, p_):
                continue
            p.love_partner_id = o.person_id
            o.love_partner_id = p.person_id
            p.happiness = min(1.0, p.happiness + 0.04)
            o.happiness = min(1.0, o.happiness + 0.04)
            self._pair_trust[pair] = min(1.0, tr + 0.05)
            formed_love += 1
        if formed_love and self.rng.random() < 0.28:
            stories.append(f"Y{year + 1}: {formed_love} new love bonds among neighbors")

        if wrm.enabled:
            seen_m: set[tuple[int, int]] = set()
            wed = 0
            for p in alive:
                lid = p.love_partner_id
                if lid is None or p.is_married:
                    continue
                o = id_map.get(lid)
                if o is None or o.is_married:
                    continue
                a, b = sorted((p.person_id, o.person_id))
                if (a, b) in seen_m:
                    continue
                seen_m.add((a, b))
                if p.age < wrm.marriage_min_age or o.age < wrm.marriage_min_age:
                    continue
                pair_m = normalize_pair(a, b)
                if self._pair_trust.get(pair_m, 0.5) < wrm.marriage_min_trust:
                    continue
                if self.rng.random() < wrm.marriage_probability_if_eligible:
                    p.is_married = True
                    o.is_married = True
                    p.happiness = min(1.0, p.happiness + 0.018)
                    o.happiness = min(1.0, o.happiness + 0.018)
                    wed += 1
            if wed and self.rng.random() < 0.3:
                stories.append(f"Y{year + 1}: {wed} marriage(s) formalized among partners.")

        n_as = min(sl.assault_samples_per_year, max(24, len(alive) // 2))
        assaults = 0
        jailed_assault = 0
        for _ in range(n_as):
            ag = self.rng.choice(alive)
            if ag.jail_years_remaining > 0 or ag.aggression < 0.36:
                continue
            nbs = [id_map[n] for n in self.contact_graph.get(ag.person_id, []) if n in id_map]
            if not nbs:
                continue
            vict = self.rng.choice(nbs)
            if vict.jail_years_remaining > 0:
                continue
            pair = normalize_pair(ag.person_id, vict.person_id)
            base = sl.assault_base_probability * (0.45 + ag.aggression) * (0.88 + ag.stress)
            base *= 0.75 + 0.55 * (1.0 - self._pair_trust.get(pair, 0.5))
            if pair in self.friendships:
                base *= 0.22
            if self.rng.random() > min(0.1, base):
                continue
            dmg = self.rng.uniform(0.045, 0.115)
            vict.health = max(0.0, vict.health - dmg)
            vict.stress = min(1.0, vict.stress + 0.11)
            vict.happiness = max(0.0, vict.happiness - 0.08)
            ag.stress = min(1.0, ag.stress + 0.04)
            self.enmities.add(pair)
            self.friendships.discard(pair)
            self._pair_trust[pair] = max(0.08, self._pair_trust.get(pair, 0.5) - 0.22)
            assaults += 1
            enf = float(self.region_policies.get(ag.region_id, {}).get("theft_enforcement", 0.35))
            catch = enf * (0.2 + 0.35 * vict.reputation) * (0.72 + 0.28 * self.rng.random())
            if self.rng.random() < catch:
                ag.crimes_caught_count += 1
                ag.jail_years_remaining = max(ag.jail_years_remaining, sl.jail_years_assault_caught)
                ag.reputation = max(0.0, ag.reputation - 0.12)
                jailed_assault += 1
        if assaults and self.rng.random() < 0.22:
            stories.append(
                f"Y{year + 1}: {assaults} violent assaults; {jailed_assault} aggressors jailed."
            )

        for p in alive:
            lid = p.love_partner_id
            if lid is None:
                continue
            o = id_map.get(lid)
            if o is None:
                continue
            pair_e = normalize_pair(p.person_id, o.person_id)
            if pair_e in self.enmities:
                p.love_partner_id = None
                o.love_partner_id = None
                p.is_married = False
                o.is_married = False

        n_prophets = sum(1 for p in alive if p.is_prophet)
        world_temple = any(s.get("kind") == "temple" for s in self.world_structures)
        avg_sp = sum(p.spiritual_tendency for p in alive) / len(alive)
        if (
            len(alive) >= sl.prophet_min_population
            and n_prophets < sl.max_prophets_world
            and (world_temple or avg_sp > 0.46)
        ):
            candidates = [
                p
                for p in alive
                if not p.is_prophet
                and not p.belief_group.startswith("cult_")
                and not is_prophet_movement_belief(p.belief_group)
                and p.spiritual_tendency >= sl.prophet_min_spiritual
                and p.knowledge >= sl.prophet_min_knowledge
                and p.reputation >= sl.prophet_min_reputation
                and 26 <= p.age <= 68
            ]
            self.rng.shuffle(candidates)
            for p in candidates[:6]:
                pr = sl.prophet_emerge_probability * (0.72 + 0.55 * p.spiritual_tendency)
                if self.rng.random() > pr:
                    continue
                mov = prophet_movement_id(p.person_id)
                p.is_prophet = True
                p.belief_group = mov
                p.reputation = min(1.0, p.reputation + 0.05)
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Prophet arises (#{p.person_id})",
                        f"A new teaching draws followers: {mov}.",
                    )
                )
                break

        prophets = [p for p in alive if p.is_prophet]
        converts = 0
        for pr in prophets:
            mov = prophet_movement_id(pr.person_id)
            if pr.belief_group != mov:
                pr.belief_group = mov
            nbs = [id_map[n] for n in self.contact_graph.get(pr.person_id, []) if n in id_map]
            for o in nbs:
                if o.region_id != pr.region_id:
                    continue
                if o.belief_group.startswith("cult_") or o.belief_group == mov:
                    continue
                conv = (
                    0.028
                    * (0.9 + pr.spiritual_tendency)
                    * (1.0 + 0.55 * o.stress)
                    * (0.75 + 0.45 * o.spiritual_tendency)
                )
                if pr.political_power > 0.45:
                    conv *= 1.08
                if self.rng.random() < min(0.14, conv):
                    o.belief_group = mov
                    o.spiritual_tendency = min(1.0, o.spiritual_tendency + 0.015)
                    converts += 1
        if converts and self.rng.random() < 0.2:
            stories.append(f"Y{year + 1}: {converts} people joined prophet-led worship")

        shrines = 0
        for pr in prophets:
            mov = prophet_movement_id(pr.person_id)
            byr = followers_by_region(alive, mov)
            for rid, c in byr.items():
                if c < sl.worship_followers_for_shrine:
                    continue
                if has_worship_shrine(self.world_structures, rid, mov):
                    continue
                idx = len([s for s in self.world_structures if s.get("kind") == "shrine"])
                self.world_structures.append(
                    {
                        "id": f"shrine_{rid}_{pr.person_id}_{idx}",
                        "kind": "shrine",
                        "movement": mov,
                        "prophet_id": pr.person_id,
                        "region_id": rid,
                        "slot": max(0.08, min(0.92, self.rng.uniform(0.18, 0.82))),
                        "slot_y": max(0.38, min(0.9, self.rng.uniform(0.44, 0.86))),
                    }
                )
                shrines += 1
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Worship hall: {self._region_name(rid)}",
                        f"Followers of {mov} established a gathering place.",
                    )
                )

        for p in alive:
            lid = p.love_partner_id
            if lid is None:
                continue
            o = id_map.get(lid)
            if o and o.alive and p.region_id == o.region_id and self.rng.random() < 0.45:
                p.happiness = min(1.0, p.happiness + 0.006)
                o.happiness = min(1.0, o.happiness + 0.006)

        return stories

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
        mood = self.world_dynamics.social_modifiers(self._world_aggression())
        friend_bar = cfg["friend_trust_threshold"] + mood["friend_trust_shift"]
        enemy_bar = cfg["enemy_conflict_threshold"] + mood["enemy_threshold_shift"]
        if self.config.conflict.preset.lower().strip() == "high_conflict":
            for p in alive:
                p.aggression = min(1.0, p.aggression + 0.004)
                p.stress = min(1.0, p.stress + 0.002)
        id_map = {p.person_id: p for p in alive}
        for person in alive:
            if person.jail_years_remaining > 0:
                continue
            for nid in self.contact_graph.get(person.person_id, []):
                if person.person_id >= nid:
                    continue
                other = id_map.get(nid)
                if other is None:
                    continue
                if other.jail_years_remaining > 0:
                    continue
                pair = normalize_pair(person.person_id, other.person_id)
                prev_trust = self._pair_trust.get(pair, 0.5)
                similarity = 1.0 - abs(person.spiritual_tendency - other.spiritual_tendency)
                trust_signal = (
                    similarity * 0.45
                    + (1.0 - abs(person.happiness - other.happiness)) * 0.25
                    + (1.0 - abs(person.aggression - other.aggression)) * 0.3
                )
                conflict_signal = (person.aggression + other.aggression) * 0.5 + abs(person.stress - other.stress) * 0.5
                conflict_signal += (person.ambition + other.ambition) * 0.18

                faction_tension = cfg["faction_tension"] if person.faction != other.faction else -0.02
                language_tension = cfg["language_tension"] if person.language != other.language else -0.01
                conflict_signal += faction_tension + language_tension

                trust_margin = trust_signal - friend_bar + (prev_trust - 0.5) * 0.24
                conflict_margin = conflict_signal - enemy_bar
                instability = self.world_dynamics.global_instability
                wa = self._world_aggression()
                bond_forms = trust_margin > (
                    0.06 + 0.05 * instability + 0.03 * max(0.0, wa - 1.0) - 0.05 * max(0.0, 1.0 - wa)
                )
                feud_forms = (
                    conflict_margin
                    > 0.04 + 0.06 * (1.0 - instability) - 0.03 * max(0.0, wa - 1.0) + 0.06 * max(0.0, 1.0 - wa)
                    and conflict_margin > trust_margin - 0.08 - 0.04 * max(0.0, wa - 1.0)
                )

                learned_trust = 0.91 * prev_trust + 0.09 * trust_signal
                learned_trust = max(0.0, min(1.0, learned_trust))

                if bond_forms and trust_margin * mood["friend_prob_scale"] > cfg["friend_prob"] * 0.35:
                    self.friendships.add(pair)
                    if pair in self.enmities:
                        self.enmities.discard(pair)
                    self._pair_trust[pair] = min(1.0, learned_trust + 0.13)
                elif feud_forms and conflict_margin * mood["enemy_prob_scale"] > cfg["enemy_prob"] * 0.55:
                    self.enmities.add(pair)
                    if pair in self.friendships:
                        self.friendships.discard(pair)
                    self._pair_trust[pair] = max(0.0, learned_trust - 0.21)
                else:
                    self._pair_trust[pair] = learned_trust

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
        self._pair_trust = {
            pair: v for pair, v in self._pair_trust.items() if pair[0] in alive_ids and pair[1] in alive_ids
        }

    def _rebuild_social_degree_cache(self) -> None:
        self._friend_degree_cache = {}
        self._enemy_degree_cache = {}
        for a, b in self.friendships:
            self._friend_degree_cache[a] = self._friend_degree_cache.get(a, 0) + 1
            self._friend_degree_cache[b] = self._friend_degree_cache.get(b, 0) + 1
        for a, b in self.enmities:
            self._enemy_degree_cache[a] = self._enemy_degree_cache.get(a, 0) + 1
            self._enemy_degree_cache[b] = self._enemy_degree_cache.get(b, 0) + 1

    def _individual_productivity(self, person: Individual) -> float:
        friend_count = self._friend_degree_cache.get(person.person_id, 0)
        enemy_count = self._enemy_degree_cache.get(person.person_id, 0)
        social_bonus = min(0.2, friend_count * 0.01) - min(0.18, enemy_count * 0.015)
        emotional = person.happiness * 0.45 - person.stress * 0.35 - person.aggression * 0.15
        skill = person.knowledge * 0.22 + person.tool_skill * 0.28
        assets = min(0.15, person.personal_tools * 0.01 + person.books_authored * 0.015)
        mount = (
            min(0.055, person.riding_skill * 0.048)
            if "animal_husbandry" in self.world_inventions
            else 0.0
        )
        mutation_penalty = min(0.25, person.mutation_burden * 0.3)
        return max(-0.45, min(0.9, social_bonus + emotional + skill + assets + mount - mutation_penalty))

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
        if "agriculture" not in self.unlocked_milestones and self.current_era in (
            "agrarian",
            "classical",
            "industrial",
            "modern",
            "information age",
        ):
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
                    "slot": max(0.1, min(0.9, self.rng.uniform(0.18, 0.82))),
                    "slot_y": max(0.38, min(0.9, self.rng.uniform(0.48, 0.86))),
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
        by_region: dict[int, list[Individual]] = defaultdict(list)
        for person in alive:
            by_region[person.region_id].append(person)

        if (not self.agriculture_unlocked) and (self.current_era != "hunter-gatherer" or avg_tools > 0.42):
            self.agriculture_unlocked = True
            stories.append(
                self._register_timeline_transition(
                    year,
                    "Agriculture practice adopted",
                    "Food production shifted from gathering to cultivation.",
                )
            )

        # Settlement progression as concrete state transitions per region/community.
        for region_id, residents in by_region.items():
            settlement = self._get_settlement_structure(region_id)
            if settlement is None:
                settlement = {
                    "id": f"settlement_{region_id}",
                    "name": self._generate_settlement_name(region_id),
                    "kind": "settlement",
                    "level": "camp",
                    "region_id": region_id,
                    "slot": max(0.08, min(0.92, self.rng.uniform(0.12, 0.88))),
                    "slot_y": max(0.34, min(0.92, self.rng.uniform(0.42, 0.88))),
                    "culture": "tribal",
                    "religion": "ancestor",
                }
                self.world_structures.append(settlement)
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"New settlement founded: {settlement['name']}",
                        "Migrants established a new community center.",
                    )
                )

            rpop = len(residents)
            ravg_knowledge = sum(p.knowledge for p in residents) / rpop
            ravg_tools = sum(p.tool_skill for p in residents) / rpop
            level = str(settlement["level"])
            next_level = None
            if level == "camp" and rpop >= 14 and self.agriculture_unlocked:
                next_level = "village"
            elif level == "village" and rpop >= 55 and (ravg_knowledge + ravg_tools) * 0.5 > 0.34:
                next_level = "town"
            elif level == "town" and rpop >= 130 and (ravg_knowledge + ravg_tools) * 0.5 > 0.5:
                next_level = "city"
            if next_level:
                settlement["level"] = next_level
                if next_level == "city":
                    settlement.setdefault("polity", "city")
                    if not str(settlement.get("name", "")).endswith(" City"):
                        stem = settlement_name_stem(str(settlement.get("name", "")))
                        settlement["name"] = format_settlement_polity_name(stem, "city", (ravg_knowledge + ravg_tools) * 0.5)
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Settlement evolved: {settlement['name']} {level} -> {next_level}",
                        f"Regional population and institutions enabled a {next_level}.",
                    )
                )

            # City-scale polity: unified country, then empire (same region; names + government bias).
            if str(settlement.get("level", "")) == "city":
                pol = str(settlement.get("polity", "city"))
                civ_r = (ravg_knowledge + ravg_tools) * 0.5
                pcfg = self.config.politics
                if pcfg.polity_progression and pol == "city":
                    state_ok = ("state" in self.unlocked_milestones) or (rpop >= max(pcfg.country_min_city_pop + 40, 205))
                    if (
                        rpop >= pcfg.country_min_city_pop
                        and civ_r >= pcfg.country_min_civ_region
                        and (not pcfg.country_requires_state_milestone or state_ok)
                    ):
                        settlement["polity"] = "country"
                        stem = settlement_name_stem(str(settlement.get("name", "")))
                        settlement["name"] = format_settlement_polity_name(stem, "country", civ_r)
                        stories.append(
                            self._register_timeline_transition(
                                year,
                                f"Country formed: {settlement['name']}",
                                "A city-scale polity unifies law, borders, and identity as a country.",
                            )
                        )
                elif pcfg.polity_progression and pol == "country":
                    avg_amb = sum(p.ambition for p in residents) / rpop
                    expansion = avg_amb > pcfg.empire_ambition_threshold or "iron_working" in self.world_inventions
                    if (
                        rpop >= pcfg.empire_min_country_pop
                        and civ_r >= pcfg.empire_min_civ_region
                        and expansion
                    ):
                        settlement["polity"] = "empire"
                        stem = settlement_name_stem(str(settlement.get("name", "")))
                        settlement["name"] = format_settlement_polity_name(stem, "empire", civ_r)
                        stories.append(
                            self._register_timeline_transition(
                                year,
                                f"Empire proclaimed: {settlement['name']}",
                                "Central power expands; the realm is recognized as an empire.",
                            )
                        )

        # Agriculture fields become visible structures after agriculture unlock.
        if self.agriculture_unlocked:
            existing_fields = [s for s in self.world_structures if s["kind"] == "field"]
            existing_by_region: dict[int, int] = defaultdict(int)
            for field in existing_fields:
                existing_by_region[int(field.get("region_id", 0))] += 1
            for region_id, residents in by_region.items():
                target_fields = min(8, max(1, int(len(residents) / 26)))
                while existing_by_region.get(region_id, 0) < target_fields:
                    idx = len([s for s in self.world_structures if s.get("kind") == "field"])
                    self.world_structures.append(
                        {
                            "id": f"field_{region_id}_{idx}",
                            "kind": "field",
                            "level": 1,
                            "region_id": region_id,
                            "slot": max(0.06, min(0.94, self.rng.uniform(0.08, 0.92))),
                            "slot_y": max(0.32, min(0.94, self.rng.uniform(0.36, 0.92))),
                        }
                    )
                    existing_by_region[region_id] += 1
            # Register first-time field establishment only once.
            if any(s.get("kind") == "field" for s in self.world_structures):
                if not any(e["title"] == "Agricultural fields established" for e in self.timeline_events):
                    stories.append(
                        self._register_timeline_transition(
                            year,
                            "Agricultural fields established",
                            "Farms appeared around the settlements.",
                        )
                    )

        # Institutions based on social practice per region.
        for region_id, residents in by_region.items():
            rpop = len(residents)
            ravg_knowledge = sum(p.knowledge for p in residents) / rpop
            ravg_tools = sum(p.tool_skill for p in residents) / rpop
            ravg_spiritual = sum(p.spiritual_tendency for p in residents) / rpop
            rbelief_groups = len({p.belief_group for p in residents})

            if ravg_knowledge > 0.55 and rpop >= 45 and not self._has_structure_in_region("school", region_id):
                self.world_structures.append(
                    {
                        "id": f"school_{region_id}",
                        "kind": "school",
                        "level": 1,
                        "region_id": region_id,
                        "slot": max(0.08, min(0.9, self.rng.uniform(0.15, 0.45))),
                        "slot_y": max(0.4, min(0.9, self.rng.uniform(0.5, 0.88))),
                    }
                )
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"School founded in {self._region_name(region_id)}",
                        "Knowledge density supports formal learning.",
                    )
                )
            if ravg_tools > 0.58 and rpop >= 40 and not self._has_structure_in_region("workshop", region_id):
                self.world_structures.append(
                    {
                        "id": f"workshop_{region_id}",
                        "kind": "workshop",
                        "level": 1,
                        "region_id": region_id,
                        "slot": max(0.1, min(0.92, self.rng.uniform(0.55, 0.9))),
                        "slot_y": max(0.42, min(0.9, self.rng.uniform(0.52, 0.86))),
                    }
                )
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Workshop district formed in {self._region_name(region_id)}",
                        "Craft specialization creates workshops.",
                    )
                )
            if (ravg_spiritual > 0.55 or rbelief_groups >= 2) and rpop >= 40 and not self._has_structure_in_region(
                "temple",
                region_id,
            ):
                self.world_structures.append(
                    {
                        "id": f"temple_{region_id}",
                        "kind": "temple",
                        "level": 1,
                        "region_id": region_id,
                        "slot": max(0.1, min(0.9, self.rng.uniform(0.25, 0.75))),
                        "slot_y": max(0.38, min(0.88, self.rng.uniform(0.44, 0.82))),
                    }
                )
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Temple built in {self._region_name(region_id)}",
                        "Shared ritual practices establish a temple.",
                    )
                )

        split_stories = self._simulate_city_separation_and_civil_war(year, alive, by_region)
        stories.extend(split_stories)

        politics_stories = self._update_politics(year, alive)
        stories.extend(politics_stories)
        self._refresh_settlement_identities(alive)
        return stories

    def _register_timeline_transition(self, year: int, title: str, details: str) -> str:
        record = {"year": year + 1, "title": title, "details": details}
        self.timeline_events.append(record)
        self.major_events.append(record)
        return f"Y{year + 1}: {title}"

    def _has_structure(self, kind: str) -> bool:
        return any(s["kind"] == kind for s in self.world_structures)

    def _has_structure_in_region(self, kind: str, region_id: int) -> bool:
        return any(s["kind"] == kind and int(s.get("region_id", -1)) == region_id for s in self.world_structures)

    def _get_settlement_structure(self, region_id: int) -> dict[str, int | str | float] | None:
        for structure in self.world_structures:
            if structure["kind"] == "settlement" and structure["region_id"] == region_id:
                return structure
        return None

    def _region_name(self, region_id: int) -> str:
        settlement = self._get_settlement_structure(region_id)
        if settlement is None:
            return f"Region-{region_id}"
        return str(settlement.get("name", f"Region-{region_id}"))

    def _settlement_level_max_band_index(self, level: str) -> int:
        return {"camp": 0, "village": 1, "town": 2, "city": 3}.get(level, 0)

    def _sample_living_context(self, region_id: int) -> str:
        st = self._get_settlement_structure(region_id)
        level = str(st.get("level", "camp")) if st else "camp"
        max_i = self._settlement_level_max_band_index(level)
        bands = self._LIVING_BANDS[: max_i + 1]
        if len(bands) == 1:
            return bands[0]
        polity = str(st.get("polity", "city")) if st and level == "city" else "city"
        if max_i == 1:
            weights = [0.52, 0.48]
        elif max_i == 2:
            weights = [0.34, 0.36, 0.30]
        else:
            w_city = 0.38
            if polity == "country":
                w_city = 0.32
            elif polity == "empire":
                w_city = 0.44
            rem = 1.0 - w_city
            weights = [rem * 0.42, rem * 0.33, rem * 0.25, w_city]
        return self.rng.choices(bands, weights=weights, k=1)[0]

    def _clamp_living_context(self, person: Individual) -> None:
        st = self._get_settlement_structure(person.region_id)
        level = str(st.get("level", "camp")) if st else "camp"
        max_i = self._settlement_level_max_band_index(level)
        try:
            ci = self._LIVING_BANDS.index(person.living_context)
        except ValueError:
            ci = 0
        if ci > max_i:
            person.living_context = self._LIVING_BANDS[max_i]

    def _bootstrap_living_contexts(self, alive: list[Individual]) -> None:
        for p in alive:
            p.living_context = self._sample_living_context(p.region_id)
            self._clamp_living_context(p)

    def _evolve_living_contexts(self, alive: list[Individual], food_ratio_by_region: dict[int, float]) -> None:
        for p in alive:
            if p.jail_years_remaining > 0:
                continue
            if self.rng.random() >= 0.042:
                continue
            fr = food_ratio_by_region.get(p.region_id, 1.0)
            urban_bias = p.ambition * 0.22 + min(1.0, p.wealth / 2.8) * 0.2 + p.knowledge * 0.12
            rural_bias = max(0.0, 0.9 - fr) * 0.55
            try:
                i = self._LIVING_BANDS.index(p.living_context)
            except ValueError:
                i = 0
            st = self._get_settlement_structure(p.region_id)
            level = str(st.get("level", "camp")) if st else "camp"
            max_i = self._settlement_level_max_band_index(level)
            if rural_bias > urban_bias + 0.08 and i > 0:
                p.living_context = self._LIVING_BANDS[i - 1]
            elif urban_bias > rural_bias + 0.08 and i < max_i:
                p.living_context = self._LIVING_BANDS[i + 1]
            else:
                p.living_context = self._sample_living_context(p.region_id)
            self._clamp_living_context(p)

    def _effective_government(self, settlement_level: str, civ: float, polity: str = "city") -> str:
        mode = self.config.politics.government_mode
        if settlement_level == "camp":
            return "informal"
        if settlement_level == "village":
            return "chiefdom"
        if mode == "auto":
            if settlement_level == "town":
                return "democracy" if civ > 0.48 else "oligarchy"
            if settlement_level == "city":
                if polity == "empire":
                    return "monarchy" if civ > 0.5 else "autocracy"
                if polity == "country":
                    return "democracy" if civ > 0.53 else "oligarchy"
                return "democracy" if civ > 0.55 else "oligarchy"
            return "oligarchy"
        if mode == "chiefdom":
            return "oligarchy"
        if mode in ("democracy", "republic"):
            return "democracy"
        if mode == "oligarchy":
            return "oligarchy"
        if mode == "monarchy":
            return "monarchy"
        if mode == "autocracy":
            return "autocracy"
        return "oligarchy"

    def _leader_title_for_government(self, gov: str, polity: str = "city") -> str:
        if polity == "empire" and gov in ("monarchy", "autocracy"):
            return "Emperor"
        if gov == "democracy" and self.config.politics.government_mode == "republic":
            return "Prime Minister"
        return {
            "informal": "",
            "chiefdom": "Chief",
            "oligarchy": "Council Chair",
            "democracy": "President",
            "republic": "President",
            "monarchy": "Monarch",
            "autocracy": "Ruler",
        }.get(gov, "Leader")

    def _recompute_political_power(self, person: Individual, settlement_level: str, polity: str = "city") -> None:
        tier = {"camp": 1.0, "village": 1.06, "town": 1.14, "city": 1.24}.get(settlement_level, 1.0)
        if settlement_level == "city":
            if polity == "country":
                tier *= 1.07
            elif polity == "empire":
                tier *= 1.14
        books = min(1.0, person.books_authored / 6.0)
        tools = min(1.0, person.personal_tools / 10.0)
        age_bonus = min(0.22, max(0.0, (person.age - 20) * 0.004))
        base = (
            0.27 * person.knowledge
            + 0.24 * person.tool_skill
            + 0.12 * person.spiritual_tendency
            + 0.12 * books
            + 0.1 * tools
            + 0.11 * person.ambition
            + age_bonus
        )
        person.political_power = max(0.0, min(1.0, base * tier))

    def _update_politics(self, year: int, alive: list[Individual]) -> list[str]:
        stories: list[str] = []
        cfg = self.config.politics
        by_region: dict[int, list[Individual]] = defaultdict(list)
        for p in alive:
            by_region[p.region_id].append(p)

        for structure in self.world_structures:
            if structure.get("kind") != "settlement":
                continue
            region_id = int(structure.get("region_id", 0))
            level = str(structure.get("level", "camp"))
            polity = str(structure.get("polity", "city")) if level == "city" else "city"
            residents = by_region.get(region_id, [])
            if not residents:
                continue

            prev = self.politics_by_region.get(region_id, {})
            prev_gov = str(prev.get("government", "informal"))
            prev_leader = prev.get("leader_id")
            prev_leader_int = int(prev_leader) if isinstance(prev_leader, int) else None

            gov = self._effective_government(level, self.civilization_index, polity)
            elite_n = max(2, min(len(residents), int(len(residents) * cfg.elite_fraction) + 1))

            for person in residents:
                self._recompute_political_power(person, level, polity)

            sorted_r = sorted(residents, key=lambda x: x.political_power, reverse=True)
            elite_ids = [p.person_id for p in sorted_r[:elite_n]]

            leader_alive_id = prev_leader_int
            if leader_alive_id is not None:
                if not any(p.person_id == leader_alive_id for p in residents):
                    leader_alive_id = None

            if leader_alive_id is not None:
                for p in residents:
                    if p.person_id == leader_alive_id:
                        p.political_power = min(1.0, p.political_power + cfg.leader_power_bonus)
                        break

            new_leader_id: int | None = leader_alive_id
            next_election: int | None = prev.get("next_election_year")  # type: ignore[assignment]
            monarch_faction = prev.get("monarch_faction")

            if gov == "informal":
                state: dict[str, object] = {
                    "government": "informal",
                    "leader_id": None,
                    "leader_title": "",
                    "elite_ids": [],
                    "next_election_year": None,
                    "monarch_faction": None,
                }
                if prev_gov != "informal":
                    stories.append(
                        self._register_timeline_transition(
                            year,
                            "Politics: informal gathering",
                            "No formal ruler; leadership is situational.",
                        )
                    )
                self.politics_by_region[region_id] = state
                continue

            election_due = False
            if gov in ("democracy", "republic"):
                ne = prev.get("next_election_year")
                if ne is None or (isinstance(ne, int) and year >= ne):
                    election_due = True
                if leader_alive_id is None:
                    election_due = True

            if gov == "chiefdom":
                adults = [p for p in residents if p.age >= 20]
                pool = adults if adults else residents

                def chief_score(p: Individual) -> float:
                    return (
                        p.spiritual_tendency * 0.5
                        + min(1.0, p.age / 88.0) * 0.35
                        + p.political_power * 0.32
                    )

                best = max(pool, key=chief_score)
                if leader_alive_id is None:
                    new_leader_id = best.person_id
                else:
                    current = next((p for p in pool if p.person_id == leader_alive_id), None)
                    if current is None:
                        new_leader_id = best.person_id
                    elif chief_score(best) > chief_score(current) * 1.12:
                        new_leader_id = best.person_id
                    else:
                        new_leader_id = leader_alive_id
                next_election = None

            elif gov == "oligarchy":
                elite_people = [p for p in sorted_r if p.person_id in elite_ids]
                pool = elite_people if elite_people else sorted_r
                if leader_alive_id is None:
                    new_leader_id = pool[0].person_id
                else:
                    cur = next((p for p in pool if p.person_id == leader_alive_id), None)
                    if cur is None:
                        new_leader_id = pool[0].person_id
                    elif self.rng.random() < 0.04 and pool[0].person_id != leader_alive_id:
                        new_leader_id = pool[0].person_id
                    else:
                        new_leader_id = leader_alive_id
                next_election = None

            elif gov == "democracy":
                cand_n = max(3, min(len(sorted_r), int(len(residents) * 0.2) + 2))
                candidates = sorted_r[:cand_n]
                weights = [max(0.06, p.political_power * (0.35 + p.happiness)) for p in candidates]
                if election_due or leader_alive_id is None:
                    picked = self.rng.choices(candidates, weights=weights, k=1)[0]
                    new_leader_id = picked.person_id
                    next_election = year + max(3, cfg.election_interval_years)
                else:
                    new_leader_id = leader_alive_id
                    next_election = ne if isinstance(ne, int) else year + cfg.election_interval_years

            elif gov == "monarchy":
                fac = monarch_faction if isinstance(monarch_faction, str) else None
                if leader_alive_id is None:
                    if fac:
                        same = [p for p in sorted_r if p.faction == fac]
                        pick_from = same if same else sorted_r
                    else:
                        pick_from = sorted_r
                    new_leader_id = pick_from[0].person_id
                else:
                    new_leader_id = leader_alive_id
                next_election = None
                lp = next((p for p in residents if p.person_id == new_leader_id), None)
                monarch_faction = lp.faction if lp else fac

            else:  # autocracy
                best = max(
                    residents,
                    key=lambda p: p.political_power * (0.55 + 0.45 * p.aggression),
                )
                if leader_alive_id is None:
                    new_leader_id = best.person_id
                else:
                    cur = next((p for p in residents if p.person_id == leader_alive_id), None)
                    if cur is None:
                        new_leader_id = best.person_id
                    elif (
                        best.person_id != leader_alive_id
                        and best.political_power * (0.55 + 0.45 * best.aggression)
                        > cur.political_power * (0.55 + 0.45 * cur.aggression) * 1.2
                    ):
                        new_leader_id = best.person_id
                    else:
                        new_leader_id = leader_alive_id
                next_election = None

            title = self._leader_title_for_government(gov, polity)
            state = {
                "government": gov,
                "leader_id": new_leader_id,
                "leader_title": title,
                "elite_ids": elite_ids,
                "next_election_year": next_election,
                "monarch_faction": monarch_faction if gov == "monarchy" else None,
            }
            self.politics_by_region[region_id] = state

            if prev_gov != gov:
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Government: {gov.replace('_', ' ')}",
                        f"The settlement organizes as {gov}; power stratifies with size and literacy.",
                    )
                )
            if new_leader_id != prev_leader_int and new_leader_id is not None:
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"{title} #{new_leader_id}",
                        f"Leadership passes in a {gov} system; elites hold ~{cfg.elite_fraction:.0%} of formal sway.",
                    )
                )

        return stories

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
            pol = self.politics_by_region.get(region_id, {})
            structure["government"] = str(pol.get("government", "informal"))
            structure["leader_id"] = pol.get("leader_id")
            lt = pol.get("leader_title", "")
            structure["leader_title"] = str(lt) if lt is not None else ""
            power_style = self._power_style_label(residents, str(pol.get("government", "informal")))
            community = self._community_model_label(residents, structure.get("level", "camp"))
            structure["power_style"] = power_style
            structure["community"] = community
            structure["resource_score"] = round(self.environments[region_id].resource_score(), 3)
            if structure.get("level") == "city":
                self.city_summaries.append(
                    {
                        "name": str(structure.get("name", "Unknown")),
                        "polity": str(structure.get("polity", "city")),
                        "culture": str(culture),
                        "religion": str(religion),
                        "faction": str(faction),
                        "language": str(language),
                        "population": pop,
                        "government": str(pol.get("government", "informal")),
                        "leader_title": str(lt) if lt is not None else "",
                        "leader_id": pol.get("leader_id"),
                        "power_style": str(power_style),
                        "community": str(community),
                        "resource_score": round(self.environments[region_id].resource_score(), 3),
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

    def _power_style_label(self, residents: list[Individual], government: str) -> str:
        if not residents:
            return "informal"
        powers = sorted((p.political_power for p in residents), reverse=True)
        top_n = max(1, min(len(powers), int(len(powers) * 0.1)))
        concentration = sum(powers[:top_n]) / max(1e-6, sum(powers))
        avg_aggr = sum(p.aggression for p in residents) / len(residents)
        if government in ("autocracy", "monarchy") and concentration > 0.36:
            return "centralized-rule"
        if government == "oligarchy" or concentration > 0.43:
            return "elite-council"
        if avg_aggr > 0.62:
            return "military-command"
        if government in ("democracy", "republic"):
            return "civic-representation"
        return "local-assembly"

    def _community_model_label(self, residents: list[Individual], level: str | int) -> str:
        if not residents:
            return "mixed"
        avg_knowledge = sum(p.knowledge for p in residents) / len(residents)
        avg_tools = sum(p.tool_skill for p in residents) / len(residents)
        avg_spiritual = sum(p.spiritual_tendency for p in residents) / len(residents)
        avg_aggr = sum(p.aggression for p in residents) / len(residents)
        if avg_aggr > 0.62:
            return "militant"
        if avg_spiritual > 0.7:
            return "faith-centered"
        if avg_knowledge > 0.7:
            return "scholarly"
        if avg_tools > 0.68:
            return "craft-industrial"
        if level == "camp":
            return "tribal"
        return "civic-mercantile"

    def _simulate_city_separation_and_civil_war(
        self,
        year: int,
        alive: list[Individual],
        by_region: dict[int, list[Individual]],
    ) -> list[str]:
        stories: list[str] = []
        if len(alive) < 80:
            return stories
        for region_id, residents in list(by_region.items()):
            if len(residents) < 70:
                continue
            settlement = self._get_settlement_structure(region_id)
            if settlement is None:
                continue
            level = str(settlement.get("level", "camp"))
            if level not in ("town", "city"):
                continue

            avg_stress = sum(p.stress for p in residents) / len(residents)
            avg_aggr = sum(p.aggression for p in residents) / len(residents)
            factions = {p.faction for p in residents}
            languages = {p.language for p in residents}
            fragmentation = min(1.0, (len(factions) - 1) * 0.16 + (len(languages) - 1) * 0.11)
            civil_war_prob = min(0.22, max(0.0, (avg_stress - 0.5) * 0.3 + (avg_aggr - 0.5) * 0.25 + fragmentation * 0.2))
            secession_prob = min(0.16, max(0.0, (avg_stress - 0.45) * 0.24 + fragmentation * 0.18))

            event: str | None = None
            if self.rng.random() < civil_war_prob:
                event = "civil_war"
            elif self.rng.random() < secession_prob:
                event = "secession"
            if event is None:
                continue

            newcomer_region = len(self.environments)
            self.environments.append(Environment(self.config.environment, self.rng))
            self.config.demographics.region_count = len(self.environments)
            self.region_harvest_factor[newcomer_region] = self.rng.uniform(0.86, 1.07)

            split_by_faction: dict[str, list[Individual]] = defaultdict(list)
            for p in residents:
                split_by_faction[p.faction].append(p)
            target_faction, target_group = max(split_by_faction.items(), key=lambda kv: len(kv[1]))
            movers = target_group[:]
            min_movers = max(18, int(len(residents) * 0.22))
            if len(movers) < min_movers:
                others = [p for p in residents if p.faction != target_faction]
                self.rng.shuffle(others)
                movers.extend(others[: max(0, min_movers - len(movers))])

            self.rng.shuffle(movers)
            movers = movers[: max(12, min(len(movers), int(len(residents) * 0.45)))]
            new_name = self._generate_settlement_name(newcomer_region)
            new_level = "village" if len(movers) < 80 else "town"
            self.world_structures.append(
                {
                    "id": f"settlement_{newcomer_region}",
                    "name": new_name,
                    "kind": "settlement",
                    "level": new_level,
                    "region_id": newcomer_region,
                    "slot": max(0.1, min(0.9, self.rng.uniform(0.2, 0.82))),
                    "slot_y": max(0.36, min(0.9, self.rng.uniform(0.44, 0.86))),
                    "culture": "mixed",
                    "religion": self._dominant_belief(movers),
                }
            )
            for p in movers:
                p.region_id = newcomer_region
                p.living_context = self._sample_living_context(newcomer_region)
                self._clamp_living_context(p)
                p.stress = max(0.0, p.stress - 0.06)

            if event == "civil_war":
                casualties = self._apply_direct_deaths(movers, 0.07, 2)
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Civil war in {self._region_name(region_id)}",
                        f"{casualties} deaths; rebels founded {new_name}.",
                    )
                )
            else:
                stories.append(
                    self._register_timeline_transition(
                        year,
                        f"Secession: {new_name} splits",
                        f"A {target_faction} bloc left {self._region_name(region_id)} and formed a new polity.",
                    )
                )
        return stories

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
        tcfg = self.config.technology
        craft_rate = 1.0 + 0.34 * int("advanced_tools" in self.world_inventions)
        drain_s = max(0.0, tcfg.resource_drain_scale)
        for person in alive:
            env = self.environments[person.region_id]
            if tcfg.resource_gated_tool_crafting and not region_can_craft_tools(env):
                craft_chance_scale = 0.22
            else:
                craft_chance_scale = 1.0
            if person.tool_skill > 0.45 and self.rng.random() < (
                (0.01 + person.tool_skill * 0.02) * craft_rate * craft_chance_scale
            ):
                if tcfg.resource_gated_tool_crafting and not region_can_craft_tools(env):
                    continue
                person.personal_tools += 1
                self.total_tools_crafted += 1
                if tcfg.resource_gated_tool_crafting:
                    apply_tool_craft_drain(env, scale=drain_s)

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

        # Defensive pacts build from sustained calm between belief blocs (latent goodwill).
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1 :]:
                key = tuple(sorted((g1, g2)))
                if key in self.alliances:
                    continue
                a1 = sum(p.aggression for p in by_belief[g1]) / len(by_belief[g1])
                a2 = sum(p.aggression for p in by_belief[g2]) / len(by_belief[g2])
                s1 = sum(p.stress for p in by_belief[g1]) / len(by_belief[g1])
                s2 = sum(p.stress for p in by_belief[g2]) / len(by_belief[g2])
                if self.world_dynamics.step_belief_alliance(g1, g2, a1, a2, s1, s2):
                    self.alliances.add(key)
                    msg = self._register_timeline_transition(
                        year,
                        "Alliance formed",
                        f"Groups {g1} and {g2} formed a defensive pact.",
                    )
                    stories.append(msg)

        # Internal resource war: pressure accumulates until the social system tips.
        pop = len(alive)
        avg_stress = sum(p.stress for p in alive) / pop
        food_pc = self._available_food_total(alive) / pop if pop else 1.0
        alliance_factor = max(0.0, 1.0 - len(self.alliances) * cfg["alliance_damp"])
        war_pressure = max(0.0, (avg_stress - 0.45) + max(0.0, 1.0 - food_pc) + (len(groups) - 2) * 0.08)
        war_pressure = min(cfg["war_cap"], war_pressure * cfg["war_scale"] * alliance_factor)
        if self.world_dynamics.step_internal_war(war_pressure, year, self._world_aggression()):
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
        wa = self._world_aggression()
        if preset == "high_conflict":
            d = {
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
        else:
            d = {
                "friend_trust_threshold": 0.65,
                "friend_prob": 0.08,
                "enemy_conflict_threshold": 0.56,
                "enemy_prob": 0.11,
                "alliance_prob": 0.02,
                "alliance_damp": 0.06,
                "war_scale": 0.06,
                "war_cap": 0.18,
                "faction_tension": 0.08,
                "language_tension": 0.06,
            }
        d["faction_tension"] = min(0.28, d["faction_tension"] * wa)
        d["language_tension"] = min(0.22, d["language_tension"] * wa)
        d["war_scale"] = min(0.22, d["war_scale"] * (0.65 + 0.35 * wa))
        d["war_cap"] = min(0.42, d["war_cap"] * (0.7 + 0.3 * wa))
        return d

