from dataclasses import dataclass, field


@dataclass
class PathogenConfig:
    name: str = "flu"
    initial_infected_fraction: float = 0.02
    infection_rate: float = 0.15
    recovery_rate: float = 0.12
    mortality_rate: float = 0.03
    immunity_loss_rate: float = 0.01
    mutation_rate: float = 0.01
    mutation_strength: float = 0.03


@dataclass
class EnvironmentConfig:
    base_food_per_capita: float = 1.0
    food_variability: float = 0.15
    environmental_stress: float = 0.05
    shock_probability: float = 0.03
    shock_severity: float = 0.35


@dataclass
class SeasonConfig:
    """Annual cycle: each simulated year advances one season (spring→summer→autumn→winter)."""

    enabled: bool = True
    names: tuple[str, str, str, str] = ("spring", "summer", "autumn", "winter")
    # Applied to regional food after environment.available_food and trade/ecology modifiers.
    food_multiplier: tuple[float, float, float, float] = (0.93, 1.14, 1.03, 0.81)
    # Extra migration pressure (e.g. harsh winter / lean season movement).
    migration_multiplier: tuple[float, float, float, float] = (1.02, 0.94, 1.04, 1.16)
    # Contact-season transmission (e.g. winter crowding).
    disease_transmission_multiplier: tuple[float, float, float, float] = (0.95, 0.97, 1.03, 1.1)
    # Foraging / wild harvest / pasture quality (scaled on non-field ecology bonus only).
    wildlife_food_multiplier: tuple[float, float, float, float] = (1.03, 1.06, 0.97, 0.86)
    # Shifts which season year 0 uses: index = (year + phase_offset) % 4.
    phase_offset: int = 0


@dataclass
class MigrationConfig:
    enabled: bool = True
    migration_rate: float = 0.02
    cross_region_contact_rate: float = 0.05


@dataclass
class VaccinationPolicyConfig:
    enabled: bool = True
    start_year: int = 20
    annual_coverage_fraction: float = 0.08
    min_age: int = 12
    effectiveness: float = 0.6
    max_per_year: int = 250


@dataclass
class ContactNetworkConfig:
    avg_contacts: int = 10
    rewiring_rate: float = 0.1


@dataclass
class BehaviorConfig:
    enabled: bool = True
    migration_food_weight: float = 0.7
    migration_infection_weight: float = 0.3
    contact_avoid_infected_bias: float = 0.65
    stress_birth_penalty_weight: float = 0.55


@dataclass
class WorldRealismConfig:
    """Marriage, birth risk, schools, corruption, unrest, sanitation, elder support."""

    enabled: bool = True
    marriage_probability_if_eligible: float = 0.11
    marriage_min_trust: float = 0.56
    marriage_min_age: int = 20
    breakup_long_distance_probability: float = 0.11
    breakup_high_stress_threshold: float = 0.9
    breakup_low_trust: float = 0.38
    breakup_low_trust_probability: float = 0.08
    married_birth_fertility_bonus: float = 0.065
    maternal_mortality_enabled: bool = True
    maternal_mortality_base: float = 0.0036
    maternal_mortality_health_scale: float = 1.05
    school_learning_bonus: float = 0.005
    school_max_age: int = 38
    treasury_corruption_enabled: bool = True
    treasury_corruption_enforcement_lt: float = 0.38
    treasury_corruption_max_fraction: float = 0.014
    unrest_food_ratio_below: float = 0.82
    unrest_region_mean_stress_above: float = 0.58
    unrest_event_probability: float = 0.15
    unrest_instability_bump: float = 0.042
    sanitation_requires_writing: bool = True
    sanitation_min_mean_settlement_tier: float = 1.12
    sanitation_transmission_reduction: float = 0.11
    elder_support_age_min: int = 58
    elder_partner_health_bonus: float = 0.0075
    # Regional harvest quality (weather / pests / soil luck), persistent year to year.
    harvest_weather_enabled: bool = True
    harvest_volatility: float = 0.085
    harvest_persistence: float = 0.62
    harvest_min_factor: float = 0.58
    harvest_max_factor: float = 1.28
    # Slow-moving nutrition vs food ratio: chronic stress and health when diets stay poor.
    nutrition_tracking_enabled: bool = True
    nutrition_ema_alpha: float = 0.17
    nutrition_stress_threshold: float = 0.56
    nutrition_chronic_stress_per_year: float = 0.009
    nutrition_chronic_health_penalty: float = 0.0055
    nutrition_recovery_health_bonus: float = 0.0028
    # More transmission when many people live in town/city bands (crowding).
    urban_crowding_disease_boost: float = 0.072
    # Town/city households pay more when local food is tight (rent / imports).
    dense_housing_rent_drag: float = 0.0038
    # Non-jailed partner absorbs stress while spouse is incarcerated.
    partner_jail_stress: float = 0.013
    # Rare multi-year shocks (drought, flood, crop blight) on top of harvest noise.
    regional_disaster_enabled: bool = True
    regional_disaster_probability: float = 0.042
    regional_disaster_min_years: int = 1
    regional_disaster_max_years: int = 3
    regional_disaster_food_mult_low: float = 0.66
    regional_disaster_food_mult_high: float = 0.91
    # Contacts grieve when someone they know dies in the yearly mortality pass.
    bereavement_enabled: bool = True
    bereavement_stress: float = 0.035
    bereavement_happiness_hit: float = 0.021
    bereavement_max_contacts: int = 12
    # Neighbors lean on each other when food is tight (trust + mood).
    mutual_aid_enabled: bool = True
    mutual_aid_food_below: float = 0.86
    mutual_aid_attempt_probability: float = 0.13
    mutual_aid_happiness: float = 0.012
    mutual_aid_trust: float = 0.015
    mutual_aid_min_trust: float = 0.51
    # Sustained ambition / striving adds baseline stress (overwork proxy).
    work_fatigue_enabled: bool = True
    work_fatigue_stress: float = 0.0055


@dataclass
class SocialLifeConfig:
    """Faith movements, romance, violence, and jail tied to enforcement."""

    enabled: bool = True
    love_pair_attempts_per_year: int = 160
    love_base_probability: float = 0.012
    assault_samples_per_year: int = 72
    assault_base_probability: float = 0.009
    jail_years_assault_caught: int = 2
    jail_years_theft: int = 1
    theft_jail_after_catches: int = 2
    theft_jail_min_enforcement: float = 0.42
    max_prophets_world: int = 6
    prophet_min_population: int = 48
    prophet_min_spiritual: float = 0.62
    prophet_min_knowledge: float = 0.26
    prophet_min_reputation: float = 0.36
    prophet_emerge_probability: float = 0.052
    worship_followers_for_shrine: int = 10
    jail_wealth_multiplier: float = 0.26


@dataclass
class ConflictConfig:
    preset: str = "balanced"  # balanced | high_conflict
    # Global violence dial: scales border tension, internal wars, and social friction.
    # 0.25 ≈ very peaceful, 1.0 default, 2.0+ very hostile world.
    world_aggression: float = 1.0


@dataclass
class PoliticsConfig:
    """How settlements govern themselves as they grow (camp → village → town → city)."""

    # auto: chiefdom at village, oligarchy vs democracy by town/city; country/empire bias below
    # Other values force that style once the settlement is large enough (town+).
    government_mode: str = "auto"  # auto | democracy | republic | oligarchy | chiefdom | monarchy | autocracy
    election_interval_years: int = 12  # democracy / republic
    elite_fraction: float = 0.12  # share of population with outsized influence
    leader_power_bonus: float = 0.03  # annual institutional boost while in office (capped in sim)
    # After level reaches city, polity can rise to a multi-community country then empire (same region).
    polity_progression: bool = True
    country_min_city_pop: int = 170
    country_min_civ_region: float = 0.44
    country_requires_state_milestone: bool = True
    empire_min_country_pop: int = 235
    empire_min_civ_region: float = 0.52
    # Empire more likely when ambition is high or iron exists (expansionist / coercive capacity).
    empire_ambition_threshold: float = 0.43


@dataclass
class CognitionConfig:
    """Global + per-agent cognition: goal choice and migration use a softmax 'brain'."""

    # Slider / dial: how sharp collective decisions are (higher → more rational, lower → noisier).
    world_iq: float = 0.65
    # 0 ≈ newborns cluster near average IQ; 1 ≈ full spread (matches legacy wide uniform).
    birth_iq_diversity: float = 1.0
    # Slow global "science stock" from civ + inventions + books; speeds learning and tool innovation.
    science_accumulation: bool = True
    # Adults' cognitive_iq drifts from nutrition, stress, instability, civ, science, and mild competition.
    dynamic_brain_feedback: bool = True
    # Regions below average food see slightly higher aggression (resource pressure).
    resource_competition_aggression: bool = True
    # Learned MLP over agent+macro features (NumPy REINFORCE + imitation); see learned_policy.py.
    learned_goal_network: bool = True
    # Blend: (1-mix)*heuristic_logits + mix*MLP_logits before softmax sample.
    learned_goal_mix: float = 0.22
    learned_goal_hidden: int = 24
    # Years of behavioral cloning toward heuristic before policy-gradient updates dominate.
    learned_goal_imitation_years: int = 22
    learned_goal_lr: float = 0.035
    learned_goal_lr_imitation: float = 0.07


@dataclass
class TechnologyConfig:
    """Resource gates for inventions and tools; dynamic era labels from tech + civ."""

    dynamic_eras: bool = True
    # Require regional ore/timber (etc.) floors before a breakthrough can fire.
    resource_gated_inventions: bool = True
    resource_gated_tool_crafting: bool = True
    # Multiplier on resource drain when inventions fire or tools are crafted.
    resource_drain_scale: float = 1.0


@dataclass
class EconomyConfig:
    """Currency, trade, theft, and local rules (treasury + policies per region)."""

    enabled: bool = True
    pairwise_trade_attempts: int = 2  # per person per year (capped by contacts)
    trade_goal_bias: float = 0.22
    theft_attempts: int = 1
    inter_region_trade_volume: float = 0.07


@dataclass
class DemographicsConfig:
    initial_population: int = 500
    region_count: int = 3
    max_age: int = 100
    reproductive_age_min: int = 18
    reproductive_age_max: int = 45
    base_birth_rate: float = 0.18
    partner_match_rate: float = 0.7
    child_mortality: float = 0.01
    natural_mortality_base: float = 0.005
    natural_mortality_age_factor: float = 0.00045
    fertility_trait_weight: float = 0.4


@dataclass
class SimulationConfig:
    years: int = 150
    random_seed: int = 42
    demographics: DemographicsConfig = field(default_factory=DemographicsConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    seasons: SeasonConfig = field(default_factory=SeasonConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    vaccination: VaccinationPolicyConfig = field(default_factory=VaccinationPolicyConfig)
    contact_network: ContactNetworkConfig = field(default_factory=ContactNetworkConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    social_life: SocialLifeConfig = field(default_factory=SocialLifeConfig)
    world_realism: WorldRealismConfig = field(default_factory=WorldRealismConfig)
    politics: PoliticsConfig = field(default_factory=PoliticsConfig)
    cognition: CognitionConfig = field(default_factory=CognitionConfig)
    technology: TechnologyConfig = field(default_factory=TechnologyConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    pathogens: list[PathogenConfig] = field(default_factory=lambda: [PathogenConfig()])

