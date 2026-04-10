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
class ConflictConfig:
    preset: str = "balanced"  # balanced | high_conflict
    # Global violence dial: scales border tension, internal wars, and social friction.
    # 0.25 ≈ very peaceful, 1.0 default, 2.0+ very hostile world.
    world_aggression: float = 1.0


@dataclass
class PoliticsConfig:
    """How settlements govern themselves as they grow (camp → village → town → city)."""

    # auto: chiefdom at village, oligarchy vs democracy by civ index at town/city
    # Other values force that style once the settlement is large enough (town+).
    government_mode: str = "auto"  # auto | democracy | republic | oligarchy | chiefdom | monarchy | autocracy
    election_interval_years: int = 12  # democracy / republic
    elite_fraction: float = 0.12  # share of population with outsized influence
    leader_power_bonus: float = 0.03  # annual institutional boost while in office (capped in sim)


@dataclass
class CognitionConfig:
    """Global + per-agent cognition: goal choice and migration use a softmax 'brain'."""

    # Slider / dial: how sharp collective decisions are (higher → more rational, lower → noisier).
    world_iq: float = 0.65
    # 0 ≈ newborns cluster near average IQ; 1 ≈ full spread (matches legacy wide uniform).
    birth_iq_diversity: float = 1.0
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
    politics: PoliticsConfig = field(default_factory=PoliticsConfig)
    cognition: CognitionConfig = field(default_factory=CognitionConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    pathogens: list[PathogenConfig] = field(default_factory=lambda: [PathogenConfig()])

