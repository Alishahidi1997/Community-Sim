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
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    vaccination: VaccinationPolicyConfig = field(default_factory=VaccinationPolicyConfig)
    contact_network: ContactNetworkConfig = field(default_factory=ContactNetworkConfig)
    pathogens: list[PathogenConfig] = field(default_factory=lambda: [PathogenConfig()])

