from pathlib import Path

from population_sim.config import (
    BehaviorConfig,
    ContactNetworkConfig,
    ConflictConfig,
    DemographicsConfig,
    EnvironmentConfig,
    MigrationConfig,
    PathogenConfig,
    PoliticsConfig,
    SimulationConfig,
    VaccinationPolicyConfig,
)
from population_sim.simulation import SimulationEngine
from population_sim.visualize import plot_stats


def build_default_config() -> SimulationConfig:
    return SimulationConfig(
        years=1000,
        random_seed=7,
        demographics=DemographicsConfig(
            initial_population=90,
            region_count=6,
            max_age=110,
            reproductive_age_min=18,
            reproductive_age_max=46,
            base_birth_rate=0.42,
            partner_match_rate=0.95,
            child_mortality=0.002,
            natural_mortality_base=0.0015,
            natural_mortality_age_factor=0.00022,
            fertility_trait_weight=0.45,
        ),
        environment=EnvironmentConfig(
            base_food_per_capita=1.75,
            food_variability=0.05,
            environmental_stress=0.01,
            shock_probability=0.015,
            shock_severity=0.18,
        ),
        migration=MigrationConfig(
            enabled=True,
            migration_rate=0.02,
            cross_region_contact_rate=0.08,
        ),
        vaccination=VaccinationPolicyConfig(
            enabled=True,
            start_year=20,
            annual_coverage_fraction=0.08,
            min_age=12,
            effectiveness=0.6,
            max_per_year=300,
        ),
        contact_network=ContactNetworkConfig(
            avg_contacts=12,
            rewiring_rate=0.12,
        ),
        behavior=BehaviorConfig(
            enabled=True,
            migration_food_weight=0.72,
            migration_infection_weight=0.28,
            contact_avoid_infected_bias=0.68,
            stress_birth_penalty_weight=0.55,
        ),
        conflict=ConflictConfig(
            preset="balanced",
        ),
        politics=PoliticsConfig(
            government_mode="auto",
            election_interval_years=12,
            elite_fraction=0.12,
        ),
        pathogens=[
            PathogenConfig(
                name="flu",
                initial_infected_fraction=0.0,
                infection_rate=0.08,
                recovery_rate=0.2,
                mortality_rate=0.008,
                immunity_loss_rate=0.01,
                mutation_rate=0.02,
                mutation_strength=0.02,
            ),
            PathogenConfig(
                name="respiratory",
                initial_infected_fraction=0.0,
                infection_rate=0.06,
                recovery_rate=0.16,
                mortality_rate=0.01,
                immunity_loss_rate=0.008,
                mutation_rate=0.015,
                mutation_strength=0.015,
            ),
        ],
    )


def print_final_summary(stats_rows: list[dict]) -> None:
    if not stats_rows:
        print("No simulation output produced.")
        return
    last = stats_rows[-1]
    print("Simulation complete.")
    print(f"Years simulated: {int(last['year']) + 1}")
    print(f"Final population: {int(last['population'])}")
    print(f"Average age: {last['avg_age']:.2f}")
    print(f"Average health: {last['avg_health']:.2f}")
    print(
        "Disease states any-pathogen (S/I/R): "
        f"{int(last['susceptible'])}/{int(last['infected'])}/{int(last['recovered'])} "
        f"| multi-infected: {int(last['multi_infected'])}"
    )
    print(f"Vaccinated: {int(last['vaccinated'])}")
    print(
        f"Emotion averages (happiness/stress/aggression): "
        f"{last['avg_happiness']:.2f}/{last['avg_stress']:.2f}/{last['avg_aggression']:.2f}"
    )
    print(f"Relations (friendships/enmities): {int(last['friendships'])}/{int(last['enmities'])}")
    print(
        f"Regional population (r0/r1/r2): "
        f"{int(last['region_0'])}/{int(last['region_1'])}/{int(last['region_2'])}"
    )
    print(
        "Mean genetic traits (resilience/fertility/immunity): "
        f"{last['mean_resilience']:.2f}/{last['mean_fertility']:.2f}/{last['mean_immunity']:.2f}"
    )
    print(f"Genetic diversity index: {last['genetic_diversity']:.4f}")


def print_city_summary(engine: SimulationEngine) -> None:
    if not engine.city_summaries:
        print("Cities: none yet.")
        return
    print(f"Cities: {len(engine.city_summaries)}")
    for city in engine.city_summaries[:10]:
        print(
            f"- {city['name']} | pop={city['population']} | "
            f"community={city.get('community', city['culture'])} | religion={city['religion']} | "
            f"faction={city['faction']} | language={city['language']} | "
            f"power={city.get('power_style', '?')} | resources={city.get('resource_score', '?')}"
        )


def print_decadal_logs(stats_rows: list[dict], step: int = 25, max_year: int = 1000) -> None:
    if not stats_rows:
        return
    print(f"\nProgress log (every {step} years):")
    for row in stats_rows:
        year_number = int(row["year"]) + 1
        if year_number > max_year:
            break
        if year_number % step != 0:
            continue
        print(
            f"Year {year_number:>3}: "
            f"pop={int(row['population'])}, "
            f"health={row['avg_health']:.2f}, "
            f"S/I/R={int(row['susceptible'])}/{int(row['infected'])}/{int(row['recovered'])}, "
            f"vacc={int(row['vaccinated'])}, "
            f"food_pc={row['food_per_capita']:.2f}"
        )


def print_event_summary(engine: SimulationEngine, max_items: int = 60) -> None:
    if not engine.major_events:
        print("\nMajor world events: none recorded this run.")
        return
    print("\nMajor world events:")
    for event in engine.major_events[:max_items]:
        print(f"- Year {event['year']}: {event['title']} - {event['details']}")
    if len(engine.major_events) > max_items:
        print(f"- ... and {len(engine.major_events) - max_items} more events")


def main() -> None:
    config = build_default_config()
    engine = SimulationEngine(config)
    stats = engine.run()

    output_dir = Path("outputs")
    csv_path = output_dir / "population_stats.csv"
    chart_path = output_dir / "population_trends.png"

    stats.export_csv(csv_path)
    plot_stats(stats, chart_path)
    rows = stats.to_rows()
    print_decadal_logs(rows, step=25, max_year=config.years)
    print_event_summary(engine)
    print_city_summary(engine)
    print_final_summary(rows)
    print(f"CSV saved to: {csv_path}")
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    main()

