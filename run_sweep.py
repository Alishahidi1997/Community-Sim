from pathlib import Path

from main import build_default_config
from population_sim.sweep import run_sensitivity_sweep


def main() -> None:
    config = build_default_config()
    output_path = Path("outputs") / "sensitivity_sweep.csv"
    run_sensitivity_sweep(
        base_config=config,
        output_path=output_path,
        food_levels=[0.8, 1.0, 1.2],
        infection_rates=[0.08, 0.12, 0.18],
        mortality_rates=[0.01, 0.02, 0.03],
    )
    print(f"Sweep results saved to: {output_path}")


if __name__ == "__main__":
    main()
