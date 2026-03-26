from __future__ import annotations

from dataclasses import replace
import itertools
from pathlib import Path

from population_sim.config import EnvironmentConfig, PathogenConfig, SimulationConfig
from population_sim.simulation import SimulationEngine


def run_sensitivity_sweep(
    base_config: SimulationConfig,
    output_path: Path,
    food_levels: list[float],
    infection_rates: list[float],
    mortality_rates: list[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "food_per_capita,infection_rate,mortality_rate,final_population,avg_health,infected,vaccinated"
    ]

    for food, inf, mort in itertools.product(food_levels, infection_rates, mortality_rates):
        cfg = replace(base_config)
        cfg.environment = replace(base_config.environment, base_food_per_capita=food)
        cfg.pathogens = [
            replace(p, infection_rate=inf, mortality_rate=mort) if i == 0 else replace(p)
            for i, p in enumerate(base_config.pathogens)
        ]
        engine = SimulationEngine(cfg)
        stats = engine.run().to_rows()
        last = stats[-1] if stats else {}
        lines.append(
            ",".join(
                [
                    f"{food:.3f}",
                    f"{inf:.3f}",
                    f"{mort:.3f}",
                    str(int(last.get("population", 0))),
                    f"{float(last.get('avg_health', 0.0)):.4f}",
                    str(int(last.get("infected", 0))),
                    str(int(last.get("vaccinated", 0))),
                ]
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
