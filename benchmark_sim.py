#!/usr/bin/env python3
"""Rough throughput benchmark: simulated years per second (CPU-bound)."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from population_sim.config import DemographicsConfig, SimulationConfig
from population_sim.simulation import SimulationEngine


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark SimulationEngine.step loop.")
    p.add_argument("--years", type=int, default=40, help="Simulated years to run")
    p.add_argument("--pop", type=int, default=220, help="Initial population")
    p.add_argument("--regions", type=int, default=3, help="Region count")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    cfg = SimulationConfig(
        years=args.years,
        random_seed=args.seed,
        demographics=replace(
            DemographicsConfig(),
            initial_population=max(20, args.pop),
            region_count=max(1, args.regions),
        ),
    )
    t0 = time.perf_counter()
    eng = SimulationEngine(cfg)
    for y in range(cfg.years):
        eng.step(y)
    dt = time.perf_counter() - t0
    alive = len([p for p in eng.population if p.alive])
    yps = cfg.years / dt if dt > 0 else 0.0
    print(f"years={cfg.years} initial_pop={cfg.demographics.initial_population} regions={cfg.demographics.region_count}")
    print(f"wall_s={dt:.4f}  years_per_s={yps:.2f}  alive_end={alive}  era={eng.current_era}")


if __name__ == "__main__":
    main()
