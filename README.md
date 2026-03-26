# Dynamic Human Population Simulation

This project simulates a human population over time with:

- Individual-level attributes (age, gender, region, health, traits, disease state)
- Biological processes (aging, reproduction, death)
- Multi-region environment constraints (food availability and shocks)
- Social contact network transmission (instead of homogeneous mixing)
- Multiple diseases with stochastic pathogen mutation
- Vaccination intervention policy engine
- Configurable parameters and scenario tuning
- Time-series statistics, region tracking, and chart-based visualization
- Parameter sensitivity sweeps over scenario grids

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the simulation:

```bash
python main.py
```

3. Output:
- Console summary of final population metrics
- Charts saved to `outputs/population_trends.png`
- CSV stats saved to `outputs/population_stats.csv`

Optional sensitivity sweep:

```bash
python run_sweep.py
```

This writes grid results to `outputs/sensitivity_sweep.csv`.

## Project Structure

- `main.py`: Run script and default scenario
- `population_sim/config.py`: All simulation parameters (single source of truth)
- `population_sim/models.py`: Individual model and genetics helpers
- `population_sim/environment.py`: Food/resource dynamics
- `population_sim/disease.py`: Disease spread and progression model
- `population_sim/stats.py`: Metrics tracking and export
- `population_sim/simulation.py`: Main simulation engine
- `population_sim/visualize.py`: Plotting utilities
- `population_sim/sweep.py`: Parameter-grid sensitivity runner
- `run_sweep.py`: Example sweep entrypoint

## Tuning Parameters

Edit `build_default_config()` in `main.py` (or import and create your own config) to change:

- Food supply and environmental variability
- Birth and death behavior
- Partner matching and fertility
- Migration rate and cross-region contact
- Contact-network structure
- Vaccination rollout strategy
- Disease infection/recovery/mortality/mutation dynamics
- Simulation horizon and random seed

## Extensibility Ideas

- Multiple diseases and mutation
- Migration and region-based populations
- Healthcare systems and interventions
- Economic layers (workforce, income, food markets)
- Explicit family trees and social networks
