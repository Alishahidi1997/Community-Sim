# Dynamic Human Civilization Simulation (Python + Pygame)

This project simulates a human population from early hunter-gatherer origins toward organized civilization, with realtime strategy-style visualization and event-driven world changes.

## What The Simulation Includes

### Population and Biology

- Individuals with:
  - age
  - gender
  - region
  - health
  - disease susceptibility
  - inherited genetic traits (`resilience`, `fertility`, `immunity`)
- Life cycle mechanics:
  - aging
  - reproduction (partner matching, fertility constraints)
  - death (natural causes, disease, disaster, war)
- Birth-based growth only (no random NPC spawning during runtime)

### Social and Emotional Behavior

- Per-individual social/emotional state:
  - `happiness`
  - `stress`
  - `aggression`
  - `knowledge`
  - `tool_skill`
  - `spiritual_tendency`
  - `belief_group`
- Communication through contact networks
- Relationship graph:
  - friendships
  - enmities
- Emotions and relationships affect productivity, health trajectory, and fertility outcomes

### Disease and Public Health

- Multiple pathogens in parallel
- Disease transmission through contact graph (not homogeneous mixing)
- Recovery, mortality, immunity loss
- Pathogen mutation (infection/recovery/mortality rates drift stochastically)
- Vaccination policy layer with adaptive responses

### Environment and Resources

- Food supply with variability, stress, and random shocks
- Natural disasters with real multi-year consequences:
  - drought
  - flood
  - volcanic winter
- Resource pressure feeds back into mortality, fertility, policy adaptation, and migration

### Civilization and World Evolution

- Era progression:
  - hunter-gatherer
  - agrarian
  - industrial
  - modern
- Practice-driven world state transitions:
  - settlement evolution: `camp -> village -> town -> city`
  - agriculture fields appear when agriculture is adopted
  - institutions appear from population behavior:
    - school (knowledge-driven)
    - workshop (tool-skill-driven)
    - temple (belief/spiritual-driven)
- NPCs prefer to move/work near structures that match their profile

### Events and Timeline

- Event system with real conditions and real effects
- Historical summaries are generated from actual simulation transitions
- Includes discoveries, governance shifts, conflict, and disasters
- Realtime timeline panel keeps the latest major world events visible

### Adaptive Governance (Auto-Adjusting Parameters)

Core parameters can self-adjust each year based on state signals (population, health, infection, food, civilization index), including:

- food supply policy
- birth policy
- vaccination intensity
- migration openness
- infection control pressure

These updates are visible live in the UI and event feed.

## Realtime Strategy View

Run:

```bash
python realtime_view.py
```

### Realtime UI Layout

- **Left panel**: world timeline (recent major events)
- **Center world**: terrain, structures, NPCs, social links
- **Right panel**: live sliders for policy/parameter tuning

### Realtime Controls

- `SPACE` pause/resume
- Mouse drag sliders to tune live parameters
- `,` / `.` slower/faster simulation stepping
- `L` toggle labels
- `ESC` quit

## Batch Simulation

Run:

```bash
python main.py
```

Output:

- decadal log in terminal
- major event summary in terminal
- `outputs/population_stats.csv`
- `outputs/population_trends.png`

## Sensitivity Sweep

Run:

```bash
python run_sweep.py
```

Output:

- `outputs/sensitivity_sweep.csv`

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:

- `matplotlib`
- `pygame-ce`

## Project Structure

- `main.py`: batch simulation entrypoint + summary logging
- `realtime_view.py`: realtime visualization entrypoint
- `run_sweep.py`: parameter grid/sensitivity runner
- `population_sim/config.py`: all configuration dataclasses
- `population_sim/models.py`: individual model + inheritance/social helpers
- `population_sim/environment.py`: resource and environmental dynamics
- `population_sim/disease.py`: multi-disease transmission/progression/mutation
- `population_sim/simulation.py`: core simulation engine and event/governance logic
- `population_sim/stats.py`: metrics, snapshots, CSV export
- `population_sim/visualize.py`: batch chart rendering
- `population_sim/sweep.py`: sweep automation utilities
- `population_sim/realtime.py`: Pygame strategy-style renderer and UI

## Key Metrics Tracked

- population, births, deaths
- age and health averages
- susceptible / infected / recovered counts
- vaccination counts
- genetic diversity
- region distribution
- civilization index
- knowledge/tool skill averages
- emotional averages
- relationship counts (friendships/enmities)

## Notes

- The simulation uses stochastic, rule-based agents with per-agent state (not deep neural policy brains).
- Randomness affects outcomes, but key world transitions and timeline entries are condition-driven by actual state changes.

## Suggested Next Extensions

- Multi-settlement diplomacy and trade routes
- Family lineages and dynastic inheritance
- Occupation system (farmers, artisans, scholars, soldiers, priests)
- Explicit economy (production, storage, demand)
- Save/load simulation state and replay mode
