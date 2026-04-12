# Dynamic Human Civilization Simulation (Python + Pygame)

This project simulates a human population from early hunter-gatherer origins toward organized civilization, with realtime strategy-style visualization and event-driven world changes.

The model emphasizes **emergent outcomes**: pressure, memory, and resource geography accumulate over time, so wars, alliances, and trade arise from **world state**, not only one-off random rolls.

## How the system works: “brain” vs rules

### Brain-like (softmax) decisions

Implemented in `population_sim/agent_cognition.py` and wired from `simulation.py`:

- **Yearly primary goal**:  Each living person picks among six goals: `survive`, `prosper`, `status`, `trade`, `connect`, `accumulate`. Logits combine **personal state** (health, stress, ambition, age, happiness, knowledge, tools, remembered food, wealth, reputation) with **`WorldGoalContext`**: civilization level, global instability, food inequality, mean food across regions, local settlement tier (camp→city), and how poor the person is versus the regional median wealth. **Macro layer (same context):** local food vs world mean, **resource endowment vs other regions**, **public treasury per capita** (normalized), **local policies** (income tax, theft enforcement, institutional openness), whether the **region has an open border trade route**, **faction strength in that region** (coalition/country-like bloc), and **within-region wealth spread**.
- **Learned policy (optional but on by default)** :  `population_sim/learned_policy.py` defines a **small NumPy MLP** (two layers, softmax over the same six goals). Each year its logits are **blended** with the hand-designed heuristic logits (`learned_goal_mix`), then sampled with the same IQ temperature. The network is trained **online**: first **behavioral cloning** (cross-entropy toward the heuristic distribution) for `learned_goal_imitation_years`, then **REINFORCE** using a simple end-of-year reward from health, happiness, and wealth change (penalty if the agent died). That is a **real learned policy**, not the same thing as fixed if/then rules, though it is small, not deep RL at scale. Set `CognitionConfig.learned_goal_network = False` to disable and use only the heuristic teacher.
- **Migration destination** :  Each region gets a score from food, infection, resources, and goal weights, plus a **goal-weighted institution bonus** (treasury stability, low-tax attractiveness for accumulators, trade-route hubs for traders, openness for “connect”). The engine **samples** a destination with softmax temperature from **effective IQ** (world + personal), instead of always moving to the single best region.

### IQ-related settings

- **`cognitive_iq`** (per person, inherited): enters **effective cognition IQ** with **`world_iq`** to set softmax temperature for goals and migration.
- **`CognitionConfig.world_iq`** (global, “Brain IQ” slider in realtime): sharper or noisier collective decisions; also **scales social learning** (knowledge/tool copy rate) and **modulates border diplomacy** in `world_dynamics.py` (pressure, trade goodwill growth, war thresholds, war intensity).
- **`CognitionConfig.birth_iq_diversity`** (“IQ spread (birth)” slider): how wide new people’s `cognitive_iq` is at **initial spawn** and **birth**; existing agents are unchanged when you move the slider.

### Rule-based majority (explicit simulation logic)

Most of the world still runs on **equations, thresholds, and config**:

- Health, aging, nutrition, eras; disease transmission and outcomes; birth and death formulas; contact graphs; friendship/enmity **margin vs threshold** rules; invention unlocks; settlement upgrades (population/civ gates); war **casualty fractions** after a war event fires; taxes and treasury arithmetic.

**Randomness** appears in many places (disease checks, some culture events, theft/trade attempts in the economy layer), but that is usually **event sampling from tuned probabilities**, not the same unified logits+softmax “brain” used for goals and migration.

### Economy, currency, and local rules

`population_sim/economy.py` and engine state add **wealth**, **reputation**, **regional treasuries**, **per-region policies** (tax rate, theft enforcement, public “gossip” openness), **pairwise trade** along contacts, **theft** with catch rules, **inter-regional treasury flows**, and **town/city broadcasts** (`last_edict` on settlements) that nudge knowledge and trust. These are **mechanical + stochastic**, not softmax policies.

### Where to read the code

| Piece | File(s) |
|--------|---------|
| Goals, softmax, `WorldGoalContext`, heuristic logits, migration sample | `agent_cognition.py` |
| Learned goal MLP (imitation + REINFORCE) | `learned_policy.py` |
| Economy helpers | `economy.py` |
| Year loop, macro cache for goals/migration, policies, city comms | `simulation.py` |
| Border tension, trade goodwill, `world_iq` diplomacy tuning | `world_dynamics.py` |
| Eras, invention/tool resource rules (pure helpers) | `tech_society.py` |
| Sanitation transmission helper | `world_realism.py` |
| Person fields, inheritance | `models.py` |
| Config (cognition, economy, technology, …) | `config.py` |

## Video Demo

![Demo](Demo.mp4) 

## Demo

![Demo](Demo.gif)

## What The Simulation Includes

### Population and Biology

- Individuals with:
  - age, gender, region
  - **`living_context`** ,  `rural` / `village` / `town` / `city` within a region (mixed populations; migration and yearly drift update bands)
  - **`nutrition_ema`** ,  slow smoothed diet adequacy vs regional food; chronic stress and health when low; newborns inherit from the mother with noise
  - health, disease susceptibility
  - inherited genetic traits (`resilience`, `fertility`, `immunity`)
- Life cycle mechanics:
  - aging
  - reproduction (partner matching, fertility constraints)
  - death (natural causes, disease, disaster, war)
- Birth-based growth only (no random NPC spawning during runtime)

### Social, Emotional, and Political Traits

- Per-individual state:
  - `happiness`, `stress`, `aggression`
  - **`ambition`** (drives rivalry, migration appetite, and political weight)
  - `knowledge`, `tool_skill`, `spiritual_tendency`
  - `belief_group`, `faction`, `language`
  - **`political_power`** (compounds with skills, age, and office)
  - **`cognitive_iq`** (inherited; with `world_iq`, controls softmax “sharpness” for goals and migration)
  - **`wealth`**, **`reputation`** (economy and standing; affect goals and some events)
  - **`primary_goal`** (replanned yearly via logits + softmax: survive / prosper / status / trade / connect / accumulate)
- Communication through **contact networks**
- Relationship graph: **friendships** and **enmities** (formation is gated by trust/conflict margins modulated by **global mood**, not only independent dice)
- Emotions and relationships affect productivity, health trajectory, and fertility outcomes

### World Dynamics (Latent global state)

`population_sim/world_dynamics.py` keeps **slow-moving state** that couples regions and years (this is **not** the same as the per-agent softmax “brain,” but it drives geopolitical mood):

- **Global instability**, **collective stress**, **food inequality** (spread of food per capita across regions)
- **Per-border tension** and **trade goodwill** (inertia: pressure builds and releases)
- **Internal war charge** for society-wide scarcity conflicts
- **Belief-group alliance goodwill** (defensive pacts form after sustained calm between groups)

Outcomes are **unpredictable but not meaningless**: the same snapshot rarely guarantees war or peace next year; sustained pressure or a sharp tension spike does.

### Geography, Resources, and Migration

- **Seasons** ,  Each simulated year advances one phase in a **four-season cycle** (`SeasonConfig`: spring / summer / autumn / winter by default). Tunable multipliers adjust **regional food**, **wildlife/forage** (ecology bonus), **migration pressure**, and **disease transmission**. Disable with `seasons.enabled = False` or shift the calendar with `phase_offset`.
- **Multiple regions** (configurable `region_count`) act as a larger map; migration uses **scored** regions (food, infection, resources, goal weights) and **softmax sampling** over those scores when behavior is enabled, not only a single “best tile” rule
- Each region’s `Environment` carries **natural resource richness** (water, fertile land, timber, ore) and **territory size**, feeding into food and attractiveness
- **Carrying-capacity-style** food scaling avoids runaway population that would make the sim unusably slow in late years

### Geopolitics: Trade, Diplomacy, and War

- **Trade links** between adjacent regions; food can flow along trade routes (surplus toward deficit)
- **Border wars** and **trade ruptures** emerge from latent border tension and scarcity, not a single yearly coin flip
- **Belief-group alliances** and **resource wars** use accumulation/discharge mechanics where possible

### Civilization and Settlements

- Era progression: hunter-gatherer → agrarian → industrial → modern → information age
- **Per-region settlement** evolution: `camp → village → town → city`
- Agriculture fields and institutions (school, workshop, temple) can appear **per region**
- **City labels** can include **community** and **power style** (e.g. civic-representation, elite-council) and **resource score**
- Large settlements can **split** (secession / civil-war style events) with new regions and settlements

### Disease and Public Health

- Multiple pathogens in parallel
- Disease transmission through the contact graph (with optional GPU-assisted batch sampling; see below)
- Recovery, mortality, immunity loss; pathogen mutation
- Vaccination policy with adaptive responses

### Environment and Disasters

- Food supply with variability, stress, and environment shocks (`EnvironmentConfig`)
- **Regional harvest weather** ,  persistent per-region harvest multipliers layered on food (`WorldRealismConfig`, `region_harvest_factor`)
- **Regional disaster events** ,  rare multi-year food penalties (drought, flood, crop blight, etc.) with timeline entries (`regional_disaster_*` in config)
- World-event system and auto-adjust can still apply temporary global effects

### Events and Timeline

- Event system tied to real conditions and effects
- Timeline and major-event feeds for discoveries, governance, conflict, and disasters
- Realtime timeline panel shows recent major events

### Adaptive Governance (Auto-Adjusting Parameters)

Core parameters can self-adjust from state signals (population, health, infection, food, civilization index), including food policy, birth policy, vaccination, migration, and infection control.

### Technology, Resources, and Diplomacy

- **Dynamic eras** ,  Labels such as hunter-gatherer → agrarian → classical → industrial → modern → information age follow **civilization level**, **unlocked inventions** (husbandry, wheel, advanced tools, iron), and **milestones** (writing, country, etc.), with a light calendar bias. Set `TechnologyConfig.dynamic_eras = False` to restore the old **year-only** era table.
- **Resource-gated inventions** ,  Breakthroughs require sufficient **timber / ore / fertile land** in the inventor’s region and **draw down** those stocks when they fire (`TechnologyConfig.resource_gated_inventions`, `resource_drain_scale`).
- **Tool crafting** ,  New personal tools consume **ore and timber** when `resource_gated_tool_crafting` is enabled; depleted regions craft slowly until regrowth catches up (environments slowly **regenerate** richness each year).
- **Border diplomacy** ,  `WorldDynamics.step_border` factors in **regional resource scores**, **military/economic power asymmetry** (population × health × tools × food × wealth), and **material + food pressure**, so trade goodwill and war tension respond to scarcity and predation-like gaps, not only food per capita.
- **Country and empire** ,  Settlements that reach **city** can mature into a **country** (renamed *Republic* or *Kingdom* by local civ; democracy vs oligarchy bias) and then an **Empire** (monarchy/autocracy bias, leader title **Emperor**). Tuned via `PoliticsConfig` (`polity_progression`, population and civ thresholds, `country_requires_state_milestone`, `empire_ambition_threshold`).
- **Faith, love, violence, jail** ,  Under `SocialLifeConfig`: a **prophet** can emerge (high spirituality, temples or spiritual age), founding a `way_of_<id>` movement; **conversions** and **shrine** structures appear as followings grow. **Love bonds** form between trusted same-region contacts; **assaults** harm victims and may **jail** aggressors when enforcement catches them; repeat **theft** with strong enforcement can also **jail**. Incarceration blocks migration and social learning and cuts yearly wealth gain until the sentence ticks down at year end.
- **World realism layer** ,  `WorldRealismConfig`: **marriage** / **breakups**; **maternal mortality**; **schools**; **treasury corruption**; **civic unrest**; **sanitation** (with writing + urban mix); **elder support**. Also: **nutrition tracking** (EMA), **urban crowding** (disease), **dense housing** wealth pressure when food is tight, **partner jail stress**, **contact bereavement**, **mutual aid** under scarcity, **work fatigue** (ambition), plus **harvest weather** and **regional disasters** (see Environment and Disasters).

### Tests and Benchmark

```bash
pip install -r requirements-dev.txt
python -m pytest tests/          # includes test_world_realism, test_tech_society, test_realism_extras
python benchmark_sim.py --years 50 --pop 220
```

## Performance

- Contact graph and social metrics are optimized for large populations (adaptive contact counts, cached friendship/enemy degrees)
- Optional **GPU path** for disease transmission random sampling: set environment variable `POP_SIM_USE_GPU=1` and install **CuPy** matching your CUDA stack (e.g. `cupy-cuda12x`). If CuPy is missing or fails, the sim falls back to CPU.

## Realtime Strategy View

```bash
python realtime_view.py
```

The window opens **fitted to your primary monitor** (it will not be wider or taller than the usable desktop area). You can **resize** the window by dragging edges; the size stays **clamped** so it stays on-screen.

### Realtime UI Layout

- **Left panel**: timeline, city ledger, major events (mouse wheel scrolls the ledger when the cursor is over the panel)
- **Center world**: regions, terrain, structures, people, animals, optional social link samples, HUD strip (year, era, season, population, politics summary, etc.)
- **Right panel** (“Simulation controls”):
  - **Sliders** (each shows name and live numeric value): **Food supply**, **Birth rate**, **Infection rate**, **Disease mortality**, **Migration rate**, **World aggression**, **Brain IQ (world)**, **IQ spread (birth)**, **Learned goal mix**, **Vaccination coverage**, **Simulation speed** (frames per sim year; lower = faster years)
  - **Region colors** button (or **`R`**): toggles distinct colors per region and a **legend** on the map (settlement level, polity, government, `living_context` mix, food, treasury)
  - Footer **hints** for zoom, pan, and region map
- **Typography**: prefers **Segoe UI**, **Calibri**, **Tahoma**, or **Arial** when installed; otherwise **Consolas**

### Realtime Controls

| Input | Action |
|--------|--------|
| `SPACE` | Pause / resume simulation stepping |
| Drag slider | Change parameter (click track or drag handle) |
| `,` / `.` | Slower / faster sim speed (more / fewer frames per year) |
| `L` | Toggle sparse per-person debug labels on the map |
| `R` | Toggle **region color overlay** and on-map legend (same as the button) |
| `W` `A` `S` `D` | Pan the world |
| Mouse wheel | Over map: pan vertically; **Ctrl+wheel**: zoom toward cursor |
| Middle mouse drag | Pan |
| `+` / `-` (or numpad) | Zoom in / out |
| `0` | Reset zoom and center camera |
| `PgUp` / `PgDn` | Scroll city ledger (when cursor over left panel, wheel also scrolls) |
| `F11` | Fullscreen on / off |
| `ESC` | Exit fullscreen, or quit if windowed |

Hover or click people on the map for a **tooltip** (belief, health, goals, **Living** band, **Diet (long-run)** / `nutrition_ema`, etc.).

## Batch Simulation

```bash
python main.py
```

Output:

- decadal log in terminal
- major event summary in terminal
- `outputs/population_stats.csv`
- `outputs/population_trends.png`

## Sensitivity Sweep

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
- `population_sim/config.py`: configuration dataclasses (including cognition, economy, seasons)
- `population_sim/models.py`: individual model + inheritance helpers
- `population_sim/agent_cognition.py`: goals, `WorldGoalContext`, softmax sampling, migration sample helper
- `population_sim/economy.py`: taxes, trade/theft helpers, regional policy defaults
- `population_sim/environment.py`: regional resources and food dynamics
- `population_sim/world_dynamics.py`: latent global/border state (tension, trade goodwill, war charge; `world_iq` hooks)
- `population_sim/disease.py`: multi-disease transmission (optional GPU batch sampling)
- `population_sim/simulation.py`: core engine, geopolitics, social dynamics, settlements, economy step
- `population_sim/stats.py`: metrics, CSV export
- `population_sim/visualize.py`: batch chart rendering
- `population_sim/sweep.py`: sweep automation
- `population_sim/realtime.py`: Pygame renderer and UI

## Key Metrics Tracked

- population, births, deaths
- age and health averages
- disease S/I/R counts, vaccination
- genetic diversity
- region distribution
- civilization index
- knowledge / tool skill / emotional averages
- friendships / enmities

## Disclaimer

This project was built with AI assistance and manual engineering decisions: architecture, balancing, edge cases, and UX iteration.
