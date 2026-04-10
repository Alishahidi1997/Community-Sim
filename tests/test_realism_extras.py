"""Regional harvest, nutrition EMA, and crowding multiplier wiring."""

from population_sim.config import SimulationConfig
from population_sim.simulation import SimulationEngine


def test_harvest_factors_stay_in_bounds_after_ticks() -> None:
    cfg = SimulationConfig(years=5)
    cfg.demographics.region_count = 2
    cfg.demographics.initial_population = 80
    wr = cfg.world_realism
    wr.harvest_weather_enabled = True
    wr.harvest_volatility = 0.12
    eng = SimulationEngine(cfg)
    for y in range(15):
        eng.step(y)
    lo, hi = wr.harvest_min_factor, wr.harvest_max_factor
    for rid in range(cfg.demographics.region_count):
        f = eng.region_harvest_factor.get(rid, 1.0)
        assert lo <= f <= hi


def test_nutrition_ema_updates_and_bounded() -> None:
    cfg = SimulationConfig(years=3)
    cfg.demographics.initial_population = 60
    cfg.world_realism.nutrition_tracking_enabled = True
    eng = SimulationEngine(cfg)
    for y in range(8):
        eng.step(y)
    alive = [p for p in eng.population if p.alive]
    assert alive
    for p in alive:
        assert 0.1 <= p.nutrition_ema <= 1.4


def test_disaster_food_mult_in_configured_range_when_active() -> None:
    cfg = SimulationConfig()
    cfg.demographics.initial_population = 100
    cfg.world_realism.regional_disaster_enabled = True
    cfg.world_realism.regional_disaster_probability = 1.0
    cfg.world_realism.regional_disaster_min_years = 2
    cfg.world_realism.regional_disaster_max_years = 2
    eng = SimulationEngine(cfg)
    eng._roll_new_regional_disasters(0)
    lo, hi = cfg.world_realism.regional_disaster_food_mult_low, cfg.world_realism.regional_disaster_food_mult_high
    for rid in range(cfg.demographics.region_count):
        if eng.region_disaster_years_left.get(rid, 0) > 0:
            m = eng.region_disaster_food_mult.get(rid, 1.0)
            assert min(lo, hi) <= m <= max(lo, hi)


def test_urban_crowding_boosts_transmission_scale() -> None:
    cfg = SimulationConfig()
    cfg.demographics.initial_population = 40
    eng = SimulationEngine(cfg)
    alive = [p for p in eng.population if p.alive]
    base = eng._urban_crowding_disease_mult(alive)
    for p in alive:
        p.living_context = "city"
    boosted = eng._urban_crowding_disease_mult(alive)
    assert boosted >= base
