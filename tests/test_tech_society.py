"""Unit tests for resource gates, era computation, and invention multipliers."""

from __future__ import annotations

import pytest

from population_sim.config import EnvironmentConfig
from population_sim.environment import Environment
from population_sim.social_life import is_prophet_movement_belief, prophet_movement_id
from population_sim.tech_society import (
    INVENTION_RESOURCE_FLOORS,
    apply_invention_resource_drain,
    apply_tool_craft_drain,
    compute_era_profile,
    format_settlement_polity_name,
    invention_roll_multiplier,
    legacy_era_by_year,
    material_pressure,
    region_can_craft_tools,
    region_meets_invention_minimums,
    settlement_name_stem,
)


@pytest.fixture
def rich_env() -> Environment:
    rng = __import__("random").Random(1)
    e = Environment(EnvironmentConfig(), rng)
    e.resource_richness = {"water": 0.9, "fertile_land": 0.85, "timber": 0.9, "ore": 0.85}
    return e


@pytest.fixture
def poor_env() -> Environment:
    rng = __import__("random").Random(2)
    e = Environment(EnvironmentConfig(), rng)
    e.resource_richness = {"water": 0.4, "fertile_land": 0.2, "timber": 0.12, "ore": 0.1}
    return e


def test_region_meets_invention_minimums(rich_env: Environment, poor_env: Environment) -> None:
    assert region_meets_invention_minimums(rich_env, "wheel")
    assert not region_meets_invention_minimums(poor_env, "wheel")
    for key in INVENTION_RESOURCE_FLOORS:
        assert region_meets_invention_minimums(rich_env, key)


def test_apply_invention_drain_clamps(poor_env: Environment) -> None:
    poor_env.resource_richness["ore"] = 0.34
    poor_env.resource_richness["timber"] = 0.2
    apply_invention_resource_drain(poor_env, "iron_working", scale=1.0)
    for v in poor_env.resource_richness.values():
        assert v >= 0.08


def test_tool_craft_gate(rich_env: Environment, poor_env: Environment) -> None:
    assert region_can_craft_tools(rich_env)
    assert not region_can_craft_tools(poor_env)


def test_apply_tool_drain(rich_env: Environment) -> None:
    ore0 = rich_env.resource_richness["ore"]
    apply_tool_craft_drain(rich_env, scale=1.0)
    assert rich_env.resource_richness["ore"] < ore0


def test_material_pressure() -> None:
    assert material_pressure(0.8, 1.0) < material_pressure(0.2, 0.5)


def test_invention_roll_multiplier_stress_reduces() -> None:
    low = invention_roll_multiplier(
        0.7, 0.5, 0.5, 0.1, invention_key="wheel", regional_food_ratio=1.0
    )
    high = invention_roll_multiplier(
        0.7, 0.5, 0.5, 0.95, invention_key="wheel", regional_food_ratio=1.0
    )
    assert low > high


def test_invention_roll_husbandry_food_need() -> None:
    lean = invention_roll_multiplier(
        0.65, 0.4, 0.35, 0.3, invention_key="animal_husbandry", regional_food_ratio=0.25
    )
    plenty = invention_roll_multiplier(
        0.65, 0.4, 0.35, 0.3, invention_key="animal_husbandry", regional_food_ratio=1.1
    )
    assert lean > plenty


def test_compute_era_dynamic_vs_legacy() -> None:
    inv = {"animal_husbandry", "wheel"}
    mil = {"writing"}
    dyn = compute_era_profile(10, 0.55, inv, mil, True, dynamic=True)
    leg = compute_era_profile(10, 0.55, inv, mil, True, dynamic=False)
    assert dyn["name"] != leg["name"] or dyn["food_effect"] != leg["food_effect"]


def test_prophet_movement_id_roundtrip() -> None:
    assert prophet_movement_id(42) == "way_of_42"
    assert is_prophet_movement_belief("way_of_42")
    assert not is_prophet_movement_belief("ancestor")
    assert not is_prophet_movement_belief("way_of_x")


def test_settlement_name_stem_roundtrip() -> None:
    assert settlement_name_stem("Aster City") == "Aster"
    assert settlement_name_stem("Varden Republic") == "Varden"
    assert settlement_name_stem("Sunhold Empire") == "Sunhold"


def test_format_settlement_polity_name() -> None:
    assert "City" in format_settlement_polity_name("Aster", "city", 0.5)
    assert format_settlement_polity_name("Aster", "country", 0.6).endswith("Republic")
    assert format_settlement_polity_name("Aster", "country", 0.4).endswith("Kingdom")
    assert format_settlement_polity_name("Aster", "empire", 0.5).endswith("Empire")


def test_legacy_era_by_year_monotonic_name() -> None:
    names = [str(legacy_era_by_year(y)["name"]) for y in (0, 150, 300, 500, 800)]
    assert names[0] == "hunter-gatherer"
    assert "information" in names[-1] or names[-1] == "information age"
