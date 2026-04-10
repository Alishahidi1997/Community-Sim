"""Resource gates, era labels, and invention/tool pressure — pure helpers for tests and simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from population_sim.environment import Environment


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# Minimum regional resource richness (0–1) before a breakthrough can originate there.
INVENTION_RESOURCE_FLOORS: dict[str, dict[str, float]] = {
    "animal_husbandry": {"timber": 0.2, "fertile_land": 0.16},
    "wheel": {"timber": 0.26, "ore": 0.14},
    "advanced_tools": {"ore": 0.2, "timber": 0.18},
    "iron_working": {"ore": 0.32, "timber": 0.14},
}

# One-time draw on the inventor's region when a global invention fires.
INVENTION_RESOURCE_DRAIN: dict[str, dict[str, float]] = {
    "animal_husbandry": {"timber": 0.035, "fertile_land": 0.02},
    "wheel": {"timber": 0.045, "ore": 0.028},
    "advanced_tools": {"ore": 0.04, "timber": 0.032},
    "iron_working": {"ore": 0.055, "timber": 0.025},
}

TOOL_CRAFT_DRAIN = {"ore": 0.014, "timber": 0.018}
TOOL_CRAFT_MIN = {"ore": 0.11, "timber": 0.1}


def region_meets_invention_minimums(env: Environment, invention_key: str) -> bool:
    req = INVENTION_RESOURCE_FLOORS.get(invention_key)
    if not req:
        return True
    return all(env.resource_richness.get(k, 0.0) >= v for k, v in req.items())


def apply_invention_resource_drain(env: Environment, invention_key: str, scale: float = 1.0) -> None:
    drain = INVENTION_RESOURCE_DRAIN.get(invention_key)
    if not drain:
        return
    s = max(0.0, scale)
    for k, amt in drain.items():
        cur = env.resource_richness.get(k, 0.0)
        env.resource_richness[k] = max(0.08, cur - amt * s)


def apply_tool_craft_drain(env: Environment, scale: float = 1.0) -> None:
    s = max(0.0, scale)
    for k, amt in TOOL_CRAFT_DRAIN.items():
        cur = env.resource_richness.get(k, 0.0)
        env.resource_richness[k] = max(0.08, cur - amt * s)


def region_can_craft_tools(env: Environment) -> bool:
    return all(env.resource_richness.get(k, 0.0) >= v for k, v in TOOL_CRAFT_MIN.items())


def invention_roll_multiplier(
    cognitive_iq: float,
    knowledge: float,
    tool_skill: float,
    stress: float,
    *,
    invention_key: str,
    regional_food_ratio: float,
) -> float:
    """Scales annual invention probability; kept bounded for stability."""
    iq = _clamp(cognitive_iq, 0.05, 0.99)
    k = _clamp(knowledge, 0.0, 1.0)
    t = _clamp(tool_skill, 0.0, 1.0)
    st = _clamp(stress, 0.0, 1.0)
    fr = _clamp(regional_food_ratio, 0.2, 1.8)

    base = 0.52 + 0.55 * iq
    skill_edge = 0.92 + 0.22 * (0.5 * k + 0.5 * t)
    calm = 1.0 - 0.38 * st
    need = 1.0
    if invention_key == "animal_husbandry":
        need = 1.0 + 0.55 * max(0.0, 0.42 - fr)
    elif invention_key == "advanced_tools" or invention_key == "iron_working":
        need = 1.0 + 0.25 * max(0.0, 0.55 - t)
    return _clamp(base * skill_edge * calm * need, 0.15, 2.2)


def legacy_era_by_year(year: int) -> dict[str, float | str]:
    if year < 120:
        return {
            "name": "hunter-gatherer",
            "food_effect": 1.15,
            "birth_multiplier": 1.2,
            "mortality_multiplier": 1.1,
        }
    if year < 280:
        return {
            "name": "agrarian",
            "food_effect": 1.0,
            "birth_multiplier": 1.0,
            "mortality_multiplier": 1.0,
        }
    if year < 480:
        return {
            "name": "industrial",
            "food_effect": 0.9,
            "birth_multiplier": 1.15,
            "mortality_multiplier": 0.85,
        }
    if year < 760:
        return {
            "name": "modern",
            "food_effect": 0.85,
            "birth_multiplier": 0.92,
            "mortality_multiplier": 0.72,
        }
    return {
        "name": "information age",
        "food_effect": 0.85,
        "birth_multiplier": 0.84,
        "mortality_multiplier": 0.64,
    }


def compute_era_profile(
    year: int,
    civ_index: float,
    world_inventions: set[str],
    unlocked_milestones: set[str],
    agriculture_unlocked: bool,
    *,
    dynamic: bool = True,
    year_blend: float = 0.14,
) -> dict[str, float | str]:
    """Era name and demographic modifiers from tech + institutions, softly biased by calendar year."""
    if not dynamic:
        return legacy_era_by_year(year)

    civ = _clamp(civ_index, 0.0, 1.0)
    tech_tier = 0.0
    if agriculture_unlocked or "animal_husbandry" in world_inventions:
        tech_tier += 0.16
    if "wheel" in world_inventions:
        tech_tier += 0.11
    if "advanced_tools" in world_inventions:
        tech_tier += 0.14
    if "iron_working" in world_inventions:
        tech_tier += 0.17
    if "writing" in unlocked_milestones:
        tech_tier += 0.09
    if "country" in unlocked_milestones:
        tech_tier += 0.14
    if "digital_age" in unlocked_milestones:
        tech_tier += 0.12
    tech_tier = min(1.0, tech_tier)

    soc = _clamp(0.52 * civ + 0.48 * tech_tier, 0.0, 1.0)
    yb = _clamp(year_blend, 0.0, 0.45)
    year_pull = min(1.0, year / 720.0)
    blend = _clamp((1.0 - yb) * soc + yb * year_pull, 0.0, 1.0)

    if blend < 0.2:
        return {
            "name": "hunter-gatherer",
            "food_effect": 1.12 + 0.04 * (1.0 - blend),
            "birth_multiplier": 1.18,
            "mortality_multiplier": 1.09,
        }
    if blend < 0.38:
        return {
            "name": "agrarian",
            "food_effect": 1.02 - 0.02 * (blend - 0.2) / 0.18,
            "birth_multiplier": 1.05,
            "mortality_multiplier": 0.98,
        }
    if blend < 0.55:
        return {
            "name": "classical",
            "food_effect": 1.0,
            "birth_multiplier": 1.02,
            "mortality_multiplier": 0.94,
        }
    if blend < 0.72:
        return {
            "name": "industrial",
            "food_effect": 0.93,
            "birth_multiplier": 1.1,
            "mortality_multiplier": 0.88,
        }
    if blend < 0.88:
        return {
            "name": "modern",
            "food_effect": 0.86,
            "birth_multiplier": 0.95,
            "mortality_multiplier": 0.75,
        }
    return {
        "name": "information age",
        "food_effect": 0.84,
        "birth_multiplier": 0.86,
        "mortality_multiplier": 0.66,
    }


def settlement_name_stem(display_name: str) -> str:
    """Strip polity suffixes so upgrades can rename consistently."""
    n = display_name.strip()
    for suffix in (" Empire", " Kingdom", " Republic", " Confederacy", " City"):
        if n.endswith(suffix):
            return n[: -len(suffix)].strip()
    return n


def format_settlement_polity_name(stem: str, polity: str, civ_avg: float) -> str:
    """Display name for settlement scale (level stays city; polity is macro scale)."""
    stem = stem.strip() or "Settlement"
    c = _clamp(civ_avg, 0.0, 1.0)
    if polity == "city":
        return f"{stem} City"
    if polity == "country":
        return f"{stem} Republic" if c > 0.52 else f"{stem} Kingdom"
    if polity == "empire":
        return f"{stem} Empire"
    return stem


def material_pressure(resource_score: float, food_ratio: float) -> float:
    """0 = comfortable materials + food; 1 = acute shortage / hunger."""
    rs = _clamp(resource_score, 0.0, 1.2)
    fr = _clamp(food_ratio, 0.25, 1.5)
    mat = max(0.0, 0.52 - rs) * 1.85
    food_stress = max(0.0, 0.95 - fr) * 1.1
    return _clamp(0.5 * mat + 0.5 * food_stress, 0.0, 1.0)
