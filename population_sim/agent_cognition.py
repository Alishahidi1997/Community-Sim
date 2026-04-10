from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum

from population_sim.models import Individual


@dataclass
class WorldGoalContext:
    """Signals from world + place that reshape what an agent prioritizes.

    Personal + global mood (food memory, civ, instability) mix with **macro** signals:
    local vs world food, resource endowment, public treasury, tax/security/openness policy,
    border trade access, local faction strength, and within-region wealth spread.
    """

    civ_index: float = 0.0
    global_instability: float = 0.25
    food_inequality: float = 0.0
    mean_food_ratio: float = 1.0
    settlement_tier: int = 0  # 0 camp, 1 village, 2 town, 3 city
    regional_wealth_poor: float = 0.0  # 0–1 how poor vs regional median this person is
    # Local food supply vs world mean: about -1 (much worse) .. +1 (much better)
    local_food_vs_world: float = 0.0
    # This region's resource richness vs richest region (0–1)
    resource_index_local: float = 0.5
    # Normalized per-capita public treasury (fiscal capacity / stability proxy, 0–1)
    treasury_strength_local: float = 0.5
    # Regional policy levers (same units as engine.region_policies)
    policy_tax_burden: float = 0.06
    policy_security: float = 0.35
    policy_institutional_openness: float = 0.5
    # 1.0 if this region has at least one open trade pact with a neighbor
    region_trade_connected: float = 0.0
    # Share of people in this region who share this agent's faction (0–1)
    faction_local_power: float = 0.5
    # Within-region wealth inequality proxy (0–1, higher = more spread)
    wealth_spread_local: float = 0.25


class PrimaryGoal(str, Enum):
    """Yearly replanned objective; migration weights use goal_migration_multipliers."""

    SURVIVE = "survive"
    PROSPER = "prosper"
    STATUS = "status"
    TRADE = "trade"
    CONNECT = "connect"
    ACCUMULATE = "accumulate"


GOAL_VALUES: tuple[str, ...] = tuple(g.value for g in PrimaryGoal)


def normalize_pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def effective_cognition_iq(world_iq: float, cognitive_iq: float) -> float:
    w = max(0.12, min(1.0, float(world_iq)))
    c = max(0.08, min(0.98, float(cognitive_iq)))
    return max(0.1, min(1.0, w * (0.42 + 0.58 * c)))


def _softmax_sample_index(logits: list[float], temperature: float, rng: random.Random) -> int:
    if len(logits) == 1:
        return 0
    t = max(1e-6, float(temperature))
    if t < 0.02:
        return max(range(len(logits)), key=lambda i: logits[i])
    mx = max(logits)
    exps = [math.exp((x - mx) / t) for x in logits]
    total = sum(exps)
    r = rng.random() * total
    acc = 0.0
    for i, e in enumerate(exps):
        acc += e
        if r <= acc:
            return i
    return len(logits) - 1


def _goal_logits_fields(
    health: float,
    stress: float,
    ambition: float,
    age: int,
    happiness: float,
    knowledge: float,
    tool_skill: float,
    observed_food_ema: float,
    observed_food_trend: float,
    wealth: float,
    reputation: float,
    ctx: WorldGoalContext,
) -> tuple[float, float, float, float, float, float]:
    h = health
    st = stress
    amb = ambition
    hap = happiness
    food_ema = observed_food_ema
    trend = observed_food_trend
    age_f = min(1.0, age / 72.0)
    food_scarcity = max(0.0, 1.12 - food_ema)
    civ = max(0.0, min(1.0, ctx.civ_index))
    instab = max(0.0, min(1.0, ctx.global_instability))
    ineq = max(0.0, min(1.0, ctx.food_inequality))
    mean_f = max(0.2, min(1.5, ctx.mean_food_ratio))
    world_hunger = max(0.0, 1.05 - mean_f)
    tier = max(0, min(3, ctx.settlement_tier))
    rel_poor = max(0.0, min(1.0, ctx.regional_wealth_poor))
    w_idx = max(0.15, wealth)

    lf = max(-1.0, min(1.0, ctx.local_food_vs_world))
    ri = max(0.0, min(1.0, ctx.resource_index_local))
    ts_macro = max(0.0, min(1.0, ctx.treasury_strength_local))
    tax_n = max(0.0, min(1.0, ctx.policy_tax_burden / 0.22))
    sec = max(0.0, min(1.0, ctx.policy_security))
    opn = max(0.0, min(1.0, ctx.policy_institutional_openness))
    trc = max(0.0, min(1.0, ctx.region_trade_connected))
    fp = max(0.0, min(1.0, ctx.faction_local_power))
    wsp = max(0.0, min(1.0, ctx.wealth_spread_local))

    survive = (
        2.05 * (1.0 - h)
        + 1.32 * st
        + 0.82 * food_scarcity
        + 0.52 * max(0.0, -trend)
        + (0.42 if age > 58 else 0.0)
        + 0.55 * world_hunger
        + 0.45 * instab
        + 0.35 * rel_poor
        + 0.5 * max(0.0, -lf)
        + 0.24 * (1.0 - ts_macro)
        + 0.2 * wsp
        + 0.16 * tax_n
        + 0.12 * (1.0 - opn) * instab
    )
    prosper = (
        0.38
        + 0.68 * h * (1.0 - st)
        + 0.38 * max(0.0, trend)
        + 0.28 * food_ema
        + 0.18 * (hap - 0.5)
        + 0.22 * (1.0 - rel_poor)
        - 0.12 * instab * (1.0 - h)
        + 0.34 * max(0.0, lf)
        + 0.2 * ri
        + 0.14 * ts_macro
        + 0.1 * (1.0 - tax_n)
    )
    skill = 0.5 * (knowledge + tool_skill)
    status = (
        0.22
        + 1.22 * amb * (1.0 - 0.88 * age_f)
        + 0.34 * h
        - 0.38 * st
        + 0.16 * skill
        + 0.12 * reputation
        + 0.08 * tier
        + 0.44 * fp
        + 0.12 * ts_macro
        + 0.08 * trc
        + 0.06 * opn
    )
    trade = (
        0.15
        + 0.55 * civ
        + 0.62 * ineq
        + 0.28 * tier
        + 0.18 * skill
        - 0.15 * world_hunger
        + 0.12 * amb
        + 0.52 * trc
        + 0.18 * opn
        + 0.2 * max(0.0, -lf)
        + 0.12 * ri
    )
    connect = (
        0.1
        + 0.45 * tier
        + 0.22 * civ
        + 0.15 * (1.0 - st)
        + 0.1 * reputation
        - 0.08 * instab
        + 0.34 * opn
        + 0.2 * ts_macro
        + 0.14 * trc
        + 0.08 * fp
    )
    accumulate = (
        0.12
        + 0.95 * amb * (0.35 + 0.65 * civ)
        + 0.35 * rel_poor
        - 0.2 * math.log(w_idx)
        + 0.15 * tier
        - 0.12 * (hap - 0.55)
        + 0.32 * ri
        + 0.26 * (1.0 - tax_n)
        + 0.12 * max(0.0, -lf)
        - 0.09 * sec
    )
    return survive, prosper, status, trade, connect, accumulate


def heuristic_goal_logits(person: Individual, ctx: WorldGoalContext) -> list[float]:
    """Hand-designed scores for the six goals (teacher for imitation / blending)."""
    return list(
        _goal_logits_fields(
            person.health,
            person.stress,
            person.ambition,
            person.age,
            person.happiness,
            person.knowledge,
            person.tool_skill,
            person.observed_food_ema,
            person.observed_food_trend,
            person.wealth,
            person.reputation,
            ctx,
        )
    )


def brain_choose_primary_goal(
    person: Individual,
    world_iq: float,
    rng: random.Random,
    ctx: WorldGoalContext | None = None,
) -> str:
    c = ctx or WorldGoalContext()
    return brain_choose_primary_goal_fields(
        world_iq,
        person.cognitive_iq,
        health=person.health,
        stress=person.stress,
        ambition=person.ambition,
        age=person.age,
        happiness=person.happiness,
        knowledge=person.knowledge,
        tool_skill=person.tool_skill,
        observed_food_ema=person.observed_food_ema,
        observed_food_trend=person.observed_food_trend,
        wealth=person.wealth,
        reputation=person.reputation,
        ctx=c,
        rng=rng,
    )


def brain_choose_primary_goal_fields(
    world_iq: float,
    cognitive_iq: float,
    *,
    health: float,
    stress: float,
    ambition: float,
    age: int,
    happiness: float,
    knowledge: float,
    tool_skill: float,
    observed_food_ema: float,
    observed_food_trend: float,
    wealth: float,
    reputation: float,
    ctx: WorldGoalContext,
    rng: random.Random,
) -> str:
    eff = effective_cognition_iq(world_iq, cognitive_iq)
    logits = list(
        _goal_logits_fields(
            health,
            stress,
            ambition,
            age,
            happiness,
            knowledge,
            tool_skill,
            observed_food_ema,
            observed_food_trend,
            wealth,
            reputation,
            ctx,
        )
    )
    temp = max(0.07, 0.64 * math.exp(2.35 * (1.0 - eff)))
    idx = _softmax_sample_index(logits, temp, rng)
    return GOAL_VALUES[idx]


def brain_choose_migration_region(scores: list[float], world_iq: float, cognitive_iq: float, rng: random.Random) -> int:
    """Sample destination region index; lower effective IQ → flatter softmax → more exploratory moves."""
    if not scores:
        return 0
    eff = effective_cognition_iq(world_iq, cognitive_iq)
    temp = max(0.055, 0.48 * math.exp(2.05 * (1.0 - eff)))
    return _softmax_sample_index(scores, temp, rng)


def replan_primary_goal(health: float, stress: float, ambition: float, age: int) -> str:
    if health < 0.48 or stress > 0.74:
        return PrimaryGoal.SURVIVE.value
    if ambition > 0.58 and age < 52 and health > 0.52:
        return PrimaryGoal.STATUS.value
    return PrimaryGoal.PROSPER.value


def goal_migration_multipliers(goal: str) -> tuple[float, float, float]:
    """Scale (food, infection avoidance, resource) weights relative to baseline 1.0."""
    if goal == PrimaryGoal.SURVIVE.value:
        return (1.12, 1.22, 0.88)
    if goal == PrimaryGoal.STATUS.value:
        return (0.94, 0.9, 1.32)
    if goal == PrimaryGoal.TRADE.value:
        return (1.06, 1.02, 1.14)
    if goal == PrimaryGoal.CONNECT.value:
        return (0.98, 1.04, 1.06)
    if goal == PrimaryGoal.ACCUMULATE.value:
        return (1.02, 0.98, 1.26)
    return (1.06, 1.0, 1.04)
