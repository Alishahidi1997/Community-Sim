from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from population_sim.models import Individual


def default_region_policy(settlement_tier: int) -> dict[str, float]:
    """Rules enforced locally: theft enforcement, income tax, openness of public communication."""
    base_theft = 0.22 + 0.12 * settlement_tier
    base_tax = 0.04 + 0.015 * settlement_tier
    gossip = 0.35 + 0.12 * settlement_tier
    return {
        "theft_enforcement": min(0.92, base_theft),
        "income_tax": min(0.14, base_tax),
        "public_gossip": min(0.95, gossip),
    }


def apply_income_taxes(
    alive: list[Any],
    region_median_wealth: dict[int, float],
    policies: dict[int, dict[str, float]],
    treasuries: dict[int, float],
) -> None:
    for p in alive:
        pol = policies.get(p.region_id, {})
        rate = float(pol.get("income_tax", 0.05))
        med = max(0.15, region_median_wealth.get(p.region_id, 0.5))
        base = max(0.0, p.wealth)
        liability = rate * min(base, med * 2.5) * 0.12
        paid = min(base * 0.35, liability)
        p.wealth = max(0.0, base - paid)
        treasuries[p.region_id] = treasuries.get(p.region_id, 0.0) + paid


def inter_region_trade(
    rng: random.Random,
    treasuries: dict[int, float],
    region_count: int,
    resource_scores: list[float],
    volume: float,
    civ_index: float,
) -> None:
    if region_count < 2 or volume <= 0:
        return
    scale = volume * (0.45 + 0.55 * max(0.0, min(1.0, civ_index)))
    for a in range(region_count):
        b = rng.randrange(region_count - 1)
        if b >= a:
            b += 1
        ra, rb = resource_scores[a], resource_scores[b]
        if ra + rb < 1e-6:
            continue
        flow = scale * abs(ra - rb) / (ra + rb + 0.2)
        flow *= rng.uniform(0.65, 1.35)
        if ra > rb:
            pay = min(treasuries.get(a, 0.0) * 0.2, flow * 0.5)
            treasuries[a] = treasuries.get(a, 0.0) - pay
            treasuries[b] = treasuries.get(b, 0.0) + pay
        else:
            pay = min(treasuries.get(b, 0.0) * 0.2, flow * 0.5)
            treasuries[b] = treasuries.get(b, 0.0) - pay
            treasuries[a] = treasuries.get(a, 0.0) + pay


def pairwise_market_trade(
    rng: random.Random,
    person: Any,
    partner: Any,
    pair_trust: dict[tuple[int, int], float],
    trade_goal_bonus: float,
) -> None:
    """Voluntary exchange: both may gain a little wealth + mood when trust and goals align."""
    from population_sim.agent_cognition import normalize_pair

    if person.wealth <= 0.02 and partner.wealth <= 0.02:
        return
    pair = normalize_pair(person.person_id, partner.person_id)
    trust = pair_trust.get(pair, 0.5)
    g1 = person.primary_goal
    g2 = partner.primary_goal
    tradeish = {"trade", "prosper", "accumulate"}
    align = (g1 in tradeish or g2 in tradeish) * trade_goal_bonus
    if rng.random() > 0.14 + 0.5 * trust + align:
        return
    pool = min(person.wealth, partner.wealth) * 0.06
    if pool < 0.02:
        return
    delta = pool * (0.45 + 0.15 * trust)
    person.wealth = max(0.0, person.wealth + delta * rng.uniform(0.85, 1.15))
    partner.wealth = max(0.0, partner.wealth + delta * rng.uniform(0.85, 1.15))
    person.happiness = min(1.0, person.happiness + 0.006 + 0.004 * trust)
    partner.happiness = min(1.0, partner.happiness + 0.006 + 0.004 * trust)


def theft_attempt(
    rng: random.Random,
    thief: Any,
    victim: Any,
    enforcement: float,
    pair_trust: dict[tuple[int, int], float],
    friendships: set[tuple[int, int]],
    enmities: set[tuple[int, int]],
) -> str | None:
    from population_sim.agent_cognition import normalize_pair

    if victim.wealth < 0.12:
        return None
    pair = normalize_pair(thief.person_id, victim.person_id)
    base = 0.05 + 0.55 * thief.aggression + 0.35 * max(0.0, 0.55 - thief.reputation)
    if thief.primary_goal == "survive":
        base += 0.08
    if thief.wealth < victim.wealth * 0.35:
        base += 0.06
    if pair in friendships:
        base *= 0.35
    if rng.random() > min(0.42, base):
        return None

    take = min(victim.wealth * rng.uniform(0.06, 0.14), victim.wealth * 0.5)
    catch = enforcement * (0.18 + 0.55 * victim.reputation + 0.12 * (1.0 - thief.aggression))
    catch *= 0.75 + 0.25 * rng.random()
    if rng.random() < catch:
        thief.stress = min(1.0, thief.stress + 0.12)
        thief.reputation = max(0.0, thief.reputation - 0.14)
        victim.stress = min(1.0, victim.stress + 0.05)
        fine = min(thief.wealth * 0.22, take * 1.8)
        thief.wealth = max(0.0, thief.wealth - fine)
        pair_trust[pair] = max(0.05, pair_trust.get(pair, 0.5) - 0.28)
        enmities.add(pair)
        friendships.discard(pair)
        return "theft_caught"
    victim.wealth = max(0.0, victim.wealth - take)
    thief.wealth += take * 0.92
    thief.stress = min(1.0, thief.stress + 0.04)
    victim.stress = min(1.0, victim.stress + 0.08)
    pair_trust[pair] = max(0.05, pair_trust.get(pair, 0.5) - 0.12)
    return "theft_success"


def region_wealth_medians(alive: list[Any]) -> dict[int, float]:
    by_r: dict[int, list[float]] = defaultdict(list)
    for p in alive:
        by_r[p.region_id].append(p.wealth)
    out: dict[int, float] = {}
    for rid, vals in by_r.items():
        vals.sort()
        mid = len(vals) // 2
        out[rid] = vals[mid] if vals else 0.0
    return out
