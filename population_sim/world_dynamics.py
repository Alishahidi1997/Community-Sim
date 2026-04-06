from __future__ import annotations

import math
from dataclasses import dataclass, field


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class WorldDynamics:
    """Latent world state: evolves from material/social pressure, not one-off random rolls.

    Border tension and trade goodwill accumulate with inertia; outcomes emerge when
    trajectories cross thresholds (chaotic but deterministic given inputs + history).
    """

    global_instability: float = 0.25
    collective_stress: float = 0.35
    food_inequality: float = 0.0
    border_tension: dict[tuple[int, int], float] = field(default_factory=dict)
    border_tension_prev: dict[tuple[int, int], float] = field(default_factory=dict)
    border_trade_goodwill: dict[tuple[int, int], float] = field(default_factory=dict)
    last_border_war_year: dict[tuple[int, int], int] = field(default_factory=dict)

    # Coupled oscillator phase (cheap "climate" variability tied to era + population scale)
    _phase: float = 0.0
    # Society-wide conflict charge (belief-group wars), discharges when threshold crossed
    internal_war_charge: float = 0.0
    alliance_goodwill: dict[tuple[str, str], float] = field(default_factory=dict)

    def update_global(
        self,
        alive_count: int,
        avg_stress: float,
        food_ratios: list[float],
        civilization_index: float,
        world_aggression: float = 1.0,
    ) -> None:
        """Pull global mood from observable aggregates (smooth, no dice)."""
        self.collective_stress = 0.88 * self.collective_stress + 0.12 * _clamp(avg_stress, 0.0, 1.0)
        if len(food_ratios) >= 2:
            mean_f = sum(food_ratios) / len(food_ratios)
            var = sum((x - mean_f) ** 2 for x in food_ratios) / len(food_ratios)
            self.food_inequality = 0.85 * self.food_inequality + 0.15 * _clamp(var * 4.0, 0.0, 1.0)
        else:
            self.food_inequality *= 0.98

        # Instability rises with stress + inequality, falls with institutions/civ
        wa = _clamp(world_aggression, 0.15, 3.0)
        target_instability = (
            0.42 * self.collective_stress
            + 0.38 * self.food_inequality
            - 0.22 * _clamp(civilization_index, 0.0, 1.0)
            + 0.08 * min(1.0, alive_count / 2500.0)
        )
        target_instability *= 0.55 + 0.45 * wa
        self.global_instability = _clamp(0.82 * self.global_instability + 0.18 * target_instability, 0.0, 1.0)

        # Slow phase drift: quasi-periodic modulation (unpredictable shape, deterministic path)
        pop_scale = min(1.0, alive_count / 1800.0)
        self._phase += 0.11 + 0.06 * self.food_inequality + 0.04 * pop_scale
        self._phase %= 6.283185307179586

    def step_internal_war(self, war_pressure: float, year: int, world_aggression: float = 1.0) -> bool:
        """Accumulate cross-group tension; fire war when the system 'tips' (not a dice roll)."""
        wa = _clamp(world_aggression, 0.15, 3.0)
        self.internal_war_charge += max(0.0, war_pressure) * 0.038 * (0.55 + 0.45 * wa)
        self.internal_war_charge *= 0.988
        if year <= 70:
            return False
        threshold = _clamp(0.94 / (wa**0.4), 0.52, 1.02)
        if self.internal_war_charge > threshold:
            self.internal_war_charge = 0.28
            return True
        return False

    def step_belief_alliance(
        self,
        g1: str,
        g2: str,
        avg_agg1: float,
        avg_agg2: float,
        avg_stress1: float,
        avg_stress2: float,
    ) -> bool:
        """Build defensive pacts from sustained compatibility (no yearly coin flip)."""
        key = tuple(sorted((g1, g2)))
        gw = self.alliance_goodwill.get(key, 0.0)
        calm = (avg_agg1 + avg_agg2) < 0.72 and abs(avg_stress1 - avg_stress2) < 0.2
        if calm:
            gw += 0.045 * (1.0 - 0.5 * (avg_agg1 + avg_agg2)) * (1.0 - abs(avg_stress1 - avg_stress2))
        else:
            gw *= 0.96
        gw = _clamp(gw, 0.0, 1.15)
        self.alliance_goodwill[key] = gw
        if gw >= 1.0:
            self.alliance_goodwill[key] = 0.4
            return True
        return False

    def social_modifiers(self, world_aggression: float = 1.0) -> dict[str, float]:
        """Scale friendship/enmity dynamics from world mood (not independent RNG)."""
        wa = _clamp(world_aggression, 0.15, 3.0)
        g = self.global_instability
        s = self.collective_stress
        # High instability: trust bar rises (harder to bond), conflict bar falls (easier to feud)
        friend_trust_shift = (0.04 * g + 0.03 * s) * (0.75 + 0.25 * wa)
        enemy_threshold_shift = (-0.05 * g - 0.04 * s) * (0.6 + 0.4 * wa)
        friend_prob_scale = max(0.25, 1.0 - 0.45 * g - 0.12 * (wa - 1.0))
        enemy_prob_scale = 1.0 + 0.75 * g + 0.35 * s + 0.45 * max(0.0, wa - 1.0)
        return {
            "friend_trust_shift": friend_trust_shift,
            "enemy_threshold_shift": enemy_threshold_shift,
            "friend_prob_scale": friend_prob_scale,
            "enemy_prob_scale": enemy_prob_scale,
        }

    def border_phase_jitter(self, ra: int, rb: int) -> float:
        """Small bounded coupling term: varies with pair + global phase, no RNG."""
        u = (ra * 17 + rb * 31 + 13) % 997 / 997.0
        return 0.04 * (u - 0.5) + 0.03 * math.sin(self._phase + u * 4.2)

    def step_border(
        self,
        ra: int,
        rb: int,
        food_per_cap_a: float,
        food_per_cap_b: float,
        same_dominant_faction: bool,
        avg_aggression_pair: float,
        avg_ambition_pair: float,
        has_trade: bool,
        year: int,
        world_aggression: float = 1.0,
    ) -> tuple[str | None, float]:
        """Update latent tension/goodwill; return event type and war intensity in [0,1].

        Event is None most years; 'trade_open', 'war', or 'trade_break' when thresholds cross.
        """
        pair = (ra, rb) if ra < rb else (rb, ra)
        key = pair
        prev = self.border_tension.get(key, 0.35)
        self.border_tension_prev[key] = prev

        mean_food = (food_per_cap_a + food_per_cap_b) * 0.5
        scarcity = _clamp(1.15 - mean_food, 0.0, 1.0)
        gap = abs(food_per_cap_a - food_per_cap_b)
        inequality_edge = _clamp(gap * 1.4, 0.0, 1.0)

        wa = _clamp(world_aggression, 0.15, 3.0)
        pressure = (
            0.34 * scarcity
            + 0.28 * inequality_edge
            + 0.18 * avg_aggression_pair
            + 0.14 * avg_ambition_pair
            + (0.0 if same_dominant_faction else 0.12)
            + self.border_phase_jitter(ra, rb)
        )
        pressure *= 0.45 + 0.55 * wa
        if has_trade:
            pressure *= 0.62
            pressure -= 0.06

        # Inertial integration: smooth but can spike under sustained pressure
        new_tension = _clamp(0.88 * prev + 0.12 * pressure + 0.04 * (pressure - prev), 0.0, 1.0)
        # Recent war cooldown
        years_since_war = 999
        if key in self.last_border_war_year:
            years_since_war = year - self.last_border_war_year[key]
        if years_since_war < 8:
            new_tension *= 0.75 + 0.03 * float(years_since_war)

        self.border_tension[key] = new_tension
        d_tension = new_tension - prev

        # Trade goodwill accumulates when conditions favor cooperation
        dip = (
            (0.38 if same_dominant_faction else 0.0)
            + (1.0 - inequality_edge) * 0.22
            + (1.0 - avg_aggression_pair) * 0.22
            + (1.0 - scarcity) * 0.18
        )
        gw = self.border_trade_goodwill.get(key, 0.0)
        if not has_trade:
            gw = _clamp(gw + 0.045 * (dip - 0.55) + 0.02 * (1.0 - new_tension), 0.0, 1.2)
        else:
            gw = _clamp(0.92 * gw + 0.04 * dip, 0.0, 1.2)
        self.border_trade_goodwill[key] = gw

        event: str | None = None
        war_intensity = 0.0

        # War: release when tension is high OR sharp rise under scarcity (no p < roll)
        hi = _clamp(0.84 - 0.07 * max(0.0, wa - 1.0), 0.62, 0.9)
        lo = _clamp(0.62 - 0.06 * max(0.0, wa - 1.0), 0.48, 0.62)
        spike = _clamp(0.085 - 0.02 * max(0.0, wa - 1.0), 0.045, 0.1)
        sc_need = _clamp(0.38 - 0.06 * max(0.0, wa - 1.0), 0.22, 0.42)
        war_ready = new_tension > hi or (new_tension > lo and d_tension > spike and scarcity > sc_need)
        if war_ready and years_since_war >= 5:
            event = "war"
            war_intensity = _clamp(0.35 + 0.55 * new_tension + 0.15 * scarcity, 0.0, 1.0)
            self.border_tension[key] = _clamp(new_tension - 0.38 - 0.1 * war_intensity, 0.08, 0.95)
            self.last_border_war_year[key] = year
            self.border_trade_goodwill[key] = max(0.0, gw - 0.55)
        elif has_trade and new_tension > 0.78 and d_tension > 0.06:
            event = "trade_break"

        # Trade pact: goodwill crosses threshold (slow build)
        if event is None and not has_trade and gw >= 1.0:
            event = "trade_open"
            self.border_trade_goodwill[key] = 0.35

        return event, war_intensity
