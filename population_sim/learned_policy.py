"""Learned goal policy: small MLP over agent+macro features (NumPy only).

Trained online with (1) behavioral cloning toward heuristic logits, then (2) REINFORCE
from end-of-year outcomes. This is a real learned policy, not hand-written argmax rules.
"""

from __future__ import annotations

import numpy as np

from population_sim.agent_cognition import WorldGoalContext
from population_sim.models import Individual

# Normalized person + WorldGoalContext + bias
GOAL_POLICY_INPUT_DIM = 28
N_GOALS = 6


def build_goal_feature_vector(person: Individual, ctx: WorldGoalContext) -> np.ndarray:
    v = np.array(
        [
            person.health,
            person.stress,
            person.ambition,
            min(1.0, person.age / 80.0),
            person.happiness,
            person.knowledge,
            person.tool_skill,
            min(1.2, max(0.0, person.observed_food_ema)) / 1.2,
            np.clip(person.observed_food_trend + 0.35, 0.0, 0.7) / 0.7,
            min(1.0, person.wealth / 25.0),
            person.reputation,
            person.cognitive_iq,
            float(np.clip(ctx.civ_index, 0.0, 1.0)),
            float(np.clip(ctx.global_instability, 0.0, 1.0)),
            float(np.clip(ctx.food_inequality, 0.0, 1.0)),
            min(1.2, max(0.2, ctx.mean_food_ratio)) / 1.2,
            ctx.settlement_tier / 3.0,
            float(np.clip(ctx.regional_wealth_poor, 0.0, 1.0)),
            (float(np.clip(ctx.local_food_vs_world, -1.0, 1.0)) + 1.0) * 0.5,
            float(np.clip(ctx.resource_index_local, 0.0, 1.0)),
            float(np.clip(ctx.treasury_strength_local, 0.0, 1.0)),
            min(1.0, ctx.policy_tax_burden / 0.22),
            float(np.clip(ctx.policy_security, 0.0, 1.0)),
            float(np.clip(ctx.policy_institutional_openness, 0.0, 1.0)),
            ctx.region_trade_connected,
            float(np.clip(ctx.faction_local_power, 0.0, 1.0)),
            float(np.clip(ctx.wealth_spread_local, 0.0, 1.0)),
            1.0,
        ],
        dtype=np.float64,
    )
    assert v.shape[0] == GOAL_POLICY_INPUT_DIM
    return v


class LearnedGoalMLP:
    """Two-layer tanh MLP: state -> logits over 6 primary goals."""

    def __init__(self, seed: int, in_dim: int = GOAL_POLICY_INPUT_DIM, hidden: int = 24, n_actions: int = N_GOALS) -> None:
        rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        scale1 = np.sqrt(2.0 / (in_dim + hidden))
        self.W1 = rng.normal(0.0, scale1, (hidden, in_dim))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        scale2 = np.sqrt(2.0 / (hidden + n_actions))
        self.W2 = rng.normal(0.0, scale2, (n_actions, hidden))
        self.b2 = np.zeros(n_actions, dtype=np.float64)
        self.n_actions = n_actions

    def forward_logits(self, x: np.ndarray) -> np.ndarray:
        z = np.tanh(self.W1 @ x + self.b1)
        return self.W2 @ z + self.b2

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        m = float(np.max(logits))
        e = np.exp(logits - m)
        return e / (np.sum(e) + 1e-12)

    def _apply_backprop(
        self,
        x: np.ndarray,
        z: np.ndarray,
        grad_logits: np.ndarray,
        lr: float,
        *,
        ascent: bool,
    ) -> None:
        gn = float(np.linalg.norm(grad_logits))
        if gn > 3.0:
            grad_logits = grad_logits * (3.0 / gn)
        d_z = (self.W2.T @ grad_logits) * (1.0 - z * z)
        upd_w2 = np.outer(grad_logits, z)
        upd_w1 = np.outer(d_z, x)
        sign = 1.0 if ascent else -1.0
        self.W2 += sign * lr * upd_w2
        self.b2 += sign * lr * grad_logits
        self.W1 += sign * lr * upd_w1
        self.b1 += sign * lr * d_z

    def backward_imitation(self, x: np.ndarray, target_logits: np.ndarray, lr: float) -> None:
        """Minimize KL/CE toward teacher distribution (heuristic softmax)."""
        z = np.tanh(self.W1 @ x + self.b1)
        logits = self.W2 @ z + self.b2
        pi = self._softmax(logits)
        tgt = self._softmax(target_logits)
        grad_logits = pi - tgt
        self._apply_backprop(x, z, grad_logits, lr, ascent=False)

    def backward_reinforce(self, x: np.ndarray, action: int, reward: float, lr: float) -> None:
        """Policy-gradient step to increase expected return (baseline-free REINFORCE)."""
        z = np.tanh(self.W1 @ x + self.b1)
        logits = self.W2 @ z + self.b2
        pi = self._softmax(logits)
        e = np.zeros(self.n_actions, dtype=np.float64)
        e[int(action)] = 1.0
        grad_logits = (e - pi) * float(reward)
        self._apply_backprop(x, z, grad_logits, lr, ascent=True)


def blend_goal_logits(
    heuristic: np.ndarray,
    learned: np.ndarray,
    mix: float,
) -> np.ndarray:
    m = max(0.0, min(1.0, float(mix)))
    return (1.0 - m) * heuristic + m * learned
