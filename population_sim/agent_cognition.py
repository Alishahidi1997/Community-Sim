from __future__ import annotations

from enum import Enum


class PrimaryGoal(str, Enum):
    """Lightweight goal stack (one active goal); replanned yearly from state."""

    SURVIVE = "survive"
    PROSPER = "prosper"
    STATUS = "status"


def normalize_pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


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
    return (1.06, 1.0, 1.04)
