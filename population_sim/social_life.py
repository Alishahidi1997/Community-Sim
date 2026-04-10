"""Helpers for prophets, worship sites, love pairs, assault, and jail (used by SimulationEngine)."""

from __future__ import annotations

from typing import Any


def prophet_movement_id(person_id: int) -> str:
    return f"way_of_{person_id}"


def is_prophet_movement_belief(belief_group: str) -> bool:
    return belief_group.startswith("way_of_") and belief_group[7:].isdigit()


def count_followers_by_belief(alive: list[Any], belief: str) -> int:
    return sum(1 for p in alive if p.alive and p.belief_group == belief)


def followers_by_region(alive: list[Any], belief: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for p in alive:
        if not p.alive or p.belief_group != belief:
            continue
        rid = int(p.region_id)
        out[rid] = out.get(rid, 0) + 1
    return out


def has_worship_shrine(world_structures: list[dict[str, Any]], region_id: int, movement: str) -> bool:
    for s in world_structures:
        if s.get("kind") != "shrine":
            continue
        if int(s.get("region_id", -1)) != region_id:
            continue
        if str(s.get("movement", "")) == movement:
            return True
    return False
