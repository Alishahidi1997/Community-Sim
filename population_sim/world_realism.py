"""Sanitation / public-health style modifiers (pure helpers for tests and engine)."""

from __future__ import annotations


def sanitation_transmission_multiplier(
    mean_settlement_tier_weighted: float,
    *,
    has_writing_milestone: bool,
    min_mean_tier: float,
    max_reduction: float,
    floor: float = 0.78,
) -> float:
    """Reduce disease transmission when literacy + urban settlement mix crosses a threshold."""
    if not has_writing_milestone or mean_settlement_tier_weighted < min_mean_tier:
        return 1.0
    reduc = max_reduction * min(1.0, max(0.0, (mean_settlement_tier_weighted - 1.0) / 2.0))
    return max(floor, 1.0 - reduc)
