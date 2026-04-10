from population_sim.world_realism import sanitation_transmission_multiplier


def test_sanitation_no_effect_without_writing() -> None:
    assert sanitation_transmission_multiplier(
        2.0,
        has_writing_milestone=False,
        min_mean_tier=1.0,
        max_reduction=0.2,
    ) == 1.0


def test_sanitation_reduces_when_threshold_met() -> None:
    m = sanitation_transmission_multiplier(
        2.0,
        has_writing_milestone=True,
        min_mean_tier=1.0,
        max_reduction=0.2,
    )
    assert m < 1.0
    assert m >= 0.78


def test_sanitation_low_tier_no_reduction() -> None:
    assert (
        sanitation_transmission_multiplier(
            0.5,
            has_writing_milestone=True,
            min_mean_tier=1.12,
            max_reduction=0.15,
        )
        == 1.0
    )
