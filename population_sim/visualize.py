from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from population_sim.stats import StatsTracker


def _apply_chart_style() -> None:
    for _style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(_style)
            break
        except OSError:
            continue
    plt.rcParams.update(
        {
            "figure.facecolor": "#f4f2ef",
            "axes.facecolor": "#faf9f7",
            "axes.edgecolor": "#c8c4bc",
            "axes.labelcolor": "#2a2824",
            "text.color": "#2a2824",
            "xtick.color": "#3d3a36",
            "ytick.color": "#3d3a36",
            "grid.color": "#e0ddd6",
            "grid.linestyle": "-",
            "font.size": 10,
        }
    )


def plot_stats(stats: StatsTracker, output_path: Path) -> None:
    rows = stats.to_rows()
    if not rows:
        return

    _apply_chart_style()

    years = [r["year"] for r in rows]
    population = [r["population"] for r in rows]
    infected = [r["infected"] for r in rows]
    recovered = [r["recovered"] for r in rows]
    susceptible = [r["susceptible"] for r in rows]
    vaccinated = [r["vaccinated"] for r in rows]
    avg_health = [r["avg_health"] for r in rows]
    food_per_capita = [r["food_per_capita"] for r in rows]
    region_0 = [r["region_0"] for r in rows]
    region_1 = [r["region_1"] for r in rows]
    region_2 = [r["region_2"] for r in rows]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    axes[0, 0].plot(years, population, color="#1a3a5c", linewidth=1.8, antialiased=True)
    axes[0, 0].set_title("Population Over Time")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("People")

    axes[0, 1].plot(years, susceptible, label="Susceptible", color="#8c8c90", linewidth=1.4)
    axes[0, 1].plot(years, infected, label="Infected", color="#b03030", linewidth=1.5)
    axes[0, 1].plot(years, recovered, label="Recovered", color="#2d6a4f", linewidth=1.4)
    axes[0, 1].plot(years, vaccinated, label="Vaccinated", linestyle="--", color="#5c4d7a", linewidth=1.3)
    axes[0, 1].set_title("Disease States")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].legend()

    axes[1, 0].plot(years, avg_health, color="#2f6f4e", linewidth=1.6)
    axes[1, 0].set_title("Average Health")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(years, food_per_capita, color="#c45c26", linewidth=1.6)
    axes[1, 1].axhline(1.0, color="#9a958c", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Food Per Capita")
    axes[1, 1].set_xlabel("Year")

    axes[2, 0].plot(years, region_0, label="Region 0", color="#4a6fa5", linewidth=1.4)
    axes[2, 0].plot(years, region_1, label="Region 1", color="#6b8e4e", linewidth=1.4)
    axes[2, 0].plot(years, region_2, label="Region 2", color="#a67c52", linewidth=1.4)
    axes[2, 0].set_title("Regional Population")
    axes[2, 0].set_xlabel("Year")
    axes[2, 0].legend()

    axes[2, 1].plot(years, [r["genetic_diversity"] for r in rows], color="#6b4c7a", linewidth=1.5)
    axes[2, 1].set_title("Genetic Diversity")
    axes[2, 1].set_xlabel("Year")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_fig(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_society_and_civilization(stats: StatsTracker, output_path: Path) -> Path | None:
    rows = stats.to_rows()
    if not rows:
        return None
    _apply_chart_style()
    years = [r["year"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].plot(years, [r["civilization_index"] for r in rows], color="#3d5a80", linewidth=1.7)
    axes[0, 0].set_title("Civilization index (knowledge / tools blend)")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylim(0, 1.05)

    axes[0, 1].plot(years, [r["avg_knowledge"] for r in rows], label="Avg knowledge", color="#2e6f95", linewidth=1.5)
    axes[0, 1].plot(years, [r["avg_tool_skill"] for r in rows], label="Avg tool skill", color="#bc6c25", linewidth=1.5)
    axes[0, 1].set_title("Cognitive & craft averages")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1.05)

    axes[1, 0].plot(years, [r["avg_happiness"] for r in rows], label="Happiness", color="#2d6a4f", linewidth=1.5)
    axes[1, 0].plot(years, [r["avg_stress"] for r in rows], label="Stress", color="#9d3b3b", linewidth=1.5)
    axes[1, 0].set_title("Emotional averages")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1.05)

    axes[1, 1].plot(years, [r["avg_aggression"] for r in rows], color="#7b4f8d", linewidth=1.6)
    axes[1, 1].set_title("Average aggression")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylim(0, 1.05)

    fig.suptitle("Society & civilization", fontsize=12, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, output_path)


def plot_demographics_and_social_graph(stats: StatsTracker, output_path: Path) -> Path | None:
    rows = stats.to_rows()
    if not rows:
        return None
    _apply_chart_style()
    years = [r["year"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    axes[0, 0].plot(years, [r["births"] for r in rows], label="Births", color="#2a6f4f", linewidth=1.4, alpha=0.9)
    axes[0, 0].plot(years, [r["deaths"] for r in rows], label="Deaths", color="#8b3a3a", linewidth=1.4, alpha=0.9)
    axes[0, 0].set_title("Births and deaths per year")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].legend()

    axes[0, 1].plot(years, [r["avg_age"] for r in rows], color="#4a6582", linewidth=1.6)
    axes[0, 1].set_title("Average age of living population")
    axes[0, 1].set_xlabel("Year")

    axes[1, 0].plot(years, [r["friendships"] for r in rows], label="Friendship edges", color="#3d6b4f", linewidth=1.4)
    axes[1, 0].plot(years, [r["enmities"] for r in rows], label="Enmity edges", color="#a85c5c", linewidth=1.4)
    axes[1, 0].set_title("Social graph size (undirected edges)")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].legend()

    axb = axes[1, 1]
    axb.plot(years, [r["belief_groups"] for r in rows], label="Belief groups (count)", color="#4a6fa5", linewidth=1.5)
    axb.set_xlabel("Year")
    axb.set_ylabel("Distinct belief groups", color="#2a2824")
    axt = axb.twinx()
    axt.plot(years, [r["cults"] for r in rows], label="Cults", color="#c45c26", linewidth=1.4, linestyle="--")
    axt.set_ylabel("Cult movements", color="#8b4a2a")
    axes[1, 1].set_title("Religious / belief diversity")

    fig.suptitle("Demographics & social network", fontsize=12, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, output_path)


def plot_genetics_and_diversity(stats: StatsTracker, output_path: Path) -> Path | None:
    rows = stats.to_rows()
    if not rows:
        return None
    _apply_chart_style()
    years = [r["year"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    axes[0, 0].plot(years, [r["mean_resilience"] for r in rows], label="Resilience", color="#3d6e5c", linewidth=1.4)
    axes[0, 0].plot(years, [r["mean_fertility"] for r in rows], label="Fertility", color="#8b6b4a", linewidth=1.4)
    axes[0, 0].plot(years, [r["mean_immunity"] for r in rows], label="Immunity trait", color="#5a4d8c", linewidth=1.4)
    axes[0, 0].set_title("Mean genetic traits (composite)")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1.05)

    axes[0, 1].plot(years, [r["genetic_diversity"] for r in rows], color="#6b4c7a", linewidth=1.7)
    axes[0, 1].set_title("Genetic diversity (mean variance of traits)")
    axes[0, 1].set_xlabel("Year")

    axes[1, 0].fill_between(
        years,
        [r["susceptible"] for r in rows],
        alpha=0.25,
        color="#9a9a9e",
        label="Susceptible (area)",
    )
    axes[1, 0].plot(years, [r["infected"] for r in rows], color="#b03030", linewidth=1.3, label="Infected")
    axes[1, 0].plot(years, [r["recovered"] for r in rows], color="#2d6a4f", linewidth=1.3, label="Recovered")
    axes[1, 0].plot(years, [r["multi_infected"] for r in rows], color="#7b3fa0", linewidth=1.2, linestyle=":", label="Multi-infected")
    axes[1, 0].set_title("Disease counts (living)")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    net = [r["births"] - r["deaths"] for r in rows]
    colors = ["#2a6f4f" if n >= 0 else "#8b3a3a" for n in net]
    axes[1, 1].bar(years, net, color=colors, width=0.8, alpha=0.75, edgecolor="none")
    axes[1, 1].axhline(0, color="#6a6860", linewidth=0.8)
    axes[1, 1].set_title("Net population change (births − deaths)")
    axes[1, 1].set_xlabel("Year")

    fig.suptitle("Genetics & disease summary", fontsize=12, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, output_path)


def plot_regional_snapshot(stats: StatsTracker, output_path: Path) -> Path | None:
    """Uses tracked region_0 .. region_2 counts from stats (first three region indices)."""
    rows = stats.to_rows()
    if not rows:
        return None
    _apply_chart_style()
    years = [r["year"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(years, [r["region_0"] for r in rows], label="Region 0", color="#4a6fa5", linewidth=1.5)
    ax.plot(years, [r["region_1"] for r in rows], label="Region 1", color="#6b8e4e", linewidth=1.5)
    ax.plot(years, [r["region_2"] for r in rows], label="Region 2", color="#a67c52", linewidth=1.5)
    ax.set_title("Regional population (regions 0–2 as recorded in CSV)")
    ax.set_xlabel("Year")
    ax.set_ylabel("People")
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, output_path)


def plot_supplementary_charts(stats: StatsTracker, output_dir: Path) -> list[Path]:
    """Write additional PNG dashboards beside the main population_trends figure."""
    output_dir = Path(output_dir)
    written: list[Path] = []
    for fn, plotter in (
        ("society_civilization.png", plot_society_and_civilization),
        ("demographics_social.png", plot_demographics_and_social_graph),
        ("genetics_disease.png", plot_genetics_and_diversity),
        ("regional_populations_r012.png", plot_regional_snapshot),
    ):
        path = plotter(stats, output_dir / fn)
        if path is not None:
            written.append(path)
    return written

