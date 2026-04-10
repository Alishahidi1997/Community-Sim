from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from population_sim.stats import StatsTracker


def plot_stats(stats: StatsTracker, output_path: Path) -> None:
    rows = stats.to_rows()
    if not rows:
        return

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

