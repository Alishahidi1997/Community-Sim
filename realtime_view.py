from main import build_default_config
from population_sim.realtime import RealtimeVisualizer
from population_sim.simulation import SimulationEngine


def main() -> None:
    config = build_default_config()
    engine = SimulationEngine(config)
    viewer = RealtimeVisualizer(engine)
    viewer.run()


if __name__ == "__main__":
    main()
