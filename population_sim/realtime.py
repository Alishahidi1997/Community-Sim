from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from population_sim.models import DiseaseState, Gender
from population_sim.simulation import SimulationEngine


@dataclass
class VisualAgent:
    x: float
    y: float
    vx: float
    vy: float


class RealtimeVisualizer:
    def __init__(self, engine: SimulationEngine, width: int = 1280, height: int = 800) -> None:
        self.engine = engine
        self.width = width
        self.height = height
        self.year = 0
        self.running = True
        self.paused = False
        self.show_labels = False
        self.step_every_frames = 8
        self.frame_counter = 0
        self.rng = random.Random(engine.config.random_seed + 999)
        self.visual_state: dict[int, VisualAgent] = {}

        self.region_colors = [(70, 70, 70), (55, 55, 80), (80, 55, 55)]
        self.bg_color = (18, 18, 22)
        self.grid_color = (35, 35, 40)

    def run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Population Dynamics Realtime View")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 18)

        while self.running:
            dt = clock.tick(60) / 1000.0
            self._handle_events()
            if not self.paused:
                self._tick_simulation()
                self._move_agents(dt)
            self._draw(screen, font)
            pygame.display.flip()

        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_l:
                    self.show_labels = not self.show_labels
                elif event.key == pygame.K_UP:
                    self.engine.config.environment.base_food_per_capita += 0.05
                elif event.key == pygame.K_DOWN:
                    self.engine.config.environment.base_food_per_capita = max(
                        0.2, self.engine.config.environment.base_food_per_capita - 0.05
                    )
                elif event.key == pygame.K_RIGHT:
                    self.engine.config.demographics.base_birth_rate = min(
                        0.95, self.engine.config.demographics.base_birth_rate + 0.02
                    )
                elif event.key == pygame.K_LEFT:
                    self.engine.config.demographics.base_birth_rate = max(
                        0.01, self.engine.config.demographics.base_birth_rate - 0.02
                    )
                elif event.key == pygame.K_i and self.engine.config.pathogens:
                    self.engine.config.pathogens[0].infection_rate = min(
                        0.95, self.engine.config.pathogens[0].infection_rate + 0.02
                    )
                elif event.key == pygame.K_k and self.engine.config.pathogens:
                    self.engine.config.pathogens[0].infection_rate = max(
                        0.01, self.engine.config.pathogens[0].infection_rate - 0.02
                    )
                elif event.key == pygame.K_PERIOD:
                    self.step_every_frames = max(1, self.step_every_frames - 1)
                elif event.key == pygame.K_COMMA:
                    self.step_every_frames = min(30, self.step_every_frames + 1)

    def _tick_simulation(self) -> None:
        self.frame_counter += 1
        if self.frame_counter % self.step_every_frames != 0:
            return
        births, deaths, available_food = self.engine.step(self.year)
        self.engine.stats.record(self.year, self.engine.population, births, deaths, available_food)
        self.year += 1
        if self.year >= self.engine.config.years:
            self.paused = True

    def _region_rect(self, region_id: int) -> pygame.Rect:
        regions = max(1, self.engine.config.demographics.region_count)
        region_width = self.width / regions
        x0 = int(region_id * region_width + 10)
        return pygame.Rect(x0, 60, int(region_width - 20), self.height - 80)

    def _spawn_visual_agent(self, person_id: int, region_id: int) -> VisualAgent:
        rect = self._region_rect(region_id)
        x = self.rng.uniform(rect.left + 6, rect.right - 6)
        y = self.rng.uniform(rect.top + 6, rect.bottom - 6)
        angle = self.rng.uniform(0, math.tau)
        speed = self.rng.uniform(20, 45)
        return VisualAgent(x=x, y=y, vx=math.cos(angle) * speed, vy=math.sin(angle) * speed)

    def _move_agents(self, dt: float) -> None:
        alive_people = [p for p in self.engine.population if p.alive]
        alive_ids = {p.person_id for p in alive_people}

        for person in alive_people:
            if person.person_id not in self.visual_state:
                self.visual_state[person.person_id] = self._spawn_visual_agent(person.person_id, person.region_id)

        for person_id in list(self.visual_state.keys()):
            if person_id not in alive_ids:
                del self.visual_state[person_id]

        for person in alive_people:
            agent = self.visual_state[person.person_id]
            rect = self._region_rect(person.region_id)
            noise = 15.0
            agent.vx += self.rng.uniform(-noise, noise) * dt
            agent.vy += self.rng.uniform(-noise, noise) * dt
            speed = math.hypot(agent.vx, agent.vy)
            max_speed = 55.0
            if speed > max_speed:
                factor = max_speed / speed
                agent.vx *= factor
                agent.vy *= factor

            agent.x += agent.vx * dt
            agent.y += agent.vy * dt

            if agent.x < rect.left:
                agent.x = rect.left
                agent.vx *= -0.8
            elif agent.x > rect.right:
                agent.x = rect.right
                agent.vx *= -0.8

            if agent.y < rect.top:
                agent.y = rect.top
                agent.vy *= -0.8
            elif agent.y > rect.bottom:
                agent.y = rect.bottom
                agent.vy *= -0.8

    def _health_color(self, person) -> tuple[int, int, int]:
        any_infected = any(state == DiseaseState.INFECTED for state in person.disease_states.values())
        if any_infected:
            return (220, 55, 55)
        if person.health > 0.7:
            return (70, 200, 90)
        if person.health > 0.4:
            return (240, 180, 60)
        return (190, 95, 35)

    def _draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        screen.fill(self.bg_color)
        self._draw_regions(screen)

        alive_people = [p for p in self.engine.population if p.alive]
        sample_lines = 0
        max_lines = 500
        for person in alive_people:
            if sample_lines > max_lines:
                break
            src = self.visual_state.get(person.person_id)
            if src is None:
                continue
            for neighbor_id in self.engine.contact_graph.get(person.person_id, [])[:3]:
                dst = self.visual_state.get(neighbor_id)
                if dst is None:
                    continue
                pygame.draw.line(screen, (65, 65, 75), (int(src.x), int(src.y)), (int(dst.x), int(dst.y)), 1)
                sample_lines += 1
                if sample_lines > max_lines:
                    break

        for person in alive_people:
            agent = self.visual_state.get(person.person_id)
            if agent is None:
                continue
            radius = max(2, min(9, int(2 + person.age / 14)))
            fill_color = self._health_color(person)
            pygame.draw.circle(screen, fill_color, (int(agent.x), int(agent.y)), radius)

            # Gender marker via border color while keeping circle shape.
            border = (95, 165, 245) if person.gender == Gender.MALE else (230, 120, 210)
            pygame.draw.circle(screen, border, (int(agent.x), int(agent.y)), radius, 1)

            if self.show_labels and person.person_id % 20 == 0:
                label = f"{person.person_id}:{person.health:.2f}"
                text = font.render(label, True, (210, 210, 220))
                screen.blit(text, (int(agent.x) + radius + 2, int(agent.y) - radius - 2))

        self._draw_hud(screen, font, alive_people)

    def _draw_regions(self, screen: pygame.Surface) -> None:
        for region_id in range(self.engine.config.demographics.region_count):
            rect = self._region_rect(region_id)
            region_color = self.region_colors[region_id % len(self.region_colors)]
            pygame.draw.rect(screen, region_color, rect, 1)
            for x in range(rect.left, rect.right, 28):
                pygame.draw.line(screen, self.grid_color, (x, rect.top), (x, rect.bottom), 1)

    def _draw_hud(self, screen: pygame.Surface, font: pygame.font.Font, alive_people: list) -> None:
        pop = len(alive_people)
        infected = sum(
            1 for p in alive_people if any(state == DiseaseState.INFECTED for state in p.disease_states.values())
        )
        vaccinated = sum(1 for p in alive_people if p.vaccinated)
        avg_health = (sum(p.health for p in alive_people) / pop) if pop else 0.0
        pathogen_rate = self.engine.config.pathogens[0].infection_rate if self.engine.config.pathogens else 0.0
        lines = [
            f"Year: {self.year}/{self.engine.config.years}   Population: {pop}   Infected: {infected}   Vaccinated: {vaccinated}",
            f"Avg health: {avg_health:.2f}   Food: {self.engine.config.environment.base_food_per_capita:.2f}   Birth rate: {self.engine.config.demographics.base_birth_rate:.2f}   Infection rate: {pathogen_rate:.2f}",
            "Controls: SPACE pause | UP/DOWN food | LEFT/RIGHT birth | I/K infection | ,/. speed | L labels | ESC quit",
        ]
        y = 8
        for line in lines:
            text = font.render(line, True, (235, 235, 240))
            screen.blit(text, (12, y))
            y += 22

