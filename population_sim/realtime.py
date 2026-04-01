from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import pygame

from population_sim.models import DiseaseState, Gender
from population_sim.simulation import SimulationEngine


@dataclass
class VisualAgent:
    x: float
    y: float
    vx: float
    vy: float


@dataclass
class UISlider:
    label: str
    min_value: float
    max_value: float
    getter: Callable[[], float]
    setter: Callable[[float], None]
    rect: pygame.Rect
    active: bool = False

    def current(self) -> float:
        return self.getter()

    def normalized(self) -> float:
        value = self.current()
        if self.max_value <= self.min_value:
            return 0.0
        return max(0.0, min(1.0, (value - self.min_value) / (self.max_value - self.min_value)))

    def set_from_x(self, x: int) -> None:
        t = (x - self.rect.left) / max(1, self.rect.width)
        t = max(0.0, min(1.0, t))
        value = self.min_value + t * (self.max_value - self.min_value)
        self.setter(value)


class RealtimeVisualizer:
    def __init__(self, engine: SimulationEngine, width: int = 2140, height: int = 1120) -> None:
        self.engine = engine
        self.width = width
        self.height = height
        self.left_panel_width = 360
        self.panel_width = 380
        self.year = 0
        self.running = True
        self.paused = False
        self.show_labels = False
        # At ~60 FPS, 600 frames is about 10 seconds per simulation year.
        self.step_every_frames = 600
        self.frame_counter = 0
        self.rng = random.Random(engine.config.random_seed + 999)
        self.visual_state: dict[int, VisualAgent] = {}
        self.sliders: list[UISlider] = []
        self.dragging_slider: UISlider | None = None

        self.region_colors = [
            (72, 96, 70),
            (76, 86, 108),
            (106, 82, 80),
            (118, 96, 68),
            (90, 112, 86),
            (92, 88, 122),
            (124, 88, 92),
            (84, 118, 120),
        ]
        self.bg_color = (140, 185, 235)
        self.grid_color = (90, 120, 90)
        self.ground_color = (62, 130, 72)
        self.panel_color = (25, 27, 32)
        self.panel_text_color = (235, 235, 238)
        self.recent_messages: list[tuple[str, int]] = []
        self.timeline_cache: list[str] = []
        self.last_major_event_count = 0
        self.city_scroll_offset = 0
        self.terrain_seed = random.Random(engine.config.random_seed + 2026)
        self.hills = self._build_hills()
        self._build_sliders()

    def run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Population Dynamics - 2D Human Agents")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 18)
        small_font = pygame.font.SysFont("consolas", 15)

        while self.running:
            dt = clock.tick(60) / 1000.0
            self._handle_events()
            if not self.paused:
                self._tick_simulation()
                self._move_agents(dt)
            self._draw(screen, font, small_font)
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
                elif event.key == pygame.K_PERIOD:
                    self.step_every_frames = max(1, self.step_every_frames - 60)
                elif event.key == pygame.K_COMMA:
                    self.step_every_frames = min(1800, self.step_every_frames + 60)
                elif event.key == pygame.K_PAGEUP:
                    self._scroll_city_ledger(-3)
                elif event.key == pygame.K_PAGEDOWN:
                    self._scroll_city_ledger(3)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for slider in self.sliders:
                    if slider.rect.collidepoint(event.pos):
                        slider.active = True
                        self.dragging_slider = slider
                        slider.set_from_x(event.pos[0])
                        break
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.dragging_slider is not None:
                    self.dragging_slider.active = False
                self.dragging_slider = None
            elif event.type == pygame.MOUSEMOTION and self.dragging_slider is not None:
                self.dragging_slider.set_from_x(event.pos[0])
            elif event.type == pygame.MOUSEWHEEL:
                self._scroll_city_ledger(-event.y)

    def _tick_simulation(self) -> None:
        self.frame_counter += 1
        if self.frame_counter % self.step_every_frames != 0:
            return
        births, deaths, available_food = self.engine.step(self.year)
        self.engine.stats.record(
            self.year,
            self.engine.population,
            births,
            deaths,
            available_food,
            friendships=len(self.engine.friendships),
            enmities=len(self.engine.enmities),
        )
        events = self.engine.last_step_events
        if events.get("births", 0) > 0:
            self._push_message(f"+{events['births']} births")
        if events.get("deaths", 0) > 0:
            self._push_message(f"-{events['deaths']} deaths")
        if events.get("new_infections", 0) > 0:
            self._push_message(f"+{events['new_infections']} new infections")
        for note in events.get("adjustments", [])[:2]:
            self._push_message(f"Policy: {note}")
        for story in events.get("stories", [])[-3:]:
            self._push_message(story)
        self._update_timeline_cache()
        self.year += 1
        if self.year >= self.engine.config.years:
            self.paused = True

    def _region_rect(self, region_id: int) -> pygame.Rect:
        regions = max(1, self.engine.config.demographics.region_count)
        world_left = self.left_panel_width
        world_width = self.width - self.panel_width - self.left_panel_width
        region_width = world_width / regions
        x0 = int(world_left + region_id * region_width + 10)
        return pygame.Rect(x0, 80, int(region_width - 20), self.height - 110)

    def _spawn_visual_agent(self, person_id: int, region_id: int) -> VisualAgent:
        del person_id
        rect = self._region_rect(region_id)
        x = self.rng.uniform(rect.left + 8, rect.right - 8)
        y = self.rng.uniform(rect.top + 8, rect.bottom - 8)
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
            max_speed = 62.0
            if speed > max_speed:
                factor = max_speed / speed
                agent.vx *= factor
                agent.vy *= factor

            target = self._preferred_structure_target(person)
            if target is not None:
                tx, ty = target
                steer_x = (tx - agent.x) * 0.45 * dt
                steer_y = (ty - agent.y) * 0.45 * dt
                agent.vx += steer_x
                agent.vy += steer_y

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

    def _draw(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font) -> None:
        screen.fill(self.bg_color)
        self._draw_regions(screen)
        self._draw_timeline_panel(screen, font, small_font)

        alive_people = [p for p in self.engine.population if p.alive]
        sample_lines = 0
        max_lines = 80
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
                dx = src.x - dst.x
                dy = src.y - dst.y
                if (dx * dx + dy * dy) > 140 * 140:
                    continue
                pygame.draw.line(screen, (56, 74, 92), (int(src.x), int(src.y)), (int(dst.x), int(dst.y)), 1)
                sample_lines += 1
                if sample_lines > max_lines:
                    break

        for person in alive_people:
            agent = self.visual_state.get(person.person_id)
            if agent is None:
                continue
            self._draw_human(screen, person, agent)

            if self.show_labels and person.person_id % 20 == 0:
                label = f"{person.person_id}:{person.health:.2f}"
                text = small_font.render(label, True, (16, 18, 22))
                screen.blit(text, (int(agent.x) + 8, int(agent.y) - 26))

        self._draw_hud(screen, font, alive_people)
        self._draw_messages(screen, small_font)
        self._draw_control_panel(screen, font, small_font)

    def _draw_regions(self, screen: pygame.Surface) -> None:
        world_rect = pygame.Rect(self.left_panel_width, 0, self.width - self.panel_width - self.left_panel_width, self.height)
        self._draw_sky_gradient(screen, world_rect)
        self._draw_sun_and_clouds(screen, world_rect)
        self._draw_hills(screen, world_rect)
        self._draw_river(screen, world_rect)
        ground_rect = pygame.Rect(self.left_panel_width, 70, self.width - self.panel_width - self.left_panel_width, self.height - 70)
        pygame.draw.rect(screen, self.ground_color, ground_rect)

        for region_id in range(self.engine.config.demographics.region_count):
            rect = self._region_rect(region_id)
            region_color = self.region_colors[region_id % len(self.region_colors)]
            pygame.draw.rect(screen, region_color, rect, 1)
            for x in range(rect.left, rect.right, 32):
                pygame.draw.line(screen, self.grid_color, (x, rect.bottom - 24), (x + 16, rect.bottom), 1)
        self._draw_world_structures(screen)
        self._draw_roads(screen)

    def _draw_hud(self, screen: pygame.Surface, font: pygame.font.Font, alive_people: list) -> None:
        pop = len(alive_people)
        infected = sum(
            1 for p in alive_people if any(state == DiseaseState.INFECTED for state in p.disease_states.values())
        )
        vaccinated = sum(1 for p in alive_people if p.vaccinated)
        avg_health = (sum(p.health for p in alive_people) / pop) if pop else 0.0
        avg_happiness = (sum(p.happiness for p in alive_people) / pop) if pop else 0.0
        avg_stress = (sum(p.stress for p in alive_people) / pop) if pop else 0.0
        pathogen_rate = self.engine.config.pathogens[0].infection_rate if self.engine.config.pathogens else 0.0
        lines = [
            f"Year: {self.year}/{self.engine.config.years}   Population: {pop}   Infected: {infected}   Vaccinated: {vaccinated}",
            f"Era: {self.engine.current_era}   CivIndex: {self.engine.civilization_index:.2f}   Cults: {self.engine.cult_count}",
            f"Emotion H/S: {avg_happiness:.2f}/{avg_stress:.2f}   Friends: {len(self.engine.friendships)}   Enemies: {len(self.engine.enmities)}   Alliances: {len(self.engine.alliances)}",
            f"Tools: {self.engine.total_tools_crafted}   Books: {self.engine.total_books_written}",
            f"Cities: {len(self.engine.city_summaries)} {self._city_line_summary()}",
            f"Politics: {self._politics_summary()}",
            f"Factions: {self._faction_summary_line()}   Conflict preset: {self.engine.config.conflict.preset}",
            f"Trade routes: {len(getattr(self.engine, 'region_trade_links', []))}   Regions: {self.engine.config.demographics.region_count}",
            f"Avg health: {avg_health:.2f}   Food: {self.engine.config.environment.base_food_per_capita:.2f}   Birth rate: {self.engine.config.demographics.base_birth_rate:.2f}   Infection rate: {pathogen_rate:.2f}",
            "Controls: SPACE pause | mouse drag sliders | ,/. speed | L labels | ESC quit",
        ]
        y = 8
        for line in lines:
            text = font.render(line, True, (20, 22, 30))
            screen.blit(text, (self.left_panel_width + 12, y))
            y += 22

    def _draw_messages(self, screen: pygame.Surface, small_font: pygame.font.Font) -> None:
        self.recent_messages = [(m, ttl - 1) for m, ttl in self.recent_messages if ttl > 1]
        y = 86
        for msg, _ in self.recent_messages[-6:]:
            t = small_font.render(msg, True, (18, 20, 28))
            screen.blit(t, (self.left_panel_width + 12, y))
            y += 18

    def _draw_human(self, screen: pygame.Surface, person, agent: VisualAgent) -> None:
        body_color = self._health_color(person)
        accent = (35, 90, 180) if person.gender == Gender.MALE else (180, 65, 150)
        scale = max(4, min(14, int(4 + person.age / 10)))
        x = int(agent.x)
        y = int(agent.y)

        head_r = max(2, scale // 3)
        head_y = y - scale
        pygame.draw.circle(screen, body_color, (x, head_y), head_r)

        torso_top = (x, head_y + head_r)
        torso_bottom = (x, y + scale)
        pygame.draw.line(screen, body_color, torso_top, torso_bottom, 2)

        pygame.draw.line(screen, body_color, (x - scale // 2, y - scale // 4), (x + scale // 2, y - scale // 4), 2)
        pygame.draw.line(screen, body_color, torso_bottom, (x - scale // 2, y + scale + scale // 2), 2)
        pygame.draw.line(screen, body_color, torso_bottom, (x + scale // 2, y + scale + scale // 2), 2)

        if person.gender == Gender.FEMALE:
            skirt = [(x, y + scale // 2), (x - scale // 2, y + scale + 2), (x + scale // 2, y + scale + 2)]
            pygame.draw.polygon(screen, accent, skirt, 1)
        else:
            pygame.draw.circle(screen, accent, (x, y - scale // 2), 2)

        # Strategy-game iconography for knowledge/tools/spiritual role.
        if person.tool_skill > 0.6:
            pygame.draw.rect(screen, (175, 175, 185), pygame.Rect(x - 1, y + scale + 3, 3, 5))
        if person.knowledge > 0.6:
            pygame.draw.circle(screen, (230, 230, 180), (x + scale // 2 + 2, y - scale), 2)
        if person.spiritual_tendency > 0.75:
            pygame.draw.circle(screen, (210, 150, 230), (x - scale // 2 - 2, y - scale), 2, 1)

        pol = self.engine.politics_by_region.get(person.region_id, {})
        if pol.get("leader_id") == person.person_id:
            pygame.draw.circle(screen, (218, 186, 52), (x, head_y), head_r + 4, 2)

    def _politics_summary(self) -> str:
        pol = self.engine.politics_by_region.get(0, {})
        gov = pol.get("government", "informal")
        lid = pol.get("leader_id")
        title = pol.get("leader_title") or ""
        mode = self.engine.config.politics.government_mode
        if lid is None:
            return f"{gov} (pref: {mode})"
        return f"{gov} — {title} #{lid} (pref: {mode})"

    def _build_sliders(self) -> None:
        left = self.width - self.panel_width + 24
        top = 110
        w = self.panel_width - 48
        h = 14
        gap = 56

        self.sliders = [
            UISlider(
                label="Food supply",
                min_value=0.2,
                max_value=2.0,
                getter=lambda: self.engine.config.environment.base_food_per_capita,
                setter=lambda v: setattr(self.engine.config.environment, "base_food_per_capita", v),
                rect=pygame.Rect(left, top + gap * 0, w, h),
            ),
            UISlider(
                label="Birth rate",
                min_value=0.01,
                max_value=0.9,
                getter=lambda: self.engine.config.demographics.base_birth_rate,
                setter=lambda v: setattr(self.engine.config.demographics, "base_birth_rate", v),
                rect=pygame.Rect(left, top + gap * 1, w, h),
            ),
            UISlider(
                label="Infection rate",
                min_value=0.01,
                max_value=0.9,
                getter=lambda: self.engine.config.pathogens[0].infection_rate if self.engine.config.pathogens else 0.01,
                setter=lambda v: self._set_pathogen_value("infection_rate", v),
                rect=pygame.Rect(left, top + gap * 2, w, h),
            ),
            UISlider(
                label="Disease mortality",
                min_value=0.001,
                max_value=0.3,
                getter=lambda: self.engine.config.pathogens[0].mortality_rate if self.engine.config.pathogens else 0.001,
                setter=lambda v: self._set_pathogen_value("mortality_rate", v),
                rect=pygame.Rect(left, top + gap * 3, w, h),
            ),
            UISlider(
                label="Migration rate",
                min_value=0.0,
                max_value=0.2,
                getter=lambda: self.engine.config.migration.migration_rate,
                setter=lambda v: setattr(self.engine.config.migration, "migration_rate", v),
                rect=pygame.Rect(left, top + gap * 4, w, h),
            ),
            UISlider(
                label="Vaccination coverage",
                min_value=0.0,
                max_value=0.5,
                getter=lambda: self.engine.config.vaccination.annual_coverage_fraction,
                setter=lambda v: setattr(self.engine.config.vaccination, "annual_coverage_fraction", v),
                rect=pygame.Rect(left, top + gap * 5, w, h),
            ),
            UISlider(
                label="Simulation speed",
                min_value=1.0,
                max_value=1800.0,
                getter=lambda: float(self.step_every_frames),
                setter=lambda v: setattr(self, "step_every_frames", max(1, min(1800, int(round(v))))),
                rect=pygame.Rect(left, top + gap * 6, w, h),
            ),
        ]

    def _set_pathogen_value(self, key: str, value: float) -> None:
        if not self.engine.config.pathogens:
            return
        setattr(self.engine.config.pathogens[0], key, value)

    def _draw_control_panel(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
    ) -> None:
        panel_rect = pygame.Rect(self.width - self.panel_width, 0, self.panel_width, self.height)
        pygame.draw.rect(screen, self.panel_color, panel_rect)
        pygame.draw.line(screen, (60, 62, 72), (panel_rect.left, 0), (panel_rect.left, self.height), 2)

        title = font.render("Realtime Controls", True, self.panel_text_color)
        screen.blit(title, (panel_rect.left + 20, 22))
        subtitle = small_font.render("Drag sliders to tune parameters live", True, (190, 190, 198))
        screen.blit(subtitle, (panel_rect.left + 20, 48))

        for slider in self.sliders:
            label = small_font.render(f"{slider.label}: {slider.current():.3f}", True, self.panel_text_color)
            screen.blit(label, (slider.rect.left, slider.rect.top - 22))

            pygame.draw.rect(screen, (80, 82, 92), slider.rect, border_radius=6)
            fill_width = int(slider.rect.width * slider.normalized())
            if fill_width > 0:
                fill_rect = pygame.Rect(slider.rect.left, slider.rect.top, fill_width, slider.rect.height)
                pygame.draw.rect(screen, (88, 160, 235), fill_rect, border_radius=6)

            handle_x = slider.rect.left + fill_width
            handle = pygame.Rect(handle_x - 6, slider.rect.top - 4, 12, slider.rect.height + 8)
            color = (245, 245, 250) if slider.active else (220, 220, 230)
            pygame.draw.rect(screen, color, handle, border_radius=4)

        tips = [
            "Icons: yellow=scholar, gray=toolmaker, purple=spiritual",
            "Lines show social interaction links.",
        ]
        y = self.height - 52
        for tip in tips:
            t = small_font.render(tip, True, (180, 182, 190))
            screen.blit(t, (panel_rect.left + 20, y))
            y += 18

    def _push_message(self, msg: str) -> None:
        self.recent_messages.append((msg, 240))

    def _build_hills(self) -> list[list[tuple[int, int]]]:
        world_width = self.width - self.panel_width - self.left_panel_width
        layers = []
        for layer_idx in range(3):
            points = []
            base_y = 230 + layer_idx * 55
            for x in range(0, world_width + 40, 40):
                y = base_y + self.terrain_seed.randint(-25, 25)
                points.append((self.left_panel_width + x, y))
            points.extend(
                [
                    (self.left_panel_width + world_width, self.height),
                    (self.left_panel_width, self.height),
                ]
            )
            layers.append(points)
        return layers

    def _draw_sky_gradient(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        cycle = (self.year % 240) / 240.0
        day_factor = max(0.2, 1.0 - abs(cycle - 0.5) * 1.7)
        top = (
            int(36 + 70 * day_factor),
            int(56 + 90 * day_factor),
            int(90 + 140 * day_factor),
        )
        bottom = (
            int(76 + 120 * day_factor),
            int(96 + 130 * day_factor),
            int(120 + 135 * day_factor),
        )
        height = max(1, rect.height)
        for i in range(height):
            t = i / height
            color = (
                int(top[0] + (bottom[0] - top[0]) * t),
                int(top[1] + (bottom[1] - top[1]) * t),
                int(top[2] + (bottom[2] - top[2]) * t),
            )
            pygame.draw.line(screen, color, (rect.left, i), (rect.right, i))

    def _draw_sun_and_clouds(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        cycle = (self.year % 240) / 240.0
        sun_x = int(rect.left + 80 + cycle * (rect.width - 160))
        sun_y = int(120 - max(0.0, 1.0 - abs(cycle - 0.5) * 2.0) * 55)
        pygame.draw.circle(screen, (255, 232, 140), (sun_x, sun_y), 34)
        pygame.draw.circle(screen, (255, 244, 190), (sun_x - 8, sun_y - 8), 10)
        cloud_color = (245, 250, 255)
        for cx, cy in [(160, 90), (320, 120), (520, 95)]:
            pygame.draw.circle(screen, cloud_color, (cx, cy), 18)
            pygame.draw.circle(screen, cloud_color, (cx + 20, cy - 4), 16)
            pygame.draw.circle(screen, cloud_color, (cx + 38, cy), 14)
        if cycle < 0.16 or cycle > 0.84:
            for sx in range(rect.left + 40, rect.right - 40, 90):
                pygame.draw.circle(screen, (240, 240, 255), (sx, 42 + (sx % 20)), 1)

    def _draw_hills(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        del rect
        layer_colors = [(78, 128, 92), (68, 112, 82), (60, 102, 72)]
        for color, points in zip(layer_colors, self.hills):
            pygame.draw.polygon(screen, color, points)

    def _draw_river(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        points = []
        y_base = rect.top + int(rect.height * 0.64)
        for x in range(rect.left, rect.right + 1, 20):
            wave = int(18 * math.sin((x + self.year * 2.2) * 0.01))
            points.append((x, y_base + wave))
        for idx in range(len(points) - 1):
            pygame.draw.line(screen, (76, 154, 220), points[idx], points[idx + 1], 10)
            pygame.draw.line(screen, (102, 178, 235), points[idx], points[idx + 1], 4)

    def _draw_roads(self, screen: pygame.Surface) -> None:
        structures = [s for s in self.engine.world_structures if s.get("kind") in ("settlement", "school", "workshop", "temple")]
        if not structures:
            return
        settlement = next((s for s in structures if s.get("kind") == "settlement"), None)
        if settlement is None:
            return
        sx, sy = self._structure_screen_pos(settlement)
        for s in structures:
            if s is settlement:
                continue
            x, y = self._structure_screen_pos(s)
            pygame.draw.line(screen, (128, 114, 86), (sx, sy), (x, y), 2)
            pygame.draw.line(screen, (158, 142, 108), (sx, sy), (x, y), 1)

    def _update_timeline_cache(self) -> None:
        if len(self.engine.timeline_events) == self.last_major_event_count:
            return
        self.timeline_cache = [
            f"Y{event['year']}: {event['title']}"
            for event in self.engine.timeline_events[-30:]
        ]
        self.last_major_event_count = len(self.engine.timeline_events)

    def _draw_timeline_panel(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
    ) -> None:
        panel_rect = pygame.Rect(0, 0, self.left_panel_width, self.height)
        pygame.draw.rect(screen, (20, 24, 30), panel_rect)
        pygame.draw.line(screen, (55, 60, 72), (panel_rect.right - 2, 0), (panel_rect.right - 2, self.height), 2)

        title = font.render("World Timeline", True, (235, 235, 240))
        screen.blit(title, (16, 16))
        subtitle = small_font.render("Major simulation events", True, (170, 175, 188))
        screen.blit(subtitle, (16, 42))

        city_header = small_font.render(
            f"Cities: {len(self.engine.city_summaries)}",
            True,
            (206, 212, 228),
        )
        screen.blit(city_header, (16, 64))
        y_city = 84
        for city in self.engine.city_summaries[:3]:
            line = f"{city['name']} ({city['culture']}/{city['religion']})"
            txt = small_font.render(line[:42], True, (176, 188, 220))
            screen.blit(txt, (16, y_city))
            y_city += 18

        if not self.timeline_cache:
            msg = small_font.render("No major events yet.", True, (150, 155, 170))
            screen.blit(msg, (16, 190))
        else:
            y = 190
            for item in self.timeline_cache[:12]:
                bullet = small_font.render(f"- {item}", True, (210, 214, 224))
                screen.blit(bullet, (16, y))
                y += 22

        # Dedicated scrollable city ledger area.
        ledger_top = self.height - 310
        ledger_rect = pygame.Rect(10, ledger_top, self.left_panel_width - 20, 292)
        pygame.draw.rect(screen, (24, 29, 36), ledger_rect, border_radius=6)
        pygame.draw.rect(screen, (54, 62, 76), ledger_rect, 1, border_radius=6)
        title = small_font.render("City Ledger (scroll: wheel / PgUp PgDn)", True, (220, 224, 236))
        screen.blit(title, (ledger_rect.left + 8, ledger_rect.top + 8))
        self._draw_city_ledger_rows(screen, small_font, ledger_rect)

    def _city_line_summary(self) -> str:
        if not self.engine.city_summaries:
            return ""
        preview = ", ".join(str(c["name"]) for c in self.engine.city_summaries[:3])
        return f"[{preview}]"

    def _faction_summary_line(self) -> str:
        if not self.engine.city_summaries:
            return "none"
        factions = sorted({str(c.get("faction", "unknown")) for c in self.engine.city_summaries})
        return ", ".join(factions[:4])

    def _scroll_city_ledger(self, delta: int) -> None:
        max_offset = max(0, len(self.engine.city_summaries) - 6)
        self.city_scroll_offset = max(0, min(max_offset, self.city_scroll_offset + delta))

    def _draw_city_ledger_rows(
        self,
        screen: pygame.Surface,
        small_font: pygame.font.Font,
        ledger_rect: pygame.Rect,
    ) -> None:
        cities = sorted(self.engine.city_summaries, key=lambda c: int(c.get("population", 0)), reverse=True)
        if not cities:
            txt = small_font.render("No cities founded yet.", True, (164, 170, 184))
            screen.blit(txt, (ledger_rect.left + 10, ledger_rect.top + 36))
            return

        start = self.city_scroll_offset
        rows = cities[start : start + 6]
        y = ledger_rect.top + 34
        for idx, city in enumerate(rows, start=start + 1):
            line = (
                f"{idx:>2}. {city['name']} | pop {city['population']} | "
                f"{city.get('community', city['culture'])} | {city['religion']} | "
                f"{city.get('faction','?')} | r={city.get('resource_score','?')}"
            )
            txt = small_font.render(line[:56], True, (202, 208, 223))
            screen.blit(txt, (ledger_rect.left + 8, y))
            y += 18
            gov = city.get("government", "?")
            lt = city.get("leader_title", "") or "Leader"
            lid = city.get("leader_id")
            power_style = city.get("power_style", "local-assembly")
            sub = f"     {gov}/{power_style}" + (f" — {lt} #{lid}" if lid is not None else "")
            sub_t = small_font.render(sub[:56], True, (160, 176, 198))
            screen.blit(sub_t, (ledger_rect.left + 8, y))
            y += 22

        footer = small_font.render(
            f"Showing {start + 1}-{start + len(rows)} of {len(cities)}",
            True,
            (150, 156, 172),
        )
        screen.blit(footer, (ledger_rect.left + 8, ledger_rect.bottom - 22))

    def _world_rect(self) -> pygame.Rect:
        return pygame.Rect(self.left_panel_width, 0, self.width - self.panel_width - self.left_panel_width, self.height)

    def _structure_screen_pos(self, structure: dict) -> tuple[int, int]:
        region_id = int(structure.get("region_id", 0))
        rect = self._region_rect(region_id)
        slot = float(structure.get("slot", 0.5))
        x = int(rect.left + slot * rect.width)
        kind = structure.get("kind")
        if kind == "field":
            y = int(rect.bottom - 14)
        elif kind == "settlement":
            y = int(rect.bottom - 30)
        elif kind in ("school", "workshop", "temple"):
            y = int(rect.bottom - 52)
        else:
            y = int(rect.bottom - 26)
        return x, y

    def _draw_world_structures(self, screen: pygame.Surface) -> None:
        for structure in self.engine.world_structures:
            kind = structure.get("kind")
            x, y = self._structure_screen_pos(structure)
            if kind == "field":
                pygame.draw.rect(screen, (160, 190, 90), pygame.Rect(x - 12, y - 4, 24, 8))
                pygame.draw.line(screen, (130, 160, 70), (x - 10, y), (x + 10, y), 1)
            elif kind == "settlement":
                level = structure.get("level", "camp")
                if level == "camp":
                    pygame.draw.polygon(screen, (156, 117, 82), [(x - 10, y + 3), (x, y - 10), (x + 10, y + 3)])
                elif level == "village":
                    pygame.draw.rect(screen, (150, 122, 95), pygame.Rect(x - 12, y - 12, 24, 14))
                elif level == "town":
                    pygame.draw.rect(screen, (142, 128, 120), pygame.Rect(x - 15, y - 16, 30, 18))
                    pygame.draw.rect(screen, (120, 108, 104), pygame.Rect(x - 8, y - 24, 16, 8))
                else:  # city
                    pygame.draw.rect(screen, (125, 125, 132), pygame.Rect(x - 16, y - 20, 32, 22))
                    pygame.draw.rect(screen, (112, 112, 120), pygame.Rect(x - 4, y - 34, 8, 14))
            elif kind == "school":
                pygame.draw.rect(screen, (204, 214, 235), pygame.Rect(x - 10, y - 14, 20, 12))
                pygame.draw.line(screen, (95, 105, 130), (x - 8, y - 3), (x + 8, y - 3), 2)
            elif kind == "workshop":
                pygame.draw.rect(screen, (168, 168, 176), pygame.Rect(x - 10, y - 14, 20, 12))
                pygame.draw.rect(screen, (110, 110, 116), pygame.Rect(x + 4, y - 20, 4, 8))
            elif kind == "temple":
                pygame.draw.polygon(screen, (184, 160, 205), [(x - 11, y - 2), (x, y - 16), (x + 11, y - 2)])
                pygame.draw.rect(screen, (168, 144, 190), pygame.Rect(x - 8, y - 2, 16, 10))

    def _preferred_structure_target(self, person) -> tuple[float, float] | None:
        best: tuple[float, float] | None = None
        best_score = -10.0
        for structure in self.engine.world_structures:
            if int(structure.get("region_id", 0)) != person.region_id:
                continue
            kind = structure.get("kind")
            score = 0.05
            if kind == "field":
                score = 0.3 + person.tool_skill * 0.25
            elif kind == "school":
                score = 0.2 + person.knowledge * 0.6
            elif kind == "workshop":
                score = 0.2 + person.tool_skill * 0.6
            elif kind == "temple":
                score = 0.15 + person.spiritual_tendency * 0.6
            elif kind == "settlement":
                score = 0.2
            if score > best_score:
                sx, sy = self._structure_screen_pos(structure)
                best = (float(sx), float(sy))
                best_score = score
        return best

