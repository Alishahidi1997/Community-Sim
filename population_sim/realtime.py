from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import Callable

import pygame

try:
    import pygame.gfxdraw as _gfxdraw
except Exception:  # pragma: no cover
    _gfxdraw = None

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
        self._pending_resize: tuple[int, int] | None = None
        self._pending_fullscreen_toggle = False
        self._fullscreen = False
        self._saved_windowed_size: tuple[int, int] = (width, height)
        self._layout_panels()
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
        self._cloud_offsets: list[tuple[float, float, float, int]] = []
        self._init_cloud_layout()

    def _init_cloud_layout(self) -> None:
        r = self.terrain_seed
        self._cloud_offsets = [
            (0.11 + r.random() * 0.04, 0.12 + r.random() * 0.06, 0.9 + r.random() * 0.35, 16 + r.randint(0, 8)),
            (0.28 + r.random() * 0.05, 0.14 + r.random() * 0.05, 0.85 + r.random() * 0.3, 14 + r.randint(0, 7)),
            (0.48 + r.random() * 0.06, 0.11 + r.random() * 0.05, 0.95 + r.random() * 0.4, 15 + r.randint(0, 9)),
            (0.68 + r.random() * 0.05, 0.15 + r.random() * 0.06, 0.8 + r.random() * 0.25, 13 + r.randint(0, 6)),
            (0.82 + r.random() * 0.04, 0.13 + r.random() * 0.05, 0.88 + r.random() * 0.3, 12 + r.randint(0, 5)),
        ]

    @staticmethod
    def _lerp_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, t))
        return (
            int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t),
        )

    @staticmethod
    def _mul_rgb(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        return (
            max(0, min(255, int(rgb[0] * factor))),
            max(0, min(255, int(rgb[1] * factor))),
            max(0, min(255, int(rgb[2] * factor))),
        )

    def _desktop_pixel_size(self) -> tuple[int, int]:
        """Primary monitor size in pixels (for clamping window and true fullscreen)."""
        try:
            pygame.display.init()
            sizes = pygame.display.get_desktop_sizes()
            if sizes:
                return int(sizes[0][0]), int(sizes[0][1])
        except (AttributeError, TypeError, IndexError, pygame.error):
            pass
        if sys.platform == "win32":
            try:
                import ctypes

                user32 = ctypes.windll.user32
                return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
            except Exception:
                pass
        return 1920, 1080

    def _clamp_to_desktop(self, w: int, h: int) -> tuple[int, int]:
        """Keep windowed mode inside the monitor (taskbar / frame margin)."""
        dw, dh = self._desktop_pixel_size()
        margin = 88
        max_w = max(640, dw - margin)
        max_h = max(480, dh - margin)
        return min(max(640, w), max_w), min(max(480, h), max_h)

    def _layout_panels(self) -> None:
        """Side panel widths scale with window size; center world column keeps a minimum."""
        w = max(400, self.width)
        min_left, min_right = 168, 188
        lw = max(min_left, min(int(w * 0.132), 400))
        rw = max(min_right, min(int(w * 0.166), 440))
        min_center = max(220, w // 5)
        while lw + rw + min_center > w and lw > min_left + 20:
            lw -= 12
        while lw + rw + min_center > w and rw > min_right + 20:
            rw -= 12
        if lw + rw + min_center > w:
            lw = max(min_left, w // 5)
            rw = max(min_right, w // 5)
        self.left_panel_width = lw
        self.panel_width = rw

    def _apply_window_size(self, w: int, h: int) -> None:
        w = max(640, w)
        h = max(480, h)
        if not self._fullscreen:
            w, h = self._clamp_to_desktop(w, h)
            self._saved_windowed_size = (w, h)
        self.width = w
        self.height = h
        self._layout_panels()
        self.hills = self._build_hills()
        self._build_sliders()
        self._clamp_agents_to_regions()

    def _clamp_agents_to_regions(self) -> None:
        for person in self.engine.population:
            if not person.alive:
                continue
            pid = person.person_id
            if pid not in self.visual_state:
                continue
            rect = self._region_rect(person.region_id)
            ag = self.visual_state[pid]
            pad = 6.0
            ag.x = max(float(rect.left + pad), min(float(rect.right - pad), ag.x))
            ag.y = max(float(rect.top + pad), min(float(rect.bottom - pad), ag.y))

    def run(self) -> None:
        pygame.init()
        win_flags = pygame.RESIZABLE
        # Default size was wider than many monitors; fit to screen on first open.
        self._apply_window_size(self.width, self.height)
        screen = pygame.display.set_mode((self.width, self.height), win_flags)
        pygame.display.set_caption("Population Dynamics — live world view")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 18)
        small_font = pygame.font.SysFont("consolas", 15)

        while self.running:
            dt = clock.tick(60) / 1000.0
            self._handle_events()
            if self._pending_fullscreen_toggle:
                self._pending_fullscreen_toggle = False
                if self._fullscreen:
                    self._fullscreen = False
                    w, h = self._saved_windowed_size
                    self._apply_window_size(w, h)
                    screen = pygame.display.set_mode((self.width, self.height), win_flags)
                else:
                    self._saved_windowed_size = (self.width, self.height)
                    dw, dh = self._desktop_pixel_size()
                    screen = pygame.display.set_mode((dw, dh), pygame.FULLSCREEN)
                    self._fullscreen = True
                    self.width = screen.get_width()
                    self.height = screen.get_height()
                    self._layout_panels()
                    self.hills = self._build_hills()
                    self._build_sliders()
                    self._clamp_agents_to_regions()
            if self._pending_resize is not None and not self._fullscreen:
                nw, nh = self._pending_resize
                self._pending_resize = None
                self._apply_window_size(nw, nh)
                screen = pygame.display.set_mode((self.width, self.height), win_flags)
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
                    if self._fullscreen:
                        self._pending_fullscreen_toggle = True
                    else:
                        self.running = False
                elif event.key == pygame.K_F11:
                    self._pending_fullscreen_toggle = True
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
            elif event.type == pygame.VIDEORESIZE and not self._fullscreen:
                nw = max(1, int(getattr(event, "w", event.size[0])))
                nh = max(1, int(getattr(event, "h", event.size[1])))
                self._pending_resize = self._clamp_to_desktop(nw, nh)

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

    def _hud_top(self) -> int:
        return max(64, min(88, self.height // 14))

    def _region_rect(self, region_id: int) -> pygame.Rect:
        regions = max(1, self.engine.config.demographics.region_count)
        world_left = self.left_panel_width
        world_width = self.width - self.panel_width - self.left_panel_width
        world_width = max(80, world_width)
        region_width = world_width / regions
        x0 = int(world_left + region_id * region_width + 10)
        top = self._hud_top()
        bottom_margin = max(28, min(48, self.height // 22))
        return pygame.Rect(x0, top, int(region_width - 20), max(60, self.height - top - bottom_margin))

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
            return (198, 72, 82)
        flesh_hi = (232, 205, 188)
        flesh_mid = (214, 178, 152)
        flesh_lo = (186, 142, 118)
        sick = (168, 128, 108)
        if person.health > 0.72:
            return self._lerp_rgb(flesh_mid, flesh_hi, (person.health - 0.72) / 0.28)
        if person.health > 0.45:
            return self._lerp_rgb(flesh_lo, flesh_mid, (person.health - 0.45) / 0.27)
        if person.health > 0.22:
            return self._lerp_rgb(sick, flesh_lo, (person.health - 0.22) / 0.23)
        return self._mul_rgb(sick, 0.92)

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
                pygame.draw.line(screen, (100, 112, 124), (int(src.x), int(src.y)), (int(dst.x), int(dst.y)), 1)
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
        ground_top = max(56, self._hud_top() - 8)
        ground_rect = pygame.Rect(
            self.left_panel_width,
            ground_top,
            self.width - self.panel_width - self.left_panel_width,
            max(40, self.height - ground_top),
        )
        self._draw_ground_gradient(screen, ground_rect)
        self._draw_river(screen, world_rect)
        self._draw_region_fields(screen)
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
            f"Factions: {self._faction_summary_line()}   Preset: {self.engine.config.conflict.preset}   World aggression: {self.engine._world_aggression():.2f}",
            f"Trade routes: {len(getattr(self.engine, 'region_trade_links', []))}   Regions: {self.engine.config.demographics.region_count}",
            f"Avg health: {avg_health:.2f}   Food: {self.engine.config.environment.base_food_per_capita:.2f}   Birth rate: {self.engine.config.demographics.base_birth_rate:.2f}   Infection rate: {pathogen_rate:.2f}",
            "Controls: SPACE pause | drag window edges to resize | sliders | ,/. speed | L | ESC",
        ]
        y = 8
        line_h = max(16, min(22, self.height // 48))
        for line in lines:
            text = font.render(line, True, (20, 22, 30))
            screen.blit(text, (self.left_panel_width + 12, y))
            y += line_h

    def _draw_messages(self, screen: pygame.Surface, small_font: pygame.font.Font) -> None:
        self.recent_messages = [(m, ttl - 1) for m, ttl in self.recent_messages if ttl > 1]
        y = self._hud_top() - 6
        for msg, _ in self.recent_messages[-6:]:
            t = small_font.render(msg, True, (18, 20, 28))
            screen.blit(t, (self.left_panel_width + 12, y))
            y += max(14, min(20, self.height // 55))

    def _draw_human(self, screen: pygame.Surface, person, agent: VisualAgent) -> None:
        body_color = self._health_color(person)
        shade = self._mul_rgb(body_color, 0.78)
        accent = (52, 98, 148) if person.gender == Gender.MALE else (142, 72, 118)
        accent_soft = self._lerp_rgb(accent, (240, 240, 242), 0.35)
        scale = max(4, min(14, int(4 + person.age / 10)))
        x = int(agent.x)
        y = int(agent.y)

        foot_y = y + scale + scale // 2 + 1
        pygame.draw.ellipse(screen, (26, 34, 30), pygame.Rect(x - scale, foot_y - 2, scale * 2, max(4, scale // 2)))

        head_r = max(3, scale // 3)
        head_y = y - scale
        hair = self._mul_rgb((72, 58, 48), 0.9 + min(0.2, person.age / 200.0))
        if _gfxdraw is not None:
            try:
                _gfxdraw.filled_circle(screen, x, head_y, head_r + 1, hair)
                _gfxdraw.filled_circle(screen, x, head_y, head_r, body_color)
                _gfxdraw.aacircle(screen, x, head_y, head_r, body_color)
            except (TypeError, pygame.error):
                pygame.draw.circle(screen, hair, (x, head_y - 1), head_r + 1)
                pygame.draw.circle(screen, body_color, (x, head_y), head_r)
        else:
            pygame.draw.circle(screen, hair, (x, head_y - 1), head_r + 1)
            pygame.draw.circle(screen, body_color, (x, head_y), head_r)

        torso_top = (x, head_y + head_r)
        torso_bottom = (x, y + scale)
        pygame.draw.line(screen, shade, (x + 1, torso_top[1]), (torso_bottom[0] + 1, torso_bottom[1]), max(2, scale // 4))
        pygame.draw.line(screen, body_color, torso_top, torso_bottom, max(2, scale // 4))

        arm_y = y - scale // 4
        pygame.draw.line(screen, shade, (x - scale // 2 + 1, arm_y + 1), (x + scale // 2 + 1, arm_y + 1), 2)
        pygame.draw.line(screen, body_color, (x - scale // 2, arm_y), (x + scale // 2, arm_y), 2)
        pygame.draw.line(screen, shade, (torso_bottom[0] + 1, torso_bottom[1]), (x - scale // 2 + 1, y + scale + scale // 2 + 1), 2)
        pygame.draw.line(screen, body_color, torso_bottom, (x - scale // 2, y + scale + scale // 2), 2)
        pygame.draw.line(screen, shade, (torso_bottom[0] + 1, torso_bottom[1]), (x + scale // 2 + 1, y + scale + scale // 2 + 1), 2)
        pygame.draw.line(screen, body_color, torso_bottom, (x + scale // 2, y + scale + scale // 2), 2)

        if person.gender == Gender.FEMALE:
            skirt = [(x, y + scale // 2), (x - scale // 2, y + scale + 2), (x + scale // 2, y + scale + 2)]
            pygame.draw.polygon(screen, accent_soft, skirt)
            pygame.draw.polygon(screen, accent, skirt, 1)
        else:
            pygame.draw.circle(screen, accent, (x, y - scale // 2), 3)
            pygame.draw.circle(screen, self._mul_rgb(accent, 1.15), (x - 1, y - scale // 2 - 1), 1)

        if person.tool_skill > 0.6:
            pygame.draw.rect(screen, (138, 140, 148), pygame.Rect(x - 2, y + scale + 2, 4, 6))
            pygame.draw.rect(screen, (168, 170, 178), pygame.Rect(x - 1, y + scale + 3, 2, 4))
        if person.knowledge > 0.6:
            pygame.draw.circle(screen, (228, 210, 140), (x + scale // 2 + 3, head_y - 1), 3)
        if person.spiritual_tendency > 0.75:
            pygame.draw.circle(screen, (188, 150, 210), (x - scale // 2 - 3, head_y), 3, 1)

        pol = self.engine.politics_by_region.get(person.region_id, {})
        if pol.get("leader_id") == person.person_id:
            pygame.draw.circle(screen, (218, 186, 52), (x, head_y), head_r + 5, 2)
            pygame.draw.circle(screen, (240, 220, 140), (x, head_y), head_r + 3, 1)

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
        margin_x = max(12, min(28, self.panel_width // 14))
        left = self.width - self.panel_width + margin_x
        top = max(96, min(120, int(self.height * 0.095)))
        inner_w = max(120, self.panel_width - 2 * margin_x)
        h = max(12, min(18, self.height // 70))
        n_sliders = 8
        footer = max(56, min(90, self.height // 12))
        usable = max(h * n_sliders + 8, self.height - top - footer)
        gap = max(40, min(60, usable // max(n_sliders, 1)))

        self.sliders = [
            UISlider(
                label="Food supply",
                min_value=0.2,
                max_value=2.0,
                getter=lambda: self.engine.config.environment.base_food_per_capita,
                setter=lambda v: setattr(self.engine.config.environment, "base_food_per_capita", v),
                rect=pygame.Rect(left, top + gap * 0, inner_w, h),
            ),
            UISlider(
                label="Birth rate",
                min_value=0.01,
                max_value=0.9,
                getter=lambda: self.engine.config.demographics.base_birth_rate,
                setter=lambda v: setattr(self.engine.config.demographics, "base_birth_rate", v),
                rect=pygame.Rect(left, top + gap * 1, inner_w, h),
            ),
            UISlider(
                label="Infection rate",
                min_value=0.01,
                max_value=0.9,
                getter=lambda: self.engine.config.pathogens[0].infection_rate if self.engine.config.pathogens else 0.01,
                setter=lambda v: self._set_pathogen_value("infection_rate", v),
                rect=pygame.Rect(left, top + gap * 2, inner_w, h),
            ),
            UISlider(
                label="Disease mortality",
                min_value=0.001,
                max_value=0.3,
                getter=lambda: self.engine.config.pathogens[0].mortality_rate if self.engine.config.pathogens else 0.001,
                setter=lambda v: self._set_pathogen_value("mortality_rate", v),
                rect=pygame.Rect(left, top + gap * 3, inner_w, h),
            ),
            UISlider(
                label="Migration rate",
                min_value=0.0,
                max_value=0.2,
                getter=lambda: self.engine.config.migration.migration_rate,
                setter=lambda v: setattr(self.engine.config.migration, "migration_rate", v),
                rect=pygame.Rect(left, top + gap * 4, inner_w, h),
            ),
            UISlider(
                label="World aggression",
                min_value=0.15,
                max_value=2.5,
                getter=lambda: float(getattr(self.engine.config.conflict, "world_aggression", 1.0)),
                setter=lambda v: setattr(self.engine.config.conflict, "world_aggression", v),
                rect=pygame.Rect(left, top + gap * 5, inner_w, h),
            ),
            UISlider(
                label="Vaccination coverage",
                min_value=0.0,
                max_value=0.5,
                getter=lambda: self.engine.config.vaccination.annual_coverage_fraction,
                setter=lambda v: setattr(self.engine.config.vaccination, "annual_coverage_fraction", v),
                rect=pygame.Rect(left, top + gap * 6, inner_w, h),
            ),
            UISlider(
                label="Simulation speed",
                min_value=1.0,
                max_value=1800.0,
                getter=lambda: float(self.step_every_frames),
                setter=lambda v: setattr(self, "step_every_frames", max(1, min(1800, int(round(v))))),
                rect=pygame.Rect(left, top + gap * 7, inner_w, h),
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
            base_y = int(self.height * 0.19) + layer_idx * max(36, int(self.height * 0.048))
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

    def _draw_ground_gradient(self, screen: pygame.Surface, ground_rect: pygame.Rect) -> None:
        h = max(1, ground_rect.height)
        base = self.ground_color
        dark = self._mul_rgb(base, 0.72)
        light = self._mul_rgb(base, 1.12)
        for i in range(h):
            t = i / h
            row = self._lerp_rgb(light, dark, t * t * 0.85)
            if i % 7 == 0:
                row = self._lerp_rgb(row, self._mul_rgb(row, 0.92), 0.35)
            y = ground_rect.top + i
            pygame.draw.line(screen, row, (ground_rect.left, y), (ground_rect.right, y), 1)
        for s in range(0, ground_rect.width, 17):
            x = ground_rect.left + s + (self.year * 3) % 11
            shade = self._mul_rgb(dark, 0.88)
            pygame.draw.line(
                screen,
                shade,
                (x, ground_rect.top + h // 3),
                (x + 2, ground_rect.bottom - 4),
                1,
            )

    def _draw_region_fields(self, screen: pygame.Surface) -> None:
        for region_id in range(self.engine.config.demographics.region_count):
            rect = self._region_rect(region_id)
            base = self.region_colors[region_id % len(self.region_colors)]
            tint = pygame.Surface((max(1, rect.width), max(1, rect.height)), pygame.SRCALPHA)
            tr, tg, tb = base
            tint.fill((tr, tg, tb, 28))
            screen.blit(tint, (rect.left, rect.top))
            edge = self._mul_rgb(base, 0.55)
            pygame.draw.rect(screen, edge, rect, 1)
            grid = self._mul_rgb(self.grid_color, 0.78)
            for x in range(rect.left, rect.right, 36):
                pygame.draw.line(screen, grid, (x, rect.bottom - 22), (x + 14, rect.bottom - 2), 1)

    def _draw_sky_gradient(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        cycle = (self.year % 240) / 240.0
        day_factor = max(0.22, 1.0 - abs(cycle - 0.5) * 1.65)
        zenith = (
            int(28 + 55 * day_factor),
            int(48 + 85 * day_factor),
            int(92 + 118 * day_factor),
        )
        horizon = (
            int(118 + 95 * day_factor),
            int(152 + 78 * day_factor),
            int(188 + 62 * day_factor),
        )
        haze = (
            int(200 + 40 * day_factor),
            int(188 + 35 * day_factor),
            int(168 + 50 * day_factor),
        )
        height = max(1, rect.height)
        horizon_y = int(rect.height * 0.58)
        for i in range(height):
            y_world = rect.top + i
            if i < horizon_y:
                t = i / max(1, horizon_y)
                t_ease = t * t * (3.0 - 2.0 * t)
                color = self._lerp_rgb(zenith, horizon, t_ease)
            else:
                t = (i - horizon_y) / max(1, height - horizon_y)
                color = self._lerp_rgb(horizon, haze, min(1.0, t * 1.35))
            if i == horizon_y:
                glow = self._lerp_rgb(horizon, (255, 228, 200), 0.22 * day_factor)
                color = glow
            pygame.draw.line(screen, color, (rect.left, y_world), (rect.right, y_world), 1)

    def _draw_sun_and_clouds(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        cycle = (self.year % 240) / 240.0
        pad = max(24, min(80, rect.width // 14))
        sun_x = int(rect.left + pad + cycle * max(20, rect.width - 2 * pad))
        sun_y = int(rect.top + 72 - max(0.0, 1.0 - abs(cycle - 0.5) * 2.0) * min(52, rect.height // 8))
        glow_r = 52
        glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        for r, a in [(46, 18), (36, 35), (26, 58), (16, 85), (9, 120)]:
            pygame.draw.circle(glow, (255, 248, 220, a), (glow_r, glow_r), r)
        blend = getattr(pygame, "BLEND_ALPHA_SDL2", 0)
        screen.blit(glow, (sun_x - glow_r, sun_y - glow_r), special_flags=blend)
        pygame.draw.circle(screen, (255, 248, 210), (sun_x, sun_y), 22)
        pygame.draw.circle(screen, (255, 255, 235), (sun_x - 6, sun_y - 5), 8)
        drift = (self.year * 2.7 + self.frame_counter * 0.02) % (rect.width * 0.4)
        rw = max(80, rect.width)
        cloud_surf = pygame.Surface((rect.width, int(rect.height * 0.42) + 40), pygame.SRCALPHA)
        for ox, oy, sc, r_base in self._cloud_offsets:
            cx = int(ox * rw - drift * 0.15) % (rw + 80) - 20
            cy = int(oy * max(100, rect.height * 0.32))
            alpha = 76
            base = (250, 252, 255, alpha)
            sh = (205, 214, 232, min(110, alpha + 32))
            for dx, dy, rr, col in [
                (0, 7, max(6, int(r_base * sc * 0.45)), sh),
                (0, 0, max(7, int(r_base * sc * 0.5)), base),
                (max(10, int(18 * sc)), -4, max(6, int(r_base * sc * 0.42)), base),
                (max(18, int(34 * sc)), 2, max(5, int(r_base * sc * 0.38)), base),
            ]:
                pygame.draw.circle(cloud_surf, col, (cx + dx, cy + dy), rr)
        screen.blit(cloud_surf, (rect.left, rect.top), special_flags=blend)
        if cycle < 0.14 or cycle > 0.86:
            step = max(52, min(96, rect.width // 9))
            for sx in range(rect.left + 36, rect.right - 36, step):
                pygame.draw.circle(screen, (230, 232, 245), (sx, rect.top + 38 + (sx % 17)), 1)

    def _draw_hills(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        del rect
        layer_colors = [(62, 98, 74), (52, 88, 66), (44, 78, 58)]
        highlights = [(88, 128, 98), (78, 118, 90), (68, 108, 82)]
        for color, hi, points in zip(layer_colors, highlights, self.hills):
            pygame.draw.polygon(screen, color, points)
            if len(points) >= 4:
                ridge = points[:-2]
                if len(ridge) >= 2:
                    for i in range(len(ridge) - 1):
                        a, b = ridge[i], ridge[i + 1]
                        pygame.draw.line(screen, hi, a, b, 2)
                    mid = len(ridge) // 2
                    if mid > 0:
                        p = ridge[mid]
                        pygame.draw.line(screen, self._mul_rgb(hi, 1.08), (p[0], p[1]), (p[0] + 2, p[1] + 8), 1)

    def _draw_river(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        points = []
        y_base = rect.top + int(rect.height * 0.62)
        for x in range(rect.left, rect.right + 1, 14):
            wave = int(16 * math.sin((x + self.year * 2.2) * 0.0095) + 5 * math.sin(x * 0.021))
            points.append((x, y_base + wave))
        if len(points) < 2:
            return
        bank_w = 16
        for idx in range(len(points) - 1):
            pygame.draw.line(screen, (48, 92, 68), points[idx], points[idx + 1], bank_w + 6)
        for idx in range(len(points) - 1):
            pygame.draw.line(screen, (38, 98, 138), points[idx], points[idx + 1], bank_w)
        for idx in range(len(points) - 1):
            pygame.draw.line(screen, (72, 142, 188), points[idx], points[idx + 1], 9)
            pygame.draw.line(screen, (118, 186, 224), points[idx], points[idx + 1], 4)
        for idx in range(0, len(points) - 1, 5):
            hx, hy = points[idx]
            pygame.draw.circle(screen, (200, 230, 248), (hx, hy - 1), 2)

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

        # Dedicated scrollable city ledger area (shrink on short windows).
        ledger_h = min(300, max(100, int(self.height * 0.28)))
        ledger_top = max(120, self.height - ledger_h - max(10, self.height // 80))
        ledger_rect = pygame.Rect(10, ledger_top, max(40, self.left_panel_width - 20), ledger_h)
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

