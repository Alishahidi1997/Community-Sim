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
class VisualAnimal:
    x: float
    y: float
    vx: float
    vy: float
    species: str
    region_id: int
    display_name: str = ""
    age_years: int = 0
    is_livestock: bool = False


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
        self.mouse_pos: tuple[int, int] = (0, 0)
        self.pinned_person_id: int | None = None
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.zoom = 1.0
        self.world_dragging = False
        self.world_drag_last: tuple[int, int] = (0, 0)

        self.region_colors = [
            (86, 108, 88),
            (88, 96, 104),
            (102, 94, 90),
            (94, 100, 82),
            (82, 106, 90),
            (92, 90, 102),
            (98, 96, 94),
            (88, 104, 100),
        ]
        self.bg_color = (98, 132, 168)
        self.ground_color = (76, 118, 82)
        self.panel_color = (28, 30, 36)
        self.panel_text_color = (236, 238, 244)
        self.ui_accent = (64, 148, 220)
        self.ui_text_dim = (168, 174, 188)
        self.recent_messages: list[tuple[str, int]] = []
        self.timeline_cache: list[str] = []
        self.last_major_event_count = 0
        self.city_scroll_offset = 0
        self.terrain_seed = random.Random(engine.config.random_seed + 2026)
        self.hills = self._build_hills()
        self._build_sliders()
        self._cloud_offsets: list[tuple[float, float, float, int]] = []
        self._init_cloud_layout()
        self.visual_animals: list[VisualAnimal] = []
        self._scatter_veg: list[tuple[float, float, str, float]] = []
        self._scatter_reeds: list[tuple[float, float, float]] = []
        self._decor_crops: list[tuple[float, float, float]] = []
        self._rebuild_world_decor()
        self._sync_visual_animals()

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

    def _world_rect(self) -> pygame.Rect:
        """Viewport (screen) rectangle where the map is drawn — fixed to window."""
        return pygame.Rect(self.left_panel_width, 0, self.width - self.panel_width - self.left_panel_width, self.height)

    def _viewport_size(self) -> tuple[int, int]:
        vw = max(80, self.width - self.panel_width - self.left_panel_width)
        vh = self.height
        return vw, vh

    def _map_dimensions(self) -> tuple[int, int]:
        """Logical world size in pixels (wider than the viewport for scrolling)."""
        vw, vh = self._viewport_size()
        regions = max(1, self.engine.config.demographics.region_count)
        map_w = max(int(vw * 3.0), int(vw * min(5.0, 1.15 * regions)))
        map_h = int(vh * 2.05)
        return map_w, map_h

    def _map_world_rect(self) -> pygame.Rect:
        mw, mh = self._map_dimensions()
        return pygame.Rect(0, 0, mw, mh)

    def _w2s(self, wx: float, wy: float) -> tuple[int, int]:
        z = max(0.001, self.zoom)
        sx = int(self.left_panel_width + (wx - self.camera_x) * z)
        sy = int((wy - self.camera_y) * z)
        return sx, sy

    def _s2w(self, sx: float, sy: float) -> tuple[float, float]:
        z = max(0.001, self.zoom)
        wx = (sx - self.left_panel_width) / z + self.camera_x
        wy = sy / z + self.camera_y
        return float(wx), float(wy)

    def _zs(self, d: float) -> float:
        """Scale a world-space length to screen pixels (for sprite sizes, stroke widths)."""
        return d * max(0.001, self.zoom)

    def _zi(self, d: float) -> int:
        v = int(round(self._zs(d)))
        return max(1, v) if d > 0.05 else max(0, v)

    def _clamp_camera(self) -> None:
        vw, vh = self._viewport_size()
        mw, mh = self._map_dimensions()
        z = max(0.001, self.zoom)
        max_cx = max(0.0, float(mw - vw / z))
        max_cy = max(0.0, float(mh - vh / z))
        self.camera_x = max(0.0, min(max_cx, self.camera_x))
        self.camera_y = max(0.0, min(max_cy, self.camera_y))

    def _world_on_screen(self, wx: float, wy: float, margin: float = 80.0) -> bool:
        vw, vh = self._viewport_size()
        z = max(0.001, self.zoom)
        mw = margin / z
        return (
            self.camera_x - mw <= wx <= self.camera_x + vw / z + mw
            and self.camera_y - mw <= wy <= self.camera_y + vh / z + mw
        )

    def _set_zoom(self, new_zoom: float, anchor_sx: float | None = None, anchor_sy: float | None = None) -> None:
        z = max(0.32, min(2.85, float(new_zoom)))
        old = max(0.001, self.zoom)
        if abs(z - old) < 1e-6:
            return
        vw, vh = self._viewport_size()
        lp = float(self.left_panel_width)
        ax = lp + vw / 2.0 if anchor_sx is None else float(anchor_sx)
        ay = vh / 2.0 if anchor_sy is None else float(anchor_sy)
        wx = (ax - lp) / old + self.camera_x
        wy = ay / old + self.camera_y
        self.zoom = z
        self.camera_x = wx - (ax - lp) / z
        self.camera_y = wy - ay / z
        self._clamp_camera()

    def _reset_camera_centered(self) -> None:
        vw, vh = self._viewport_size()
        mw, mh = self._map_dimensions()
        z = max(0.001, self.zoom)
        self.camera_x = max(0.0, (mw - vw / z) / 2.0)
        self.camera_y = max(0.0, (mh - vh / z) / 2.0)
        self._clamp_camera()

    def _ocean_band_height(self, world_rect: pygame.Rect) -> int:
        return max(44, min(168, int(world_rect.height * 0.14)))

    def _is_point_over_water(self, wr: pygame.Rect, px: int, py: int) -> bool:
        """True if pixel lies in drawn ocean band, left bay, or right fjord (matches _draw_ocean_and_seas)."""
        oh = self._ocean_band_height(wr)
        sea_top = wr.bottom - oh
        if py >= sea_top - 6:
            return True
        y_mid = wr.top + int(wr.height * 0.5)
        if py >= y_mid:
            frac = (py - y_mid) / max(1, wr.bottom - y_mid)
            span = int(frac * wr.width * 0.26) + 12
            if px < wr.left + span:
                return True
        y0 = wr.top + int(wr.height * 0.54)
        fj_w = max(24, int(wr.width * 0.065))
        if py >= y0 and px >= wr.right - fj_w - 2:
            return True
        return False

    def _rebuild_world_decor(self) -> None:
        wr = self._map_world_rect()
        r = self.terrain_seed
        self._scatter_veg = []
        self._scatter_reeds = []
        self._decor_crops = []
        count = int(max(40, min(200, wr.width * wr.height // 9800)))
        for _ in range(count):
            placed = False
            for _try in range(18):
                nx = r.random()
                ny = 0.22 + r.random() * 0.72
                px = int(wr.left + nx * wr.width)
                py = int(wr.top + ny * wr.height)
                if self._is_point_over_water(wr, px, py):
                    continue
                if 0.56 < ny < 0.72 and 0.38 < nx < 0.62:
                    continue
                roll = r.random()
                oh = self._ocean_band_height(wr)
                coast_ny = (wr.bottom - oh - wr.top) / max(1, wr.height)
                is_coast_strip = ny >= coast_ny - 0.14 and ny < coast_ny - 0.02
                if roll < 0.28:
                    kind = "tree"
                    sc = r.uniform(0.75, 1.35)
                elif roll < 0.42:
                    kind = "pine"
                    sc = r.uniform(0.7, 1.25)
                elif roll < 0.52 and is_coast_strip:
                    kind = "palm"
                    sc = r.uniform(0.85, 1.2)
                elif roll < 0.52:
                    kind = "tree"
                    sc = r.uniform(0.65, 1.1)
                elif roll < 0.66:
                    kind = "bush"
                    sc = r.uniform(0.5, 1.0)
                elif roll < 0.76:
                    kind = "flowers"
                    sc = r.uniform(0.55, 1.0)
                elif roll < 0.86:
                    kind = "grass"
                    sc = r.uniform(0.45, 0.95)
                elif roll < 0.93:
                    kind = "fern"
                    sc = r.uniform(0.5, 1.0)
                else:
                    kind = "rock"
                    sc = r.uniform(0.4, 0.85)
                self._scatter_veg.append((nx, ny, kind, sc))
                placed = True
                break
            if not placed:
                continue
        n_reeds = max(12, min(56, wr.width // 48))
        for _ in range(n_reeds):
            for _try in range(14):
                nx = 0.38 + r.random() * 0.24
                ny = 0.58 + r.random() * 0.18
                px = int(wr.left + nx * wr.width)
                py = int(wr.top + ny * wr.height)
                if self._is_point_over_water(wr, px, py):
                    continue
                self._scatter_reeds.append((nx, ny, r.uniform(0.65, 1.15)))
                break
        for _ in range(max(10, min(48, wr.width // 38))):
            for _try in range(16):
                nx = r.random()
                ny = 0.42 + r.random() * 0.36
                px = int(wr.left + nx * wr.width)
                py = int(wr.top + ny * wr.height)
                if self._is_point_over_water(wr, px, py):
                    continue
                self._decor_crops.append((nx, ny, r.uniform(0.7, 1.2)))
                break

    def _load_ui_fonts(self) -> tuple[pygame.font.Font, pygame.font.Font]:
        avail = set(pygame.font.get_fonts())
        for key, display in [
            ("segoeui", "Segoe UI"),
            ("calibri", "Calibri"),
            ("tahoma", "Tahoma"),
            ("arial", "Arial"),
        ]:
            if key in avail:
                return (
                    pygame.font.SysFont(display, 19),
                    pygame.font.SysFont(display, 14),
                )
        return pygame.font.SysFont("consolas", 18), pygame.font.SysFont("consolas", 15)

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
        self._rebuild_world_decor()
        self._clamp_agents_to_regions()
        self._reset_camera_centered()

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
        font, small_font = self._load_ui_fonts()

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
                    self._rebuild_world_decor()
                    self._clamp_agents_to_regions()
                    self._reset_camera_centered()
            if self._pending_resize is not None and not self._fullscreen:
                nw, nh = self._pending_resize
                self._pending_resize = None
                self._apply_window_size(nw, nh)
                screen = pygame.display.set_mode((self.width, self.height), win_flags)
            if not self.paused:
                self._tick_simulation()
                self._move_agents(dt)
                self._move_visual_animals(dt)
            self._update_camera_from_input(dt)
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
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self._set_zoom(self.zoom * 1.12)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._set_zoom(self.zoom / 1.12)
                elif event.key == pygame.K_0:
                    self.zoom = 1.0
                    self._reset_camera_centered()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                hit_slider = False
                for slider in self.sliders:
                    if slider.rect.collidepoint(event.pos):
                        slider.active = True
                        self.dragging_slider = slider
                        slider.set_from_x(event.pos[0])
                        hit_slider = True
                        break
                if not hit_slider and self.left_panel_width <= mx < self.width - self.panel_width:
                    pick = self._pick_at(mx, my)
                    if pick is not None and pick[0] == "person":
                        self.pinned_person_id = pick[1].person_id
                    else:
                        self.pinned_person_id = None
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.dragging_slider is not None:
                    self.dragging_slider.active = False
                self.dragging_slider = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                mx, my = event.pos
                if self.left_panel_width <= mx < self.width - self.panel_width:
                    self.world_dragging = True
                    self.world_drag_last = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                self.world_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
                if self.world_dragging:
                    mx, my = event.pos
                    lx, ly = self.world_drag_last
                    z = max(0.001, self.zoom)
                    self.camera_x -= (mx - lx) / z
                    self.camera_y -= (my - ly) / z
                    self.world_drag_last = event.pos
                    self._clamp_camera()
                if self.dragging_slider is not None:
                    self.dragging_slider.set_from_x(event.pos[0])
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = self.mouse_pos
                mods = pygame.key.get_mods()
                if mx < self.left_panel_width:
                    self._scroll_city_ledger(-event.y)
                elif mx < self.width - self.panel_width:
                    if mods & pygame.KMOD_CTRL:
                        factor = 1.1 ** event.y
                        self._set_zoom(self.zoom * factor, float(mx), float(my))
                    else:
                        z = max(0.001, self.zoom)
                        self.camera_y -= event.y * 56.0 / z
                        self._clamp_camera()
                # else: over right panel — ignore wheel
            elif event.type == pygame.VIDEORESIZE and not self._fullscreen:
                nw = max(1, int(getattr(event, "w", event.size[0])))
                nh = max(1, int(getattr(event, "h", event.size[1])))
                self._pending_resize = self._clamp_to_desktop(nw, nh)

    def _update_camera_from_input(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        z = max(0.001, self.zoom)
        spd = 480.0 * dt / z
        moved = False
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.camera_x -= spd
            moved = True
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.camera_x += spd
            moved = True
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.camera_y -= spd
            moved = True
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.camera_y += spd
            moved = True
        if moved:
            self._clamp_camera()

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
        self._sync_visual_animals()
        if self.year >= self.engine.config.years:
            self.paused = True

    def _hud_top(self) -> int:
        return max(64, min(88, self.height // 14))

    @staticmethod
    def _map_hud_top(map_h: int) -> int:
        return max(64, min(88, map_h // 14))

    def _region_rect(self, region_id: int) -> pygame.Rect:
        regions = max(1, self.engine.config.demographics.region_count)
        mw, mh = self._map_dimensions()
        region_width = mw / regions
        x0 = int(region_id * region_width + 10)
        top = self._map_hud_top(mh)
        bottom_margin = max(28, min(48, mh // 22))
        mmap = self._map_world_rect()
        oh = self._ocean_band_height(mmap)
        land_bottom = mh - bottom_margin - oh
        return pygame.Rect(x0, top, int(region_width - 20), max(60, land_bottom - top))

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
        max_lines = 48
        show_contacts = self.zoom >= 0.72
        for person in alive_people:
            if not show_contacts or sample_lines > max_lines:
                break
            src = self.visual_state.get(person.person_id)
            if src is None:
                continue
            for neighbor_id in self.engine.contact_graph.get(person.person_id, [])[:2]:
                dst = self.visual_state.get(neighbor_id)
                if dst is None:
                    continue
                if not self._world_on_screen(src.x, src.y, 120) and not self._world_on_screen(dst.x, dst.y, 120):
                    continue
                dx = src.x - dst.x
                dy = src.y - dst.y
                if (dx * dx + dy * dy) > 140 * 140:
                    continue
                a = self._w2s(src.x, src.y)
                b = self._w2s(dst.x, dst.y)
                pygame.draw.line(screen, (158, 168, 178), a, b, max(1, self._zi(1)))
                sample_lines += 1
                if sample_lines > max_lines:
                    break

        self._draw_animals(screen)

        for person in alive_people:
            agent = self.visual_state.get(person.person_id)
            if agent is None:
                continue
            if not self._world_on_screen(agent.x, agent.y, 48):
                continue
            self._draw_human(screen, person, agent)

            if self.show_labels and person.person_id % 20 == 0:
                label = f"{person.person_id}:{person.health:.2f}"
                text = small_font.render(label, True, (16, 18, 22))
                lx, ly = self._w2s(agent.x, agent.y)
                screen.blit(text, (lx + 8, ly - 26))

        self._draw_hud(screen, font, alive_people)
        self._draw_messages(screen, small_font)
        self._draw_control_panel(screen, font, small_font)
        self._draw_hover_tooltip(screen, small_font)

    def _draw_regions(self, screen: pygame.Surface) -> None:
        vp = self._world_rect()
        clip_prev = screen.get_clip()
        screen.set_clip(vp)
        self._draw_sky_gradient(screen, vp)
        self._draw_sun_and_clouds(screen, vp)
        self._draw_hills(screen, vp)
        self._draw_ground_gradient_world(screen)
        self._draw_river_world(screen)
        self._draw_ocean_and_seas_world(screen)
        self._draw_region_fields(screen)
        self._draw_world_vegetation(screen)
        mmap = self._map_world_rect()
        self._draw_reeds_and_shore_plants(screen, mmap)
        self._draw_decorative_farmland(screen)
        self._draw_scattered_homesteads(screen)
        self._draw_world_structures(screen)
        self._draw_livestock_enclosures(screen)
        self._draw_roads(screen)
        screen.set_clip(clip_prev)

    def _draw_hud(self, screen: pygame.Surface, font: pygame.font.Font, alive_people: list) -> None:
        wr = self._world_rect()
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
            f"Wildlife: {getattr(self.engine, 'wildlife_index', 1.0):.2f}   Livestock: {self.engine.total_livestock():.0f}   Farms (fields): {sum(1 for s in self.engine.world_structures if s.get('kind') == 'field')}",
            f"Avg health: {avg_health:.2f}   Food: {self.engine.config.environment.base_food_per_capita:.2f}   Birth rate: {self.engine.config.demographics.base_birth_rate:.2f}   Infection rate: {pathogen_rate:.2f}",
            "Map: WASD pan · wheel (Ctrl+zoom) · +/- keys · 0=reset · middle-drag | SPACE ,/. L ESC",
        ]
        line_h = max(16, min(22, self.height // 48))
        plate_h = len(lines) * line_h + 16
        plate_w = max(400, wr.width - 16)
        blend = getattr(pygame, "BLEND_ALPHA_SDL2", 0)
        plate = pygame.Surface((plate_w, plate_h), pygame.SRCALPHA)
        plate.fill((252, 253, 255, 100))
        pygame.draw.rect(plate, (255, 255, 255, 40), plate.get_rect(), border_radius=10)
        screen.blit(plate, (wr.left + 8, 6), special_flags=blend)
        pygame.draw.rect(
            screen,
            (138, 148, 168),
            pygame.Rect(wr.left + 8, 6, plate_w, plate_h),
            1,
            border_radius=10,
        )
        y = 14
        for i, line in enumerate(lines):
            col = (28, 32, 42) if i < 2 else (48, 52, 62)
            text = font.render(line, True, col)
            screen.blit(text, (wr.left + 16, y))
            y += line_h

    def _draw_messages(self, screen: pygame.Surface, small_font: pygame.font.Font) -> None:
        self.recent_messages = [(m, ttl - 1) for m, ttl in self.recent_messages if ttl > 1]
        wr = self._world_rect()
        y = self._hud_top() - 4
        for msg, _ in self.recent_messages[-6:]:
            t = small_font.render(msg, True, (42, 48, 60))
            screen.blit(t, (wr.left + 14, y))
            y += max(14, min(20, self.height // 55))

    def _draw_human(self, screen: pygame.Surface, person, agent: VisualAgent) -> None:
        body_color = self._health_color(person)
        shade = self._mul_rgb(body_color, 0.78)
        accent = (64, 108, 152) if person.gender == Gender.MALE else (148, 88, 128)
        accent_soft = self._lerp_rgb(accent, (240, 240, 242), 0.35)
        zt = max(0.001, self.zoom)
        scale = max(2, min(22, int((max(4, min(14, int(4 + person.age / 10)))) * zt)))
        x, y = self._w2s(agent.x, agent.y)

        foot_y = y + scale + scale // 2 + 1
        shw = scale * 2 + max(2, scale // 2)
        shy = max(2, scale // 3)
        pygame.draw.ellipse(screen, (36, 48, 40), pygame.Rect(x - shw // 2, foot_y - 1, shw, shy))
        pygame.draw.ellipse(screen, (28, 34, 32), pygame.Rect(x - scale, foot_y - 2, scale * 2, max(4, scale // 2)))

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
        split = int(panel_rect.height * 0.38)
        pygame.draw.rect(
            screen,
            (36, 38, 46),
            pygame.Rect(panel_rect.left, panel_rect.top, panel_rect.width, split),
        )
        pygame.draw.rect(
            screen,
            (24, 26, 32),
            pygame.Rect(panel_rect.left, panel_rect.top + split, panel_rect.width, panel_rect.height - split),
        )
        pygame.draw.rect(screen, self.ui_accent, (panel_rect.left, panel_rect.top, 4, panel_rect.height))
        pygame.draw.line(screen, (10, 11, 14), (panel_rect.left, 0), (panel_rect.left, self.height), 2)
        pygame.draw.line(screen, (52, 56, 64), (panel_rect.left + 1, 0), (panel_rect.left + 1, self.height), 1)

        title = font.render("Simulation controls", True, self.panel_text_color)
        screen.blit(title, (panel_rect.left + 22, 24))
        subtitle = small_font.render("Adjust parameters in real time", True, self.ui_text_dim)
        screen.blit(subtitle, (panel_rect.left + 22, 50))
        pygame.draw.line(
            screen,
            (52, 56, 66),
            (panel_rect.left + 18, 74),
            (panel_rect.right - 14, 74),
            1,
        )

        for slider in self.sliders:
            label = small_font.render(f"{slider.label}  ·  {slider.current():.3f}", True, (210, 214, 224))
            screen.blit(label, (slider.rect.left, slider.rect.top - 22))

            inset = pygame.Rect(slider.rect.left, slider.rect.top - 1, slider.rect.width, slider.rect.height + 2)
            pygame.draw.rect(screen, (12, 13, 16), inset, border_radius=8)
            pygame.draw.rect(screen, (44, 46, 54), slider.rect, border_radius=7)
            fill_width = int(slider.rect.width * slider.normalized())
            if fill_width > 0:
                fill_rect = pygame.Rect(slider.rect.left, slider.rect.top, fill_width, slider.rect.height)
                pygame.draw.rect(screen, (42, 118, 188), fill_rect, border_radius=7)
                hi = pygame.Rect(fill_rect.left, fill_rect.top, fill_rect.width, max(1, fill_rect.height // 2))
                pygame.draw.rect(screen, (88, 168, 228), hi, border_radius=7)

            handle_x = slider.rect.left + fill_width
            handle = pygame.Rect(handle_x - 7, slider.rect.top - 5, 14, slider.rect.height + 10)
            pygame.draw.rect(screen, (18, 20, 26), (handle.left + 1, handle.top + 1, handle.width, handle.height), border_radius=5)
            hcol = (252, 252, 255) if slider.active else (232, 234, 240)
            pygame.draw.rect(screen, hcol, handle, border_radius=5)
            pygame.draw.rect(screen, (100, 108, 124), handle, 1, border_radius=5)

        tips = [
            "Zoom: Ctrl+wheel or +/- keys · 0 resets view",
            "Pan: WASD · wheel · middle-drag · hover / click people",
        ]
        y = self.height - 52
        for tip in tips:
            t = small_font.render(tip, True, (150, 156, 170))
            screen.blit(t, (panel_rect.left + 22, y))
            y += 18

    def _person_hover_radius(self, person) -> float:
        scale = max(4, min(14, int(4 + person.age / 10)))
        z = max(0.001, self.zoom)
        return max(18.0, float(scale) * 2.7) / z

    def _animal_hover_radius(self, a: VisualAnimal) -> float:
        z = max(0.001, self.zoom)
        return float({"bird": 12.0, "deer": 17.0, "sheep": 14.0, "cattle": 22.0}.get(a.species, 14.0)) / z

    def _pick_at(self, mx: int, my: int):
        if mx < self.left_panel_width or mx >= self.width - self.panel_width:
            return None
        wx, wy = self._s2w(mx, my)

        best_p = None
        best_pd = 1e9
        for p in self.engine.population:
            if not p.alive:
                continue
            ag = self.visual_state.get(p.person_id)
            if ag is None:
                continue
            d = math.hypot(wx - ag.x, wy - ag.y)
            r = self._person_hover_radius(p)
            if d <= r and d < best_pd:
                best_pd = d
                best_p = p
        if best_p is not None:
            return ("person", best_p)

        best_a = None
        best_ad = 1e9
        for a in self.visual_animals:
            d = math.hypot(wx - a.x, wy - a.y)
            rr = self._animal_hover_radius(a)
            if d <= rr and d < best_ad:
                best_ad = d
                best_a = a
        if best_a is not None:
            return ("animal", best_a)

        best_s = None
        best_sd = 1e9
        for s in self.engine.world_structures:
            sx, sy = self._structure_world_pos(s)
            d = math.hypot(wx - sx, wy - sy)
            if d <= 28.0 / max(0.001, self.zoom) and d < best_sd:
                best_sd = d
                best_s = s
        if best_s is not None:
            return ("structure", best_s)
        return None

    def _hover_pick(self):
        return self._pick_at(*self.mouse_pos)

    def _city_name_for_region(self, region_id: int) -> str | None:
        st = self.engine._get_settlement_structure(region_id)
        if st is None:
            return None
        if str(st.get("level", "")) == "city":
            return str(st.get("name", ""))
        sname = str(st.get("name", ""))
        for c in self.engine.city_summaries:
            if str(c.get("name", "")) == sname:
                return sname
        return None

    def _person_tooltip_lines(self, person) -> list[str]:
        settlement = self.engine._region_name(person.region_id)
        city = self._city_name_for_region(person.region_id)
        loc = f"{settlement}"
        if city:
            loc = f"{city} ({settlement})"
        lines = [
            f"Person #{person.person_id}",
            f"Age {person.age} · {person.gender.value}",
            f"Place: {loc}",
        ]
        bg = person.belief_group
        if bg.startswith("cult_"):
            lines.append(f"Belief: cult — {bg}")
        else:
            lines.append(f"Belief: {bg}")
        lines.append(f"Faction: {person.faction} · Language: {person.language}")
        lines.append(f"Health {person.health:.2f} · Happy {person.happiness:.2f} · Stress {person.stress:.2f}")
        inf = [k for k, st in person.disease_states.items() if st == DiseaseState.INFECTED]
        if inf:
            lines.append(f"Infected: {', '.join(inf)}")
        if person.vaccinated:
            lines.append("Vaccinated: yes")
        lines.append(f"Goal: {person.primary_goal} · Know {person.knowledge:.2f} · Tools {person.tool_skill:.2f}")
        if person.riding_skill > 0.08:
            lines.append(f"Riding {person.riding_skill:.2f} · Inventions made {person.inventions_made}")
        else:
            lines.append(f"Inventions made {person.inventions_made}")
        pol = self.engine.politics_by_region.get(person.region_id, {})
        if pol.get("leader_id") == person.person_id:
            title = str(pol.get("leader_title") or "Leader")
            lines.append(f"Role: {title} (regional leader)")
        return lines

    def _animal_tooltip_lines(self, a: VisualAnimal) -> list[str]:
        kind = a.species.capitalize()
        origin = "Migratory / wild" if a.species == "bird" else ("Wild" if not a.is_livestock else "Herd / livestock")
        region = self.engine._region_name(a.region_id) if a.species != "bird" else "Open sky"
        lines = [
            a.display_name or kind,
            f"Type: {kind} · {origin}",
            f"Age (sim years est.): {a.age_years}",
            f"Region: {region}",
        ]
        return lines

    def _structure_tooltip_lines(self, s: dict) -> list[str]:
        kind = str(s.get("kind", "?"))
        rid = int(s.get("region_id", 0))
        reg = self.engine._region_name(rid)
        lines = [f"{kind.title()} · {reg}"]
        if "name" in s:
            lines.append(f"Name: {s['name']}")
        if "level" in s:
            lines.append(f"Level: {s['level']}")
        lines.append(f"Id: {s.get('id', '?')}")
        return lines

    def _clamp_tooltip_top_left(self, ox: int, oy: int, box_w: int, box_h: int) -> tuple[int, int]:
        margin = 6
        left = self.left_panel_width + margin
        right = self.width - self.panel_width - margin - box_w
        top = margin
        bottom = self.height - margin - box_h
        if right < left:
            ox = left
        else:
            ox = max(left, min(right, ox))
        if bottom < top:
            oy = top
        else:
            oy = max(top, min(bottom, oy))
        return ox, oy

    def _draw_tooltip_card(
        self,
        screen: pygame.Surface,
        small_font: pygame.font.Font,
        lines: list[str],
        ox: int,
        oy: int,
    ) -> None:
        pad = 10
        lh = small_font.get_height() + 4
        rendered: list[pygame.Surface] = []
        max_w = 0
        for ln in lines:
            t = small_font.render(ln[:96], True, (24, 26, 32))
            rendered.append(t)
            max_w = max(max_w, t.get_width())
        box_w = max_w + pad * 2
        box_h = len(rendered) * lh + pad * 2
        ox, oy = self._clamp_tooltip_top_left(ox, oy, box_w, box_h)

        br = 10
        pygame.draw.rect(
            screen,
            (16, 18, 24),
            pygame.Rect(ox + 4, oy + 4, box_w, box_h),
            border_radius=br,
        )
        pygame.draw.rect(screen, (252, 253, 255), pygame.Rect(ox, oy, box_w, box_h), border_radius=br)
        pygame.draw.rect(screen, (72, 118, 168), pygame.Rect(ox, oy, box_w, box_h), 1, border_radius=br)
        cy = oy + pad
        for idx, t in enumerate(rendered):
            if idx == 0:
                t0 = small_font.render(lines[idx][:96], True, (28, 72, 118))
                screen.blit(t0, (ox + pad, cy))
            else:
                screen.blit(t, (ox + pad, cy))
            cy += lh

    def _pinned_person(self):
        if self.pinned_person_id is None:
            return None
        for p in self.engine.population:
            if p.alive and p.person_id == self.pinned_person_id:
                return p
        return None

    def _draw_pinned_person_tooltip(self, screen: pygame.Surface, small_font: pygame.font.Font) -> None:
        person = self._pinned_person()
        if person is None:
            self.pinned_person_id = None
            return
        agent = self.visual_state.get(person.person_id)
        if agent is None:
            return
        lines = list(self._person_tooltip_lines(person))
        lines.append("Following — click another person to switch; empty map clears")
        pad = 10
        lh = small_font.get_height() + 4
        max_w = 0
        for ln in lines:
            t = small_font.render(ln[:96], True, (24, 26, 32))
            max_w = max(max_w, t.get_width())
        box_w = max_w + pad * 2
        box_h = len(lines) * lh + pad * 2
        ax, ay = self._w2s(agent.x, agent.y)
        ox = ax + 16
        oy = ay - box_h - 12
        if oy < self._hud_top() + 4:
            oy = ay + 22
        self._draw_tooltip_card(screen, small_font, lines, ox, oy)

    def _draw_hover_tooltip(self, screen: pygame.Surface, small_font: pygame.font.Font) -> None:
        self._draw_pinned_person_tooltip(screen, small_font)

        pick = self._hover_pick()
        if pick is None:
            return
        kind, obj = pick
        if kind == "person" and self.pinned_person_id == obj.person_id:
            return
        if kind == "person":
            lines = self._person_tooltip_lines(obj)
        elif kind == "animal":
            lines = self._animal_tooltip_lines(obj)
        else:
            lines = self._structure_tooltip_lines(obj)

        mx, my = self.mouse_pos
        pad = 10
        lh = small_font.get_height() + 4
        max_w = 0
        for ln in lines:
            t = small_font.render(ln[:96], True, (24, 26, 32))
            max_w = max(max_w, t.get_width())
        box_w = max_w + pad * 2
        box_h = len(lines) * lh + pad * 2
        ox, oy = mx + 18, my + 18
        if ox + box_w > self.width - self.panel_width - 6:
            ox = mx - box_w - 18
        if ox < self.left_panel_width + 6:
            ox = self.left_panel_width + 8
        if oy + box_h > self.height - 10:
            oy = my - box_h - 18
        if oy < 8:
            oy = 8
        ox, oy = self._clamp_tooltip_top_left(ox, oy, box_w, box_h)
        self._draw_tooltip_card(screen, small_font, lines, ox, oy)

    def _push_message(self, msg: str) -> None:
        self.recent_messages.append((msg, 240))

    def _build_hills(self) -> list[list[tuple[int, int]]]:
        mw, mh = self._map_dimensions()
        layers = []
        step = max(64, min(160, mw // 45))
        for layer_idx in range(3):
            points = []
            base_y = int(mh * 0.19) + layer_idx * max(28, int(mh * 0.042))
            x = 0
            while x <= mw + step:
                y = base_y + self.terrain_seed.randint(-11, 11)
                points.append((x, y))
                x += step
            if points[-1][0] < mw:
                y = base_y + self.terrain_seed.randint(-9, 9)
                points.append((mw, y))
            points.extend([(mw, mh), (0, mh)])
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

    def _draw_ground_gradient_world(self, screen: pygame.Surface) -> None:
        mw, mh = self._map_dimensions()
        mmap = self._map_world_rect()
        ground_top_w = max(56, self._map_hud_top(mh) - 8)
        oh = self._ocean_band_height(mmap)
        land_h = max(50, mh - ground_top_w - oh)
        vw, vh = self._viewport_size()
        z = max(0.001, self.zoom)
        vy0 = int(self.camera_y)
        vy1 = int(self.camera_y + vh / z) + 2
        h = land_h
        base = self.ground_color
        dark = self._mul_rgb(base, 0.72)
        light = self._mul_rgb(base, 1.12)
        g0 = ground_top_w
        g1 = ground_top_w + land_h
        for wy in range(max(g0, vy0), min(g1, vy1)):
            i = wy - g0
            t = i / max(1, h - 1)
            row = self._lerp_rgb(light, dark, t * t * 0.82 + t * 0.06)
            sx0, sy = self._w2s(0, wy)
            sx1, _ = self._w2s(mw, wy)
            if sx1 <= self.left_panel_width or sx0 >= self.width - self.panel_width:
                continue
            pygame.draw.line(screen, row, (sx0, sy), (sx1, sy), 1)

    def _draw_region_fields(self, screen: pygame.Surface) -> None:
        z = max(0.001, self.zoom)
        for region_id in range(self.engine.config.demographics.region_count):
            rect = self._region_rect(region_id)
            base = self.region_colors[region_id % len(self.region_colors)]
            tint = pygame.Surface((max(1, rect.width), max(1, rect.height)), pygame.SRCALPHA)
            tr, tg, tb = base
            tint.fill((tr, tg, tb, 18))
            zw = max(1, int(rect.width * z))
            zh = max(1, int(rect.height * z))
            tint_draw = pygame.transform.smoothscale(tint, (zw, zh)) if (zw, zh) != tint.get_size() else tint
            sl, st = self._w2s(rect.left, rect.top)
            screen.blit(tint_draw, (sl, st))
            edge = self._lerp_rgb(base, (52, 58, 54), 0.42)
            pygame.draw.rect(screen, edge, pygame.Rect(sl, st, zw, zh), max(1, self._zi(1)))

    def _draw_ocean_and_seas(self, screen: pygame.Surface, world_rect: pygame.Rect) -> None:
        oh = self._ocean_band_height(world_rect)
        sea_top = world_rect.bottom - oh
        t_anim = self.year * 0.07 + self.frame_counter * 0.0028

        y_mid = world_rect.top + int(world_rect.height * 0.5)
        for y in range(y_mid, world_rect.bottom):
            frac = (y - y_mid) / max(1, world_rect.bottom - y_mid)
            span = int(frac * world_rect.width * 0.26) + 8
            c = self._lerp_rgb((92, 138, 168), (34, 82, 122), min(1.0, frac * 1.05))
            pygame.draw.line(screen, c, (world_rect.left, y), (world_rect.left + span, y), 1)

        fj_w = max(24, int(world_rect.width * 0.065))
        y0 = world_rect.top + int(world_rect.height * 0.54)
        for y in range(y0, world_rect.bottom):
            frac = (y - y0) / max(1, world_rect.bottom - y0)
            c = self._lerp_rgb((78, 128, 168), (40, 88, 128), min(1.0, frac * 0.95))
            pygame.draw.line(screen, c, (world_rect.right - fj_w, y), (world_rect.right, y), 1)

        pygame.draw.rect(
            screen,
            (236, 218, 182),
            pygame.Rect(world_rect.left, sea_top - 7, world_rect.width, 9),
        )
        pygame.draw.line(
            screen,
            (198, 176, 138),
            (world_rect.left, sea_top - 1),
            (world_rect.right, sea_top - 1),
            2,
        )

        shallow = (88, 154, 198)
        mid_w = (48, 118, 168)
        deep_w = (24, 72, 118)
        for i in range(oh):
            t = i / max(1, oh - 1)
            c = self._lerp_rgb(self._lerp_rgb(shallow, mid_w, t * 0.5), deep_w, t * t * 0.88)
            y = sea_top + i
            pygame.draw.line(screen, c, (world_rect.left, y), (world_rect.right, y), 1)

        for x in range(world_rect.left, world_rect.right, 7):
            yw = sea_top + 12 + int(5 * math.sin((x * 0.042) + t_anim * 6))
            pygame.draw.ellipse(screen, (120, 186, 218), pygame.Rect(x - 2, yw - 1, 5, 3))
        for x in range(world_rect.left + 4, world_rect.right, 16):
            yf = sea_top - 4 + int(2 * math.sin((x * 0.07) + t_anim * 7))
            pygame.draw.circle(screen, (248, 252, 255), (x, yf), 1)

    def _draw_ocean_and_seas_world(self, screen: pygame.Surface) -> None:
        mmap = self._map_world_rect()
        mw, mh = mmap.w, mmap.h
        oh = self._ocean_band_height(mmap)
        sea_top = mh - oh
        t_anim = self.year * 0.07 + self.frame_counter * 0.0028
        vw, vh = self._viewport_size()
        z = max(0.001, self.zoom)
        vy0 = int(self.camera_y) - 2
        vy1 = int(self.camera_y + vh / z) + 4

        y_mid = int(mh * 0.5)
        for y in range(max(y_mid, vy0), min(mh, vy1)):
            frac = (y - y_mid) / max(1, mh - y_mid)
            span = int(frac * mw * 0.26) + 8
            c = self._lerp_rgb((118, 158, 182), (62, 102, 128), min(1.0, frac * 0.92))
            a = self._w2s(0, y)
            b = self._w2s(span, y)
            pygame.draw.line(screen, c, a, b, 1)

        fj_w = max(24, int(mw * 0.065))
        y0f = int(mh * 0.54)
        for y in range(max(y0f, vy0), min(mh, vy1)):
            frac = (y - y0f) / max(1, mh - y0f)
            c = self._lerp_rgb((108, 148, 176), (58, 96, 122), min(1.0, frac * 0.88))
            a = self._w2s(mw - fj_w, y)
            b = self._w2s(mw, y)
            pygame.draw.line(screen, c, a, b, 1)

        sx0, sy0 = self._w2s(0, sea_top - 8)
        bw = max(1, int(mw * z))
        bh = max(1, int(10 * z))
        sand_hi = (228, 214, 186)
        sand_lo = (198, 178, 148)
        for bi in range(bh):
            t = bi / max(1, bh - 1)
            row = self._lerp_rgb(sand_hi, sand_lo, t * 0.55)
            pygame.draw.line(screen, row, (sx0, sy0 + bi), (sx0 + bw, sy0 + bi), 1)
        a = self._w2s(0, sea_top - 1)
        b = self._w2s(mw, sea_top - 1)
        pygame.draw.line(screen, (176, 158, 128), a, b, max(1, self._zi(2)))

        shallow = (98, 158, 188)
        mid_w = (62, 124, 158)
        deep_w = (38, 82, 118)
        for i in range(oh):
            t = i / max(1, oh - 1)
            c = self._lerp_rgb(self._lerp_rgb(shallow, mid_w, t * 0.5), deep_w, t * t * 0.88)
            y = sea_top + i
            if y < vy0 or y >= vy1:
                continue
            p0 = self._w2s(0, y)
            p1 = self._w2s(mw, y)
            pygame.draw.line(screen, c, p0, p1, 1)

        vx0 = int(self.camera_x) - 8
        vx1 = int(self.camera_x + vw / z) + 8
        ew, eh = max(2, self._zi(4)), max(2, self._zi(2))
        foam = (142, 188, 210)
        for x in range(max(0, vx0), min(mw, vx1), 22):
            yw = sea_top + 10 + int(4 * math.sin((x * 0.031) + t_anim * 5))
            sx, sy = self._w2s(x, yw)
            pygame.draw.ellipse(screen, foam, pygame.Rect(sx - ew // 2, sy - eh // 2, ew, eh))
        for x in range(max(0, vx0 + 60), min(mw, vx1), 48):
            yf = sea_top - 3 + int(2 * math.sin((x * 0.05) + t_anim * 6))
            sx, sy = self._w2s(x, yf)
            pygame.draw.circle(screen, (220, 232, 240), (sx, sy), max(1, self._zi(1)))

    def _draw_reeds_and_shore_plants(self, screen: pygame.Surface, world_rect: pygame.Rect) -> None:
        wr = world_rect
        for nx, ny, sc in self._scatter_reeds:
            xw = int(wr.left + nx * wr.width)
            yw = int(wr.top + ny * wr.height)
            if not self._world_on_screen(float(xw), float(yw), 24):
                continue
            x, y = self._w2s(xw, yw)
            h = self._zi(14 * sc)
            for off in (-self._zi(3), 0, self._zi(3)):
                pygame.draw.line(
                    screen,
                    (68, 112, 76),
                    (x + off, y),
                    (x + off - 1, y - h),
                    max(1, self._zi(2)),
                )
            pygame.draw.circle(screen, (96, 142, 88), (x - self._zi(2), y - h + self._zi(2)), max(1, self._zi(2)))

        oh = self._ocean_band_height(world_rect)
        sea_top = wr.bottom - oh
        for i in range(max(5, wr.width // 95)):
            xw = wr.left + 18 + (i * 53 + self.year * 7) % max(1, wr.width - 36)
            yw = sea_top - 10 + ((i * 17 + self.frame_counter // 20) % 11) - 5
            if not self._world_on_screen(float(xw), float(yw), 20):
                continue
            x, y = self._w2s(xw, yw)
            pygame.draw.line(screen, (78, 118, 72), (x, y + self._zi(6)), (x - 1, y - self._zi(8)), max(1, self._zi(2)))
            pygame.draw.line(screen, (88, 128, 78), (x + self._zi(2), y + self._zi(5)), (x + 1, y - self._zi(6)), max(1, self._zi(1)))

    def _draw_world_vegetation(self, screen: pygame.Surface) -> None:
        wr = self._map_world_rect()
        for nx, ny, kind, sc in self._scatter_veg:
            xw = int(wr.left + nx * wr.width)
            yw = int(wr.top + ny * wr.height)
            if self._is_point_over_water(wr, xw, yw):
                continue
            if not self._world_on_screen(float(xw), float(yw), 32):
                continue
            x, y = self._w2s(xw, yw)
            if kind == "tree":
                trunk = self._mul_rgb((78, 58, 44), sc)
                foliage = self._mul_rgb((52, 98, 62), sc)
                hi = self._mul_rgb((88, 132, 82), sc)
                shw = max(4, self._zi(10 * sc))
                shy = max(2, self._zi(3))
                pygame.draw.ellipse(screen, (44, 58, 48), pygame.Rect(x - shw // 2, y - shy + 1, shw, shy))
                br = max(0, min(3, self._zi(1)))
                pygame.draw.rect(screen, trunk, pygame.Rect(x - self._zi(2), y - self._zi(4), self._zi(4), self._zi(14 * sc)), border_radius=br)
                pygame.draw.circle(screen, foliage, (x, y - self._zi(15 * sc)), max(2, self._zi(10 * sc)))
                pygame.draw.circle(screen, hi, (x - self._zi(3 * sc), y - self._zi(17 * sc)), max(1, self._zi(4 * sc)))
            elif kind == "pine":
                trunk = self._mul_rgb((70, 54, 42), sc)
                pygame.draw.rect(screen, trunk, pygame.Rect(x - self._zi(2), y - self._zi(3), self._zi(4), self._zi(12 * sc)))
                for i, rw in enumerate([self._zi(14 * sc), self._zi(11 * sc), self._zi(8 * sc)]):
                    oy = y - self._zi((8 + i * 7) * sc)
                    pygame.draw.polygon(
                        screen,
                        (48, 88, 64),
                        [(x, oy - self._zi(10 * sc)), (x - rw, oy + self._zi(2)), (x + rw, oy + self._zi(2))],
                    )
                    pygame.draw.polygon(
                        screen,
                        (62, 112, 78),
                        [(x, oy - self._zi(8 * sc)), (x - rw + max(1, self._zi(2)), oy + 1), (x + rw - max(1, self._zi(2)), oy + 1)],
                    )
            elif kind == "palm":
                trunk = self._mul_rgb((108, 84, 54), sc)
                pygame.draw.rect(screen, trunk, pygame.Rect(x - self._zi(2), y - self._zi(18 * sc), self._zi(4), self._zi(20 * sc)))
                for ang in range(0, 360, 45):
                    rad = math.radians(ang + self.year * 3)
                    x2 = x + int(self._zs(14 * sc) * math.cos(rad))
                    y2 = y - self._zi(16 * sc) + int(self._zs(6 * sc) * math.sin(rad))
                    pygame.draw.line(
                        screen,
                        (64, 118, 76),
                        (x, y - self._zi(18 * sc)),
                        (x2, y2),
                        max(1, self._zi(2)),
                    )
            elif kind == "bush":
                pygame.draw.circle(screen, (58, 96, 64), (x, y), max(1, self._zi(7 * sc)))
                pygame.draw.circle(screen, (74, 118, 78), (x + self._zi(4 * sc), y - self._zi(2)), max(1, self._zi(5 * sc)))
            elif kind == "flowers":
                zt = max(0.001, self.zoom)
                for fx, fy in [(0, 0), (-4, 2), (4, 1)]:
                    ci = (x * 17 + fx * 3 + int(ny * 1000)) % 3
                    col = [(188, 108, 132), (212, 188, 118), (158, 128, 178)][ci]
                    pygame.draw.circle(
                        screen,
                        col,
                        (x + int(fx * zt), y + int(fy * zt)),
                        max(2, self._zi(3 * sc)),
                    )
                pygame.draw.line(screen, (68, 108, 68), (x, y + self._zi(4)), (x, y + self._zi(8 * sc)), max(1, self._zi(2)))
            elif kind == "grass":
                for gx in (-self._zi(3), 0, self._zi(3)):
                    pygame.draw.line(screen, (62, 112, 70), (x + gx, y), (x + gx - 1, y - self._zi(10 * sc)), max(1, self._zi(2)))
            elif kind == "fern":
                pygame.draw.line(screen, (52, 96, 64), (x, y + self._zi(2)), (x - self._zi(8 * sc), y - self._zi(10 * sc)), max(1, self._zi(2)))
                pygame.draw.line(screen, (62, 108, 72), (x, y + self._zi(2)), (x + self._zi(7 * sc), y - self._zi(9 * sc)), max(1, self._zi(2)))
                pygame.draw.line(screen, (56, 100, 66), (x, y + self._zi(2)), (x, y - self._zi(12 * sc)), max(1, self._zi(2)))
            else:
                pygame.draw.ellipse(screen, (98, 96, 90), pygame.Rect(x - self._zi(6 * sc), y - self._zi(4 * sc), self._zi(12 * sc), self._zi(8 * sc)))
                pygame.draw.ellipse(screen, (82, 80, 76), pygame.Rect(x - self._zi(4 * sc), y - self._zi(3 * sc), self._zi(8 * sc), self._zi(6 * sc)))

    def _draw_decorative_farmland(self, screen: pygame.Surface) -> None:
        if not self.engine.agriculture_unlocked:
            return
        wr = self._map_world_rect()
        for nx, ny, sc in self._decor_crops:
            xw = int(wr.left + nx * wr.width)
            yw = int(wr.top + ny * wr.height)
            if self._is_point_over_water(wr, xw, yw):
                continue
            if not self._world_on_screen(float(xw), float(yw), 28):
                continue
            x, y = self._w2s(xw, yw)
            w, h = max(self._zi(10), self._zi(22 * sc)), max(self._zi(6), self._zi(10 * sc))
            pygame.draw.ellipse(screen, (128, 152, 78), pygame.Rect(x - w // 2, y - h // 2, w, h))
            pygame.draw.line(screen, (98, 128, 62), (x - w // 2 + max(1, self._zi(2)), y), (x + w // 2 - max(1, self._zi(2)), y), max(1, self._zi(1)))

    def _draw_scattered_homesteads(self, screen: pygame.Surface) -> None:
        by_region: dict[int, int] = {i: 0 for i in range(self.engine.config.demographics.region_count)}
        for p in self.engine.population:
            if p.alive:
                by_region[p.region_id] = by_region.get(p.region_id, 0) + 1
        for region_id, rpop in by_region.items():
            if rpop <= 0:
                continue
            rect = self._region_rect(region_id)
            n = min(11, max(0, rpop // 20))
            rr = random.Random(self.engine.config.random_seed + region_id * 9973 + self.year * 31)
            for _ in range(n):
                sx = rr.uniform(0.07, 0.93)
                sy = rr.uniform(0.34, 0.92)
                xw = int(rect.left + sx * rect.width)
                yw = int(rect.top + sy * rect.height)
                if not self._world_on_screen(float(xw), float(yw), 24):
                    continue
                x, y = self._w2s(xw, yw)
                o7, o8, o5 = self._zi(7), self._zi(8), self._zi(5)
                pygame.draw.polygon(screen, (132, 102, 74), [(x - o7, y + self._zi(4)), (x, y - o8), (x + o7, y + self._zi(4))])
                pygame.draw.line(screen, (92, 72, 56), (x - o7 + 1, y + self._zi(4)), (x + o7 - 1, y + self._zi(4)), max(1, self._zi(1)))
                pygame.draw.rect(screen, (112, 88, 66), pygame.Rect(x - o5, y + self._zi(2), self._zi(10), self._zi(5)))

    def _draw_livestock_enclosures(self, screen: pygame.Surface) -> None:
        lv_map = getattr(self.engine, "livestock_by_region", {})
        for region_id in range(self.engine.config.demographics.region_count):
            lv = float(lv_map.get(region_id, 0.0))
            if lv < 6.0:
                continue
            rect = self._region_rect(region_id)
            rr = random.Random(self.engine.config.random_seed + region_id * 12011 + 3)
            cxw = int(rect.left + rr.uniform(0.2, 0.8) * rect.width)
            cyw = int(rect.top + rr.uniform(0.48, 0.88) * rect.height)
            if not self._world_on_screen(float(cxw), float(cyw), 80):
                continue
            cx, cy = self._w2s(cxw, cyw)
            w = max(self._zi(28), self._zi(36 + min(40, lv)))
            h = max(self._zi(20), self._zi(24 + min(30, lv * 0.5)))
            pygame.draw.rect(screen, (108, 92, 70), pygame.Rect(cx - w // 2, cy - h // 2, w, h), max(1, self._zi(2)))
            pygame.draw.line(screen, (84, 72, 56), (cx - w // 2, cy), (cx + w // 2, cy), max(1, self._zi(1)))

    def _sync_visual_animals(self) -> None:
        wr = self._map_world_rect()
        wi = float(getattr(self.engine, "wildlife_index", 1.0))
        total_lv = self.engine.total_livestock()
        n_regions = max(1, self.engine.config.demographics.region_count)
        n_bird = int(8 + 16 * min(1.5, wi))
        n_deer = int(2 + int(7 * min(1.3, wi) * 0.55))
        n_sheep = min(42, int(total_lv * 0.14 + 2))
        n_cattle = min(28, int(total_lv * 0.07 + 1))
        target = min(115, n_bird + n_deer + n_sheep + n_cattle)
        rr = random.Random(self.engine.config.random_seed + self.year * 4049 + 17)
        lv_map = dict(getattr(self.engine, "livestock_by_region", {}))
        best_rid = 0
        if lv_map:
            best_rid = max(lv_map.keys(), key=lambda k: float(lv_map.get(k, 0.0)))

        def spawn(species: str, region_id: int) -> VisualAnimal:
            if species == "bird":
                x = rr.uniform(wr.left + 8, wr.right - 8)
                y = rr.uniform(wr.top + wr.height * 0.2, wr.top + wr.height * 0.55)
                sp = 20.0
            else:
                rid = max(0, min(n_regions - 1, region_id))
                rect = self._region_rect(rid)
                x = rr.uniform(rect.left + 6, rect.right - 6)
                y = rr.uniform(rect.top + rect.height * 0.28, rect.bottom - 8)
                sp = 12.0
            ang = rr.uniform(0, math.tau)
            r_id = 0 if species == "bird" else max(0, min(n_regions - 1, region_id))
            livestock = species in ("sheep", "cattle")
            if species == "bird":
                dname = rr.choice(["Shorebird", "Gull", "Lark", "Swallow", "Kite"])
                age_y = rr.randint(1, 9)
            elif species == "deer":
                dname = rr.choice(["Roe deer", "Red deer", "Fallow deer"])
                age_y = rr.randint(2, 14)
            elif species == "sheep":
                age_y = rr.randint(1, 11)
                dname = "Lamb" if age_y < 2 else rr.choice(["Ewe", "Ram", "Sheep"])
            else:
                age_y = rr.randint(2, 15)
                dname = rr.choice(["Cow", "Ox", "Heifer", "Bull"])
            return VisualAnimal(
                x=x,
                y=y,
                vx=math.cos(ang) * sp,
                vy=math.sin(ang) * sp,
                species=species,
                region_id=r_id,
                display_name=dname,
                age_years=age_y,
                is_livestock=livestock,
            )

        new_list: list[VisualAnimal] = []
        for _ in range(n_bird):
            new_list.append(spawn("bird", 0))
        for _ in range(n_deer):
            new_list.append(spawn("deer", rr.randint(0, n_regions - 1)))
        for i in range(n_sheep):
            rid = best_rid if total_lv >= 1.0 else (i % n_regions)
            new_list.append(spawn("sheep", rid))
        for i in range(n_cattle):
            rid = best_rid if total_lv >= 2.0 else ((i * 2) % n_regions)
            new_list.append(spawn("cattle", rid))
        while len(new_list) > target > 0:
            new_list.pop(rr.randint(0, len(new_list) - 1))
        self.visual_animals = new_list if target > 0 else []

    def _move_visual_animals(self, dt: float) -> None:
        wr = self._map_world_rect()
        rr = self.rng
        for a in self.visual_animals:
            if a.species == "bird":
                a.vx += rr.uniform(-40, 40) * dt
                a.vy += rr.uniform(-26, 26) * dt
            else:
                a.vx += rr.uniform(-22, 22) * dt
                a.vy += rr.uniform(-18, 18) * dt
            cap = 55.0 if a.species == "bird" else 38.0
            sp = math.hypot(a.vx, a.vy)
            if sp > cap:
                a.vx *= cap / sp
                a.vy *= cap / sp
            a.x += a.vx * dt
            a.y += a.vy * dt
            if a.species == "bird":
                if a.x < wr.left:
                    a.x = wr.left
                    a.vx *= -0.85
                elif a.x > wr.right:
                    a.x = wr.right
                    a.vx *= -0.85
                if a.y < wr.top:
                    a.y = wr.top
                    a.vy *= -0.85
                elif a.y > wr.top + wr.height * 0.62:
                    a.y = wr.top + wr.height * 0.62
                    a.vy *= -0.85
            else:
                rect = self._region_rect(max(0, min(self.engine.config.demographics.region_count - 1, a.region_id)))
                pad = 10.0
                if a.x < rect.left + pad:
                    a.x = rect.left + pad
                    a.vx *= -0.82
                elif a.x > rect.right - pad:
                    a.x = rect.right - pad
                    a.vx *= -0.82
                if a.y < rect.top + rect.height * 0.22:
                    a.y = rect.top + rect.height * 0.22
                    a.vy *= -0.82
                elif a.y > rect.bottom - pad:
                    a.y = rect.bottom - pad
                    a.vy *= -0.82

    def _draw_animals(self, screen: pygame.Surface) -> None:
        for a in self.visual_animals:
            if not self._world_on_screen(a.x, a.y, 40):
                continue
            x, y = self._w2s(a.x, a.y)
            if a.species == "bird":
                pygame.draw.line(screen, (68, 72, 78), (x - self._zi(4), y), (x, y - self._zi(2)), max(1, self._zi(2)))
                pygame.draw.line(screen, (68, 72, 78), (x, y - self._zi(2)), (x + self._zi(5), y + 1), max(1, self._zi(2)))
            elif a.species == "deer":
                pygame.draw.ellipse(screen, (124, 92, 70), pygame.Rect(x - self._zi(5), y - self._zi(3), self._zi(12), self._zi(7)))
                pygame.draw.circle(screen, (142, 108, 82), (x + self._zi(4), y - self._zi(5)), max(1, self._zi(3)))
                pygame.draw.line(screen, (78, 64, 54), (x - self._zi(2), y - self._zi(8)), (x - self._zi(4), y - self._zi(11)), max(1, self._zi(1)))
                pygame.draw.line(screen, (78, 64, 54), (x + 1, y - self._zi(8)), (x + self._zi(2), y - self._zi(12)), max(1, self._zi(1)))
            elif a.species == "sheep":
                pygame.draw.circle(screen, (226, 224, 232), (x, y), max(2, self._zi(6)))
                pygame.draw.circle(screen, (204, 202, 212), (x - self._zi(3), y - self._zi(2)), max(1, self._zi(4)))
                pygame.draw.circle(screen, (58, 54, 50), (x + self._zi(4), y - 1), max(1, self._zi(2)))
            else:
                pygame.draw.ellipse(screen, (112, 82, 60), pygame.Rect(x - self._zi(9), y - self._zi(5), self._zi(18), self._zi(10)))
                pygame.draw.circle(screen, (78, 62, 50), (x + self._zi(7), y - self._zi(3)), max(1, self._zi(4)))
                pygame.draw.line(screen, (54, 48, 44), (x - self._zi(6), y + self._zi(4)), (x - self._zi(6), y + self._zi(9)), max(1, self._zi(2)))
                pygame.draw.line(screen, (54, 48, 44), (x + self._zi(2), y + self._zi(4)), (x + self._zi(2), y + self._zi(9)), max(1, self._zi(2)))

    def _draw_sky_gradient(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        cycle = (self.year % 240) / 240.0
        day_factor = max(0.22, 1.0 - abs(cycle - 0.5) * 1.65)
        zenith = (
            int(42 + 48 * day_factor),
            int(72 + 72 * day_factor),
            int(118 + 92 * day_factor),
        )
        horizon = (
            int(132 + 72 * day_factor),
            int(168 + 58 * day_factor),
            int(198 + 42 * day_factor),
        )
        haze = (
            int(188 + 32 * day_factor),
            int(182 + 28 * day_factor),
            int(172 + 36 * day_factor),
        )
        height = max(1, rect.height)
        horizon_y = int(rect.height * 0.58)
        step = 2
        for i in range(0, height, step):
            y_world = rect.top + i
            if i < horizon_y:
                t = i / max(1, horizon_y)
                t_ease = t * t * (3.0 - 2.0 * t)
                color = self._lerp_rgb(zenith, horizon, t_ease)
            else:
                t = (i - horizon_y) / max(1, height - horizon_y)
                color = self._lerp_rgb(horizon, haze, min(1.0, t * 1.2))
            if horizon_y <= i < horizon_y + step:
                glow = self._lerp_rgb(horizon, (248, 232, 210), 0.18 * day_factor)
                color = glow
            pygame.draw.line(screen, color, (rect.left, y_world), (rect.right, y_world), step)

    def _draw_sun_and_clouds(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        cycle = (self.year % 240) / 240.0
        pad = max(24, min(80, rect.width // 14))
        parallax = int(self.camera_x * 0.08) % max(1, rect.width // 4)
        span = max(20, rect.width - 2 * pad)
        sun_x = int(rect.left + pad + cycle * span - parallax)
        sun_x = max(rect.left + pad, min(rect.right - pad, sun_x))
        sun_y = int(rect.top + 72 - max(0.0, 1.0 - abs(cycle - 0.5) * 2.0) * min(52, rect.height // 8))
        glow_r = 52
        glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        for r, a in [(48, 14), (38, 28), (28, 48), (18, 72), (10, 100)]:
            pygame.draw.circle(glow, (255, 250, 228, a), (glow_r, glow_r), r)
        blend = getattr(pygame, "BLEND_ALPHA_SDL2", 0)
        screen.blit(glow, (sun_x - glow_r, sun_y - glow_r), special_flags=blend)
        pygame.draw.circle(screen, (252, 244, 210), (sun_x, sun_y), 20)
        pygame.draw.circle(screen, (255, 252, 238), (sun_x - 5, sun_y - 4), 7)
        drift = (self.year * 2.7 + self.frame_counter * 0.02) % (rect.width * 0.4)
        rw = max(80, rect.width)
        cloud_surf = pygame.Surface((rect.width, int(rect.height * 0.42) + 40), pygame.SRCALPHA)
        for ox, oy, sc, r_base in self._cloud_offsets:
            cx = int(ox * rw - drift * 0.15) % (rw + 80) - 20
            cy = int(oy * max(100, rect.height * 0.32))
            alpha = 52
            base = (248, 250, 252, alpha)
            sh = (198, 208, 224, min(78, alpha + 22))
            for dx, dy, rr, col in [
                (0, 7, max(6, int(r_base * sc * 0.45)), sh),
                (0, 0, max(7, int(r_base * sc * 0.5)), base),
                (max(10, int(18 * sc)), -4, max(6, int(r_base * sc * 0.42)), base),
                (max(18, int(34 * sc)), 2, max(5, int(r_base * sc * 0.38)), base),
            ]:
                pygame.draw.circle(cloud_surf, col, (cx + dx, cy + dy), rr)
        screen.blit(cloud_surf, (rect.left, rect.top), special_flags=blend)
        if cycle < 0.12 or cycle > 0.88:
            step = max(72, min(120, rect.width // 7))
            for sx in range(rect.left + 48, rect.right - 48, step):
                pygame.draw.circle(screen, (220, 226, 238), (sx, rect.top + 36 + (sx % 13)), 1)

    def _draw_hills(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        del rect
        layer_colors = [(58, 92, 72), (52, 84, 66), (48, 78, 60)]
        for color, points in zip(layer_colors, self.hills):
            spoly = [self._w2s(px, py) for px, py in points]
            pygame.draw.polygon(screen, color, spoly)

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

    def _draw_river_world(self, screen: pygame.Surface) -> None:
        mw, mh = self._map_dimensions()
        y_base = int(mh * 0.62)
        points: list[tuple[int, int]] = []
        for x in range(0, mw + 1, 14):
            wave = int(16 * math.sin((x + self.year * 2.2) * 0.0095) + 5 * math.sin(x * 0.021))
            points.append((x, y_base + wave))
        if len(points) < 2:
            return
        sp = [self._w2s(float(px), float(py)) for px, py in points]
        b0 = max(2, self._zi(14))
        bank = (64, 108, 78)
        deep = (52, 112, 142)
        water = (88, 148, 182)
        glint = (168, 208, 228)
        for idx in range(len(sp) - 1):
            pygame.draw.line(screen, bank, sp[idx], sp[idx + 1], b0 + max(2, self._zi(5)))
        for idx in range(len(sp) - 1):
            pygame.draw.line(screen, deep, sp[idx], sp[idx + 1], b0)
        for idx in range(len(sp) - 1):
            pygame.draw.line(screen, water, sp[idx], sp[idx + 1], max(2, self._zi(8)))
            pygame.draw.line(screen, glint, sp[idx], sp[idx + 1], max(1, self._zi(3)))
        for idx in range(0, len(sp) - 1, 6):
            hx, hy = sp[idx]
            pygame.draw.circle(screen, (210, 232, 244), (hx, hy - 1), max(1, self._zi(2)))

    def _draw_roads(self, screen: pygame.Surface) -> None:
        structures = [s for s in self.engine.world_structures if s.get("kind") in ("settlement", "school", "workshop", "temple")]
        if not structures:
            return
        settlement = next((s for s in structures if s.get("kind") == "settlement"), None)
        if settlement is None:
            return
        swx, swy = self._structure_world_pos(settlement)
        for s in structures:
            if s is settlement:
                continue
            twx, twy = self._structure_world_pos(s)
            if not self._world_on_screen(float(swx), float(swy), 200) and not self._world_on_screen(
                float(twx), float(twy), 200
            ):
                continue
            sx, sy = self._w2s(swx, swy)
            x, y = self._w2s(twx, twy)
            pygame.draw.line(screen, (148, 132, 102), (sx, sy), (x, y), max(1, self._zi(2)))
            pygame.draw.line(screen, (172, 156, 124), (sx, sy), (x, y), max(1, self._zi(1)))

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
        split = int(panel_rect.height * 0.35)
        pygame.draw.rect(screen, (34, 36, 44), pygame.Rect(0, 0, panel_rect.width, split))
        pygame.draw.rect(screen, (22, 24, 30), pygame.Rect(0, split, panel_rect.width, panel_rect.height - split))
        pygame.draw.rect(screen, self.ui_accent, (panel_rect.right - 4, 0, 4, panel_rect.height))
        pygame.draw.line(screen, (8, 9, 12), (panel_rect.right - 1, 0), (panel_rect.right - 1, self.height), 1)

        title = font.render("World timeline", True, self.panel_text_color)
        screen.blit(title, (18, 18))
        subtitle = small_font.render("Major events & settlements", True, self.ui_text_dim)
        screen.blit(subtitle, (18, 44))

        city_header = small_font.render(
            f"Cities · {len(self.engine.city_summaries)}",
            True,
            (200, 206, 220),
        )
        screen.blit(city_header, (18, 68))
        y_city = 88
        for city in self.engine.city_summaries[:3]:
            line = f"{city['name']} ({city['culture']}/{city['religion']})"
            txt = small_font.render(line[:42], True, (172, 184, 208))
            screen.blit(txt, (20, y_city))
            y_city += 18

        if not self.timeline_cache:
            msg = small_font.render("No major events yet.", True, (130, 136, 150))
            screen.blit(msg, (18, 192))
        else:
            y = 192
            for item in self.timeline_cache[:12]:
                bullet = small_font.render(f"· {item}", True, (208, 212, 222))
                screen.blit(bullet, (18, y))
                y += 22

        ledger_h = min(300, max(100, int(self.height * 0.28)))
        ledger_top = max(120, self.height - ledger_h - max(10, self.height // 80))
        ledger_rect = pygame.Rect(10, ledger_top, max(40, self.left_panel_width - 20), ledger_h)
        pygame.draw.rect(screen, (30, 33, 40), ledger_rect, border_radius=8)
        pygame.draw.rect(screen, (62, 70, 86), ledger_rect, 1, border_radius=8)
        lt = small_font.render("City ledger  ·  wheel / PgUp / PgDn", True, (216, 220, 232))
        screen.blit(lt, (ledger_rect.left + 10, ledger_rect.top + 10))
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

    def _structure_world_pos(self, structure: dict) -> tuple[int, int]:
        region_id = int(structure.get("region_id", 0))
        rect = self._region_rect(region_id)
        slot = float(structure.get("slot", 0.5))
        sy = float(structure.get("slot_y", 0.78))
        sy = max(0.12, min(0.96, sy))
        x = int(rect.left + slot * rect.width)
        y_base = rect.top + sy * rect.height
        kind = structure.get("kind")
        if kind == "field":
            y = int(y_base)
        elif kind == "settlement":
            y = int(y_base - 10)
        elif kind in ("school", "workshop", "temple"):
            y = int(y_base - 14)
        else:
            y = int(y_base - 6)
        return x, y

    def _structure_screen_pos(self, structure: dict) -> tuple[int, int]:
        return self._w2s(*self._structure_world_pos(structure))

    def _draw_world_structures(self, screen: pygame.Surface) -> None:
        for structure in self.engine.world_structures:
            kind = structure.get("kind")
            wx, wy = self._structure_world_pos(structure)
            if not self._world_on_screen(float(wx), float(wy), 100):
                continue
            x, y = self._w2s(wx, wy)
            if kind == "field":
                w, h = self._zi(36), self._zi(20)
                pygame.draw.ellipse(screen, (138, 168, 88), pygame.Rect(x - w // 2, y - h // 2, w, h))
                for off in (-self._zi(5), 0, self._zi(5)):
                    pygame.draw.line(
                        screen,
                        (108, 136, 68),
                        (x - self._zi(14), y + off),
                        (x + self._zi(14), y + off),
                        max(1, self._zi(1)),
                    )
            elif kind == "settlement":
                level = structure.get("level", "camp")
                if level == "camp":
                    t = self._zi(10)
                    pygame.draw.polygon(screen, (148, 112, 80), [(x - t, y + self._zi(3)), (x, y - t), (x + t, y + self._zi(3))])
                elif level == "village":
                    pygame.draw.rect(screen, (142, 118, 94), pygame.Rect(x - self._zi(12), y - self._zi(12), self._zi(24), self._zi(14)))
                elif level == "town":
                    pygame.draw.rect(screen, (136, 124, 118), pygame.Rect(x - self._zi(15), y - self._zi(16), self._zi(30), self._zi(18)))
                    pygame.draw.rect(screen, (116, 106, 102), pygame.Rect(x - self._zi(8), y - self._zi(24), self._zi(16), self._zi(8)))
                else:  # city
                    pygame.draw.rect(screen, (118, 120, 128), pygame.Rect(x - self._zi(16), y - self._zi(20), self._zi(32), self._zi(22)))
                    pygame.draw.rect(screen, (104, 106, 114), pygame.Rect(x - self._zi(4), y - self._zi(34), self._zi(8), self._zi(14)))
            elif kind == "school":
                pygame.draw.rect(screen, (196, 206, 228), pygame.Rect(x - self._zi(10), y - self._zi(14), self._zi(20), self._zi(12)))
                pygame.draw.line(screen, (88, 98, 122), (x - self._zi(8), y - self._zi(3)), (x + self._zi(8), y - self._zi(3)), max(1, self._zi(2)))
            elif kind == "workshop":
                pygame.draw.rect(screen, (158, 160, 168), pygame.Rect(x - self._zi(10), y - self._zi(14), self._zi(20), self._zi(12)))
                pygame.draw.rect(screen, (102, 104, 110), pygame.Rect(x + self._zi(4), y - self._zi(20), self._zi(4), self._zi(8)))
            elif kind == "temple":
                t = self._zi(11)
                pygame.draw.polygon(screen, (176, 152, 198), [(x - t, y - self._zi(2)), (x, y - self._zi(16)), (x + t, y - self._zi(2))])
                pygame.draw.rect(screen, (160, 138, 182), pygame.Rect(x - self._zi(8), y - self._zi(2), self._zi(16), self._zi(10)))

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
                sx, sy = self._structure_world_pos(structure)
                best = (float(sx), float(sy))
                best_score = score
        return best

