from enum import Enum

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class Phase(Enum):
    NS = 0
    EW = 1


class Action(Enum):
    KEEP = 0
    SWITCH = 1


class TrafficIntersectionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        self.dt = 10  # seconds
        # startup lost time due to red light clearance + drivers reaction time
        self.startup_lost_time = 4.0  # seconds
        self.upstream_cycle_time = 120  # seconds

        self.saturation_flow_rate = 0.5  # cars/seconds
        self.base_arrival_rate = 0.1  # cars/seconds
        self.platoon_arrival_rate = 0.4  # cars/seconds

        minimum_headway = 1.2  # seconds 
        self.step_capacity = int(self.dt / minimum_headway)  # cars per timestep

        self.max_episode_time = 3600  # seconds (1 hour)

        # [N, E, S, W, phase, timesteps in phase]
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.array([np.inf] * 4 + [1, np.inf], dtype=np.float32),
        )
        
        # [keep, switch]
        self.action_space = spaces.Discrete(2)

        # [N, E, S, W]
        self.queues = np.zeros(4, dtype=np.float32)
        self.current_phase = Phase.NS.value
        self.timesteps_in_current_phase = 0
        self.global_time = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 600
        self.window = None
        self.clock = None

        # Visual Constants
        self.road_width = 100
        self.lane_width = self.road_width // 2
        self.center = self.window_size // 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.queues = self.np_random.integers(low=0, high=10, size=4).astype(np.float32)
        self.current_phase = self.np_random.choice([Phase.NS.value, Phase.EW.value])
        self.timesteps_in_current_phase = 0
        self.global_time = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        effective_green_time = self.dt

        if Action(action) == Action.SWITCH:
            self.current_phase = 1 - self.current_phase
            self.timesteps_in_current_phase = 0
            effective_green_time -= self.np_random.normal(self.startup_lost_time, 0.5)
    
        self.timesteps_in_current_phase += 1

        max_discharge = int(effective_green_time * self.saturation_flow_rate)

        if Phase(self.current_phase) == Phase.NS:
            active_lanes = np.array([1, 0, 1, 0], dtype=bool)
        else:
            active_lanes = np.array([0, 1, 0, 1], dtype=bool)

        leaving_cars = np.zeros(4)
        leaving_cars[active_lanes] = np.minimum(self.queues[active_lanes], max_discharge)

        self.queues -= leaving_cars

        ns_rate, ew_rate = self._get_arrival_rates()

        arrivals = self.np_random.normal(
            np.array([ns_rate, ew_rate, ns_rate, ew_rate]) * self.dt, size=4
        ).round()
        arrivals = np.clip(arrivals, 0, self.step_capacity)

        self.queues += arrivals

        self.global_time += self.dt

        state = self._get_obs()
        reward = self._compute_reward()
        terminated = False
        truncated = self.global_time >= self.max_episode_time

        info = {
            "arrivals": arrivals,
            "departures": leaving_cars,
            "total_queue": float(self.queues.sum()),
            "ns_rate": round(ns_rate, 2),
            "ew_rate": round(ew_rate, 2)
        }

        if self.render_mode == "human":
            self._render_frame()

        return state, reward, terminated, truncated, info

    def _get_obs(self):
        obs = np.concatenate([self.queues, [self.current_phase, self.timesteps_in_current_phase]])
        return obs.astype(np.float32)

    def _compute_reward(self):
        return -self.queues.sum()

    def _get_arrival_rates(self):
        cycle_pos = (self.global_time % self.upstream_cycle_time) / self.upstream_cycle_time
        cycle_rad = cycle_pos * 2 * np.pi

        ns_intensity = max(0, np.sin(cycle_rad))
        ew_intensity = max(0, np.sin(cycle_rad + np.pi))

        ns_rate = self.base_arrival_rate + ns_intensity * (self.platoon_arrival_rate - self.base_arrival_rate)
        ew_rate = self.base_arrival_rate + ew_intensity * (self.platoon_arrival_rate - self.base_arrival_rate)

        return ns_rate, ew_rate

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((30, 30, 30))  # Dark Asphalt Background

        # 1. Draw Roads and Dividers
        # Vertical Road (NS)
        pygame.draw.rect(
            canvas,
            (80, 80, 80),
            (self.center - self.lane_width, 0, self.road_width, self.window_size),
        )
        # Horizontal Road (EW)
        pygame.draw.rect(
            canvas,
            (80, 80, 80),
            (0, self.center - self.lane_width, self.window_size, self.road_width),
        )

        # Center dividers (Yellow lines)
        pygame.draw.line(
            canvas,
            (150, 150, 150),
            (self.center, 0),
            (self.center, self.window_size),
            2,
        )
        pygame.draw.line(
            canvas,
            (150, 150, 150),
            (0, self.center),
            (self.window_size, self.center),
            2,
        )

        # Intersection box clearing (remove dividers in center)
        pygame.draw.rect(
            canvas,
            (80, 80, 80),
            (
                self.center - self.lane_width,
                self.center - self.lane_width,
                self.road_width,
                self.road_width,
            ),
        )

        # 2. Draw Traffic Lights (Signals on lane dividers)
        # Colors
        red = (255, 50, 50)
        green = (50, 255, 50)
        ns_color = green if self.current_phase == Phase.NS.value else red
        ew_color = green if self.current_phase == Phase.EW.value else red

        light_radius = 10
        # Offset slightly back from the intersection box so cars don't overlap the light
        stop_offset = self.lane_width + 5

        # N Light (Controls top-right lane, positioned on center divider above intersection)
        pygame.draw.circle(
            canvas, ns_color, (self.center, self.center - stop_offset), light_radius
        )

        # S Light (Controls bottom-left lane, positioned on center divider below intersection)
        pygame.draw.circle(
            canvas, ns_color, (self.center, self.center + stop_offset), light_radius
        )

        # E Light (Controls bottom-right lane, positioned on center divider right of intersection)
        pygame.draw.circle(
            canvas, ew_color, (self.center + stop_offset, self.center), light_radius
        )

        # W Light (Controls top-left lane, positioned on center divider left of intersection)
        pygame.draw.circle(
            canvas, ew_color, (self.center - stop_offset, self.center), light_radius
        )

        # 3. Draw Queues (Cars)
        car_size = 10
        car_gap = 3

        # Helper to draw a single car
        def draw_car(x, y, vertical=True):
            color = (224, 215, 34)
            if vertical:
                rect = (x - car_size // 2, y, car_size, car_size)
            else:
                rect = (x, y - car_size // 2, car_size, car_size)
            pygame.draw.rect(canvas, color, rect)

        # Offset for where cars start queueing (behind the traffic light)
        queue_start_offset = stop_offset + light_radius + car_gap

        # Draw N Queue (Top road, incoming on right side)
        q_n = int(self.queues[0])
        for i in range(q_n):
            pos_y = (
                (self.center - queue_start_offset)
                - (i * (car_size + car_gap))
                - car_size
            )
            draw_car(self.center + self.lane_width // 2, pos_y, vertical=True)

        # Draw E Queue (Right road, incoming on bottom side)
        q_e = int(self.queues[1])
        for i in range(q_e):
            pos_x = (
                (self.center + queue_start_offset)
                + (i * (car_size + car_gap))
                + car_gap
            )
            draw_car(pos_x, self.center + self.lane_width // 2, vertical=False)

        # Draw S Queue (Bottom road, incoming on left side)
        q_s = int(self.queues[2])
        for i in range(q_s):
            pos_y = (
                (self.center + queue_start_offset)
                + (i * (car_size + car_gap))
                + car_gap
            )
            draw_car(self.center - self.lane_width // 2, pos_y, vertical=True)

        # Draw W Queue (Left road, incoming on top side)
        q_w = int(self.queues[3])
        for i in range(q_w):
            pos_x = (
                (self.center - queue_start_offset)
                - (i * (car_size + car_gap))
                - car_size
            )
            draw_car(pos_x, self.center - self.lane_width // 2, vertical=False)

        # 4. Draw Info Text
        font = pygame.font.SysFont("Arial", 16, bold=True)

        total_cars = int(self.queues.sum())

        header_str = f"Total Cars: {total_cars}"
        img = font.render(header_str, True, (255, 255, 255))
        # Draw a background box for the text for readability
        bg_rect = img.get_rect(topleft=(10, 10))
        pygame.draw.rect(canvas, (30, 30, 30), bg_rect)
        canvas.blit(img, (10, 10))

        def draw_q_count(queue, count, x, y):
            txt = font.render(f"{queue}: {int(count)}", True, (255, 255, 255))
            # Draw a small black box behind for contrast
            bg = txt.get_rect(center=(x, y))
            pygame.draw.rect(canvas, (30, 30, 30), bg)
            canvas.blit(txt, bg)

        # Offsets for text placement relative to center
        offset = self.lane_width + 27

        draw_q_count("N", q_n, self.center + offset, self.center - offset)  # N
        draw_q_count("E", q_e, self.center + offset, self.center + offset)  # E
        draw_q_count("S", q_s, self.center - offset, self.center + offset)  # S
        draw_q_count("W", q_w, self.center - offset, self.center - offset)  # W

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # Cap framerate at 4fps for readability of steps
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
