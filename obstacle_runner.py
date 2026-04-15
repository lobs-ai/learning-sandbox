#!/usr/bin/env python3
"""
Obstacle Runner — 3D RL Agent
A cube learns to navigate obstacle courses using DQN reinforcement learning.
Custom physics + pyglet rendering. No external physics engine needed.
"""

import math
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GRAVITY = -25.0
JUMP_FORCE = 10.0
MOVE_SPEED = 8.0
TERMINAL_VELOCITY = -50.0
AGENT_SIZE = 0.4

GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
REPLAY_SIZE = 10000
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.998

MAX_STEPS = 500

device = torch.device("cpu")


class DQNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_SIZE):
        self.buf = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))
        if len(self.buf) > self.capacity:
            self.buf.pop(0)

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buf, min(len(self.buf), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


LEVELS = [
    {
        "name": "First Steps",
        "platforms": [
            {"x": 0, "y": 0, "z": 0, "w": 6, "h": 1, "d": 3},
            {"x": 8, "y": 0, "z": 0, "w": 4, "h": 1, "d": 3},
        ],
        "goal": {"x": 10, "y": 1, "z": 0},
        "spawn": {"x": 0, "y": 2, "z": 0},
    },
    {
        "name": "The Hop",
        "platforms": [
            {"x": 0, "y": 0, "z": 0, "w": 4, "h": 1, "d": 3},
            {"x": 5, "y": 1, "z": 0, "w": 3, "h": 1, "d": 3},
            {"x": 9, "y": 2, "z": 0, "w": 3, "h": 1, "d": 3},
            {"x": 13, "y": 2, "z": 0, "w": 3, "h": 1, "d": 3},
        ],
        "goal": {"x": 14.5, "y": 3, "z": 0},
        "spawn": {"x": 0, "y": 2, "z": 0},
    },
    {
        "name": "Narrow Path",
        "platforms": [
            {"x": 0, "y": 0, "z": 0, "w": 4, "h": 1, "d": 3},
            {"x": 6, "y": 0, "z": 0, "w": 1.5, "h": 1, "d": 1.5},
            {"x": 10, "y": 0, "z": 0, "w": 1.5, "h": 1, "d": 1.5},
            {"x": 14, "y": 0, "z": 0, "w": 4, "h": 1, "d": 3},
        ],
        "goal": {"x": 16, "y": 1, "z": 0},
        "spawn": {"x": 0, "y": 2, "z": 0},
    },
    {
        "name": "The Climb",
        "platforms": [
            {"x": 0, "y": 0, "z": 0, "w": 3, "h": 1, "d": 3},
            {"x": 4, "y": 0.8, "z": 0, "w": 3, "h": 1, "d": 3},
            {"x": 8, "y": 1.6, "z": 0, "w": 3, "h": 1, "d": 3},
            {"x": 12, "y": 2.4, "z": 0, "w": 3, "h": 1, "d": 3},
            {"x": 16, "y": 3.2, "z": 0, "w": 4, "h": 1, "d": 3},
        ],
        "goal": {"x": 18, "y": 4.2, "z": 0},
        "spawn": {"x": 0, "y": 2, "z": 0},
    },
    {
        "name": "Moving Target",
        "platforms": [
            {"x": 0, "y": 0, "z": 0, "w": 4, "h": 1, "d": 3},
            {"x": 8, "y": 0, "z": 0, "w": 5, "h": 1, "d": 3},
            {"x": 15, "y": 0, "z": 0, "w": 4, "h": 1, "d": 3},
        ],
        "goal": {"x": 17, "y": 1, "z": 0},
        "spawn": {"x": 0, "y": 2, "z": 0},
        "obstacles": [
            {
                "x": 8,
                "y": 1.5,
                "z": 0,
                "w": 1,
                "h": 1,
                "d": 2,
                "axis": "z",
                "range": 1.0,
                "speed": 2.0,
            },
        ],
    },
]


class Agent:
    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.on_ground = False
        self.last_x = self.last_z = 0.0

    def reset(self, spawn):
        self.x = spawn["x"]
        self.y = spawn["y"]
        self.z = spawn["z"]
        self.vx = self.vy = self.vz = 0.0
        self.on_ground = False
        self.last_x = self.x
        self.last_z = self.z

    def get_position(self):
        return self.x, self.y, self.z

    def get_velocity(self):
        return self.vx, self.vy, self.vz

    def apply_action(self, action):
        if action == 0:
            self.vx = MOVE_SPEED
        elif action == 1:
            self.vx = -MOVE_SPEED
        elif action == 2:
            self.vz = -MOVE_SPEED
        elif action == 3:
            self.vz = MOVE_SPEED
        elif action == 4:
            if self.on_ground:
                self.vy = JUMP_FORCE
                self.on_ground = False
        elif action == 5:
            if self.on_ground:
                self.vy = JUMP_FORCE
                self.vx = MOVE_SPEED
                self.on_ground = False
            else:
                self.vx = MOVE_SPEED

    def perceive(self, goal, obstacles):
        vx, vy, vz = self.get_velocity()
        x, y, z = self.get_position()
        vel = np.array([vx / 10.0, vy / 15.0, vz / 10.0], dtype=np.float32)
        dx = goal["x"] - x
        dz = goal["z"] - z
        dist = math.sqrt(dx**2 + dz**2) + 1e-6
        dir_goal = np.array([dx / dist, 0.0, dz / dist], dtype=np.float32)
        ground = np.array([1.0 if self.on_ground else 0.0], dtype=np.float32)
        nearest_d = float("inf")
        nearest_r = np.zeros(3, dtype=np.float32)
        for o in obstacles:
            d = math.sqrt((x - o[0]) ** 2 + (z - o[2]) ** 2)
            if d < nearest_d:
                nearest_d = d
                nearest_r = np.array(
                    [(o[0] - x) / 10.0, (o[1] - y) / 5.0, (o[2] - z) / 10.0],
                    dtype=np.float32,
                )
        nd_norm = np.array([min(nearest_d / 10.0, 1.0)], dtype=np.float32)
        t_norm = np.array([0.0], dtype=np.float32)
        return np.concatenate(
            [vel, dir_goal, ground, nearest_r, nd_norm, t_norm]
        ).astype(np.float32)

    def physics_step(self, dt, platforms, goal_pos, obstacles):
        self.vy += GRAVITY * dt
        self.vy = max(self.vy, TERMINAL_VELOCITY)

        nx = self.x + self.vx * dt
        ny = self.y + self.vy * dt
        nz = self.z + self.vz * dt

        self._resolve_x(nx, platforms)
        self._resolve_y(ny, platforms)
        self._resolve_z(nz, platforms)

        self.vx *= 0.85
        self.vz *= 0.85

        if self.y < -10:
            return -50.0, True

        dist = math.sqrt(
            (self.x - goal_pos[0]) ** 2
            + (self.y - goal_pos[1]) ** 2
            + (self.z - goal_pos[2]) ** 2
        )
        if dist < 1.5:
            return 100.0, True

        db = math.sqrt(
            (self.last_x - goal_pos[0]) ** 2 + (self.last_z - goal_pos[2]) ** 2
        )
        da = math.sqrt((self.x - goal_pos[0]) ** 2 + (self.z - goal_pos[2]) ** 2)
        self.last_x = self.x
        self.last_z = self.z

        return (db - da) * 10.0 - 0.1, False

    def _resolve_x(self, new_x, platforms):
        old_x = self.x
        self.x = new_x
        for p in platforms:
            if self._collides(p):
                self.x = old_x
                self.vx = 0
                return

    def _resolve_y(self, new_y, platforms):
        old_y = self.y
        self.y = new_y
        self.on_ground = False
        for p in platforms:
            if self._collides(p):
                self.y = old_y
                if self.vy < 0:
                    self.y = p["y"] + p["h"] / 2.0 + AGENT_SIZE
                    self.vy = 0
                    self.on_ground = True
                else:
                    self.vy = 0
                return

    def _resolve_z(self, new_z, platforms):
        old_z = self.z
        self.z = new_z
        for p in platforms:
            if self._collides(p):
                self.z = old_z
                self.vz = 0
                return

    def _collides(self, p):
        hw = p["w"] / 2.0 + AGENT_SIZE
        hh = p["h"] / 2.0 + AGENT_SIZE
        hd = p["d"] / 2.0 + AGENT_SIZE
        return (
            abs(self.x - p["x"]) < hw
            and abs(self.y - p["y"]) < hh
            and abs(self.z - p["z"]) < hd
        )


class Camera:
    def __init__(self):
        self.yaw = -45.0
        self.pitch = -25.0
        self.dist = 20.0
        self.target = [8.0, 0.0, 0.0]

    def project(self, x, y, z, width, height):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)

        dx = x - self.target[0]
        dy = y - self.target[1]
        dz = z - self.target[2]

        rx = dx * math.cos(yaw_rad) - dz * math.sin(yaw_rad)
        ry = dy * math.cos(pitch_rad)
        rz = dx * math.sin(yaw_rad) + dz * math.cos(yaw_rad)

        dist = math.sqrt(rx**2 + ry**2 + rz**2) + 1e-6
        fx = rx / dist
        fy = ry / dist
        fz = rz / dist

        fov = 60.0
        aspect = width / height
        tan_hfov = math.tan(math.radians(fov) / 2.0)

        ndc_x = fx / (fz * tan_hfov * aspect)
        ndc_y = fy / (fz * tan_hfov)

        screen_x = (ndc_x + 1.0) * 0.5 * width
        screen_y = (1.0 - ndc_y) * 0.5 * height

        return screen_x, screen_y, fz


def select_action(state, q_net, epsilon):
    if random.random() < epsilon:
        return random.randrange(6)
    with torch.no_grad():
        return int(
            q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            .squeeze()
            .argmax()
            .item()
        )


def train_headless(episodes=200):
    q_net = DQNet().to(device)
    target_net = DQNet().to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    replay = ReplayBuffer()

    agent = Agent()
    epsilon = EPSILON_START
    wins = 0

    for ep in range(episodes):
        lvl = LEVELS[ep % len(LEVELS)]
        agent.reset(lvl["spawn"])

        state = agent.perceive(lvl["goal"], [])
        done = False
        steps = 0
        ep_reward = 0.0

        while not done and steps < MAX_STEPS:
            action = select_action(state, q_net, epsilon)
            agent.apply_action(action)
            reward, done = agent.physics_step(
                0.02,
                lvl["platforms"],
                (lvl["goal"]["x"], lvl["goal"]["y"], lvl["goal"]["z"]),
                [],
            )
            ep_reward += reward

            next_state = agent.perceive(lvl["goal"], [])
            replay.push(state, action, reward, next_state, float(done))
            state = next_state

            if len(replay) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay.sample()
                s_t = torch.tensor(states)
                a_t = torch.tensor(actions)
                r_t = torch.tensor(rewards)
                ns_t = torch.tensor(next_states)
                d_t = torch.tensor(dones)
                q_sa = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_target = r_t + GAMMA * (1 - d_t) * target_net(ns_t).max(1)[0]
                loss = loss_fn(q_sa, q_target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()

            steps += 1

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        if ep_reward > 50:
            wins += 1

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if ep % 20 == 0:
            print(
                f"Episode {ep}: steps={steps}, reward={ep_reward:.1f}, wins={wins}/{ep + 1} ({wins / (ep + 1) * 100:.0f}%), epsilon={epsilon:.1%}"
            )

    return wins, episodes


def run_gui():
    import pyglet
    from pyglet.gl import glClearColor
    from pyglet import graphics, text, clock

    class Renderer(pyglet.window.Window):
        def __init__(self, width=1280, height=720):
            super().__init__(width, height, caption="Obstacle Runner", resizable=False)
            self.width = width
            self.height = height
            self.camera = Camera()
            self.camera.target = [8.0, 0.0, 0.0]
            glClearColor(15 / 255, 15 / 255, 25 / 255, 1.0)
            self.level_label = text.Label(
                "",
                font_name="Arial",
                font_size=18,
                x=20,
                y=height - 30,
                color=(255, 255, 255, 255),
            )
            self.stats_label = text.Label(
                "",
                font_name="Arial",
                font_size=14,
                x=20,
                y=height - 60,
                color=(200, 200, 200, 255),
            )
            self.loss_label = text.Label(
                "",
                font_name="Arial",
                font_size=14,
                x=20,
                y=height - 85,
                color=(200, 200, 200, 255),
            )
            self.help_label = text.Label(
                "[R] Reset  [N] Next Level  [Arrow Keys] Rotate  [W/S] Zoom  [ESC] Quit",
                font_name="Arial",
                font_size=12,
                x=20,
                y=20,
                color=(150, 150, 150, 255),
            )
            self.platforms_draw = []
            self.obstacles_draw = []
            self.agent_pos = (0.0, 2.0, 0.0)
            self.goal_pos = (10.0, 1.0, 0.0)
            self.agent_color = (220, 220, 240)
            self.goal_color = (50, 200, 100)

        def project_box(self, cx, cy, cz, w, h, d):
            hw, hh, hd = w / 2.0, h / 2.0, d / 2.0
            corners = [
                (cx - hw, cy - hh, cz - hd),
                (cx + hw, cy - hh, cz - hd),
                (cx + hw, cy + hh, cz - hd),
                (cx - hw, cy + hh, cz - hd),
                (cx - hw, cy - hh, cz + hd),
                (cx + hw, cy - hh, cz + hd),
                (cx + hw, cy + hh, cz + hd),
                (cx - hw, cy + hh, cz + hd),
            ]
            projected = [
                self.camera.project(p[0], p[1], p[2], self.width, self.height)
                for p in corners
            ]
            pts2d = [(p[0], p[1]) for p in projected]
            depths = [p[2] for p in projected]
            return pts2d, depths

        def draw_box(self, cx, cy, cz, w, h, d, color):
            pts2d, depths = self.project_box(cx, cy, cz, w, h, d)
            face_defs = [
                (0, 1, 2, 3, [c * 0.7 for c in color]),
                (4, 5, 6, 7, [c * 0.9 for c in color]),
                (0, 1, 5, 4, [c * 0.6 for c in color]),
                (2, 3, 7, 6, [c * 0.8 for c in color]),
                (1, 2, 6, 5, [c * 0.5 for c in color]),
                (0, 3, 7, 4, [c * 0.85 for c in color]),
            ]
            face_defs.sort(key=lambda f: (depths[f[0]] + depths[f[2]]) / 2.0)
            for i0, i1, i2, i3, col in face_defs:
                r, g, b = int(col[0]), int(col[1]), int(col[2])
                cr, cg, cb = r / 255.0, g / 255.0, b / 255.0
                px0, py0 = pts2d[i0]
                px1, py1 = pts2d[i1]
                px2, py2 = pts2d[i2]
                px3, py3 = pts2d[i3]
                pos_data = (px0, py0, 0.0, px1, py1, 0.0, px2, py2, 0.0, px3, py3, 0.0)
                color_data = (
                    cr,
                    cg,
                    cb,
                    1.0,
                    cr,
                    cg,
                    cb,
                    1.0,
                    cr,
                    cg,
                    cb,
                    1.0,
                    cr,
                    cg,
                    cb,
                    1.0,
                )
                graphics.draw(
                    4,
                    pyglet.gl.GL_TRIANGLE_FAN,
                    position=("f", pos_data),
                    colors=("f", color_data),
                )

        def update_labels(self, level_name, episode, wins, epsilon, steps, loss):
            self.level_label.text = f"Level {level_name}"
            self.stats_label.text = f"Episode: {episode} | Wins: {wins} | Eps: {epsilon:.0%} | Steps: {steps}"
            self.loss_label.text = f"Loss: {loss:.4f}"

        def on_draw(self):
            self.clear()
            for bx, by, bz, bw, bh, bd, col in self.platforms_draw:
                self.draw_box(bx, by, bz, bw, bh, bd, col)
            for bx, by, bz, bw, bh, bd, col in self.obstacles_draw:
                self.draw_box(bx, by, bz, bw, bh, bd, col)
            gx, gy, gz = self.goal_pos
            self.draw_box(gx, gy, gz, 2.0, 0.3, 2.0, self.goal_color)
            ax, ay, az = self.agent_pos
            self.draw_box(
                ax,
                ay,
                az,
                AGENT_SIZE * 2,
                AGENT_SIZE * 2,
                AGENT_SIZE * 2,
                self.agent_color,
            )
            self.level_label.draw()
            self.stats_label.draw()
            self.loss_label.draw()
            self.help_label.draw()

    class Game:
        def __init__(self, window):
            self.window = window
            self.window.push_handlers(self)
            self.q_net = DQNet().to(device)
            self.target_net = DQNet().to(device)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
            self.loss_fn = nn.MSELoss()
            self.replay = ReplayBuffer()
            self.current_level = 0
            self.episode = 0
            self.wins = 0
            self.epsilon = EPSILON_START
            self.gen = 0
            self.total_loss = 0.0
            self.loss_count = 0
            self.steps = 0
            self.last_state = None
            self.last_action = None
            self.episode_reward = 0.0
            self.agent = Agent()
            self.moving_obstacles = []
            self._keys_held = set()
            self._elapsed = 0.0
            self._elapsed = 0.0
            self._build_level(0)
            clock.schedule_interval(self.update, 1.0 / 60.0)

        def _build_level(self, idx):
            lvl = LEVELS[idx]
            self.agent.reset(lvl["spawn"])
            self.window.camera.target = [
                (lvl["platforms"][0]["x"] + lvl["platforms"][-1]["x"]) / 2.0,
                0.0,
                0.0,
            ]
            self.window.platforms_draw = [
                (p["x"], p["y"], p["z"], p["w"], p["h"], p["d"], [40, 40, 55])
                for p in lvl["platforms"]
            ]
            self.window.obstacles_draw = []
            self.moving_obstacles = []
            for obs in lvl.get("obstacles", []):
                self.moving_obstacles.append(
                    {
                        "x": obs["x"],
                        "y": obs["y"],
                        "z": obs["z"],
                        "ox": obs["x"],
                        "oy": obs["y"],
                        "oz": obs["z"],
                        "w": obs["w"],
                        "h": obs["h"],
                        "d": obs["d"],
                        "axis": obs.get("axis", "z"),
                        "range": obs.get("range", 1.0),
                        "speed": obs.get("speed", 2.0),
                    }
                )
                self.window.obstacles_draw.append(
                    (
                        obs["x"],
                        obs["y"],
                        obs["z"],
                        obs["w"],
                        obs["h"],
                        obs["d"],
                        [200, 80, 60],
                    )
                )
            self.window.goal_pos = (
                lvl["goal"]["x"],
                lvl["goal"]["y"],
                lvl["goal"]["z"],
            )

        def _reset_episode(self):
            self.agent.reset(LEVELS[self.current_level]["spawn"])

        def _do_step(self, dt):
            lvl = LEVELS[self.current_level]
            self._elapsed += dt
            t = self._elapsed
            for obs in self.moving_obstacles:
                offset = math.sin(t * obs["speed"]) * obs["range"]
                if obs["axis"] == "z":
                    obs["z"] = obs["oz"] + offset
                elif obs["axis"] == "x":
                    obs["x"] = obs["ox"] + offset
            for i, obs in enumerate(self.moving_obstacles):
                self.window.obstacles_draw[i] = (
                    obs["x"],
                    obs["y"],
                    obs["z"],
                    obs["w"],
                    obs["h"],
                    obs["d"],
                    [200, 80, 60],
                )
            obstacles_pos = [(o["x"], o["y"], o["z"]) for o in self.moving_obstacles]
            action = 0
            if self.last_state is not None:
                action = select_action(self.last_state, self.q_net, self.epsilon)
                self.agent.apply_action(action)
            reward = 0.0
            done = False
            reward, done = self.agent.physics_step(
                dt,
                lvl["platforms"],
                (lvl["goal"]["x"], lvl["goal"]["y"], lvl["goal"]["z"]),
                obstacles_pos,
            )
            pos = self.agent.get_position()
            self.window.agent_pos = pos
            self.agent.last_x = pos[0]
            self.agent.last_z = pos[2]
            state = self.agent.perceive(lvl["goal"], obstacles_pos)
            if self.last_state is not None:
                self.replay.push(
                    self.last_state, self.last_action, reward, state, float(done)
                )
                self.episode_reward += reward
                if len(self.replay) >= BATCH_SIZE:
                    batch = self.replay.sample()
                    s_t = torch.tensor(batch[0])
                    a_t = torch.tensor(batch[1])
                    r_t = torch.tensor(batch[2])
                    ns_t = torch.tensor(batch[3])
                    d_t = torch.tensor(batch[4])
                    q_sa = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze()
                    with torch.no_grad():
                        q_target = (
                            r_t + GAMMA * (1 - d_t) * self.target_net(ns_t).max(1)[0]
                        )
                    loss = self.loss_fn(q_sa, q_target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                    self.optimizer.step()
                    self.total_loss += loss.item()
                    self.loss_count += 1
            self.last_state = state
            self.last_action = action
            self.steps += 1
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
            if done or self.steps >= MAX_STEPS:
                self.gen += 1
                self.episode += 1
                if self.episode_reward > 50 or done:
                    self.wins += 1
                if self.gen % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())
                self._reset_episode()
                self.steps = 0
                self.episode_reward = 0.0
                self.last_state = None
                self.last_action = None
            loss_avg = self.total_loss / max(self.loss_count, 1)
            self.window.update_labels(
                f"{self.current_level + 1}: {lvl['name']}",
                self.episode,
                self.wins,
                self.epsilon,
                self.steps,
                loss_avg,
            )

        def update(self, dt):
            from pyglet.window import key

            if key.LEFT in self._keys_held:
                self.window.camera.yaw -= 2.0
            if key.RIGHT in self._keys_held:
                self.window.camera.yaw += 2.0
            if key.UP in self._keys_held:
                self.window.camera.pitch = max(-89.0, self.window.camera.pitch - 2.0)
            if key.DOWN in self._keys_held:
                self.window.camera.pitch = min(-1.0, self.window.camera.pitch + 2.0)
            if key.W in self._keys_held:
                self.window.camera.dist = max(5.0, self.window.camera.dist - 0.5)
            if key.S in self._keys_held:
                self.window.camera.dist = min(40.0, self.window.camera.dist + 0.5)
            self._do_step(dt)

        def on_key_press(self, symbol, modifiers):
            from pyglet.window import key

            self._keys_held = getattr(self, "_keys_held", set())
            self._keys_held.add(symbol)
            if symbol == key.ESCAPE:
                pyglet.app.exit()
            elif symbol == key.R:
                self._reset_episode()
                self.steps = 0
                self.episode_reward = 0.0
                self.last_state = None
                self.last_action = None
            elif symbol == key.N:
                self.current_level = (self.current_level + 1) % len(LEVELS)
                self._build_level(self.current_level)
                self.episode = 0
                self.wins = 0
                self.epsilon = EPSILON_START
                self.steps = 0
                self.episode_reward = 0.0
                self.last_state = None
                self.last_action = None
                self.gen = 0
                self.total_loss = 0.0
                self.loss_count = 0

        def on_key_release(self, symbol, modifiers):
            from pyglet.window import key

            self._keys_held = getattr(self, "_keys_held", set())
            self._keys_held.discard(symbol)

    window = Renderer(1280, 720)
    game = Game(window)
    pyglet.app.run()


if __name__ == "__main__":
    headless = len(sys.argv) > 1 and sys.argv[1] == "--headless"
    episodes = (
        int(sys.argv[-1]) if len(sys.argv) > 1 and sys.argv[-1].isdigit() else 200
    )

    if headless:
        wins, total = train_headless(episodes)
        print(f"\nFinal: {wins}/{total} wins ({wins / total * 100:.0f}%)")
    else:
        run_gui()
