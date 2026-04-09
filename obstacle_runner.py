#!/usr/bin/env python3
"""
Obstacle Runner — 3D RL Agent
A cube learns to navigate obstacle courses.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- Constants ---
GRAVITY = -25.0
JUMP_FORCE = 10.0
MOVE_SPEED = 5.0
TERMINAL_VELOCITY = -50.0

# RL Constants
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
REPLAY_SIZE = 10000
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.998

N_EPISODES = 1000
MAX_STEPS = 500

device = torch.device("cpu")


# --- DQN ---
class DQNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
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
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# --- Levels ---
LEVELS = [
    {
        'name': "First Steps",
        'platforms': [
            {'x': 0, 'y': 0, 'z': 0, 'w': 6, 'h': 1, 'd': 3},
            {'x': 8, 'y': 0, 'z': 0, 'w': 4, 'h': 1, 'd': 3},
        ],
        'goal': {'x': 10, 'y': 1, 'z': 0},
        'spawn': {'x': 0, 'y': 2, 'z': 0},
    },
    {
        'name': "The Hop",
        'platforms': [
            {'x': 0,  'y': 0,  'z': 0, 'w': 4, 'h': 1, 'd': 3},
            {'x': 5,  'y': 1,  'z': 0, 'w': 3, 'h': 1, 'd': 3},
            {'x': 9,  'y': 2,  'z': 0, 'w': 3, 'h': 1, 'd': 3},
            {'x': 13, 'y': 2,  'z': 0, 'w': 3, 'h': 1, 'd': 3},
        ],
        'goal': {'x': 14.5, 'y': 3, 'z': 0},
        'spawn': {'x': 0, 'y': 2, 'z': 0},
    },
    {
        'name': "Narrow Path",
        'platforms': [
            {'x': 0,  'y': 0, 'z': 0,  'w': 4,  'h': 1, 'd': 3},
            {'x': 6,  'y': 0, 'z': 0,  'w': 1.5, 'h': 1, 'd': 1.5},
            {'x': 10, 'y': 0, 'z': 0,  'w': 1.5, 'h': 1, 'd': 1.5},
            {'x': 14, 'y': 0, 'z': 0,  'w': 4,  'h': 1, 'd': 3},
        ],
        'goal': {'x': 16, 'y': 1, 'z': 0},
        'spawn': {'x': 0, 'y': 2, 'z': 0},
    },
    {
        'name': "The Climb",
        'platforms': [
            {'x': 0,  'y': 0,   'z': 0, 'w': 3, 'h': 1, 'd': 3},
            {'x': 4,  'y': 0.8, 'z': 0, 'w': 3, 'h': 1, 'd': 3},
            {'x': 8,  'y': 1.6, 'z': 0, 'w': 3, 'h': 1, 'd': 3},
            {'x': 12, 'y': 2.4, 'z': 0, 'w': 3, 'h': 1, 'd': 3},
            {'x': 16, 'y': 3.2, 'z': 0, 'w': 4, 'h': 1, 'd': 3},
        ],
        'goal': {'x': 18, 'y': 4.2, 'z': 0},
        'spawn': {'x': 0, 'y': 2, 'z': 0},
    },
    {
        'name': "Moving Target",
        'platforms': [
            {'x': 0,  'y': 0, 'z': 0,  'w': 4, 'h': 1, 'd': 3},
            {'x': 8,  'y': 0, 'z': 0,  'w': 5, 'h': 1, 'd': 3},
            {'x': 15, 'y': 0, 'z': 0,  'w': 4, 'h': 1, 'd': 3},
        ],
        'goal': {'x': 17, 'y': 1, 'z': 0},
        'spawn': {'x': 0, 'y': 2, 'z': 0},
        'obstacles': [
            {'x': 8, 'y': 1.5, 'z': 0, 'w': 1, 'h': 1, 'd': 2, 'axis': 'z', 'range': 1.0, 'speed': 2.0},
        ],
    },
]


# --- Agent ---
class Agent:
    def __init__(self):
        self.x = self.y = self.z = 0
        self.vx = self.vy = self.vz = 0
        self.on_ground = False
        self.last_x = self.last_z = 0

    def reset(self, spawn):
        self.x, self.y, self.z = spawn['x'], spawn['y'], spawn['z']
        self.vx = self.vy = self.vz = 0
        self.on_ground = False
        self.last_x = self.x
        self.last_z = self.z

    def perceive(self, goal, obstacles):
        vel = np.array([self.vx / 10.0, self.vy / 15.0, self.vz / 10.0], dtype=np.float32)
        dx = goal['x'] - self.x
        dz = goal['z'] - self.z
        dist = math.sqrt(dx**2 + dz**2) + 1e-6
        dir_goal = np.array([dx / dist, 0, dz / dist], dtype=np.float32)
        ground = np.array([1.0 if self.on_ground else 0.0], dtype=np.float32)
        nearest_d = float('inf')
        nearest_r = np.zeros(3, dtype=np.float32)
        for o in obstacles:
            d = math.sqrt((self.x - o[0])**2 + (self.z - o[2])**2)
            if d < nearest_d:
                nearest_d = d
                nearest_r = np.array([(o[0] - self.x) / 10.0, (o[1] - self.y) / 5.0, (o[2] - self.z) / 10.0], dtype=np.float32)
        nd_norm = np.array([min(nearest_d / 10.0, 1.0)], dtype=np.float32)
        t_norm = np.array([0.0], dtype=np.float32)
        return np.concatenate([vel, dir_goal, ground, nearest_r, nd_norm, t_norm]).astype(np.float32)

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

    def physics_step(self, dt, platforms, goal_pos):
        self.vy += GRAVITY * dt
        self.vy = max(self.vy, TERMINAL_VELOCITY)

        # Move
        nx = self.x + self.vx * dt
        ny = self.y + self.vy * dt
        nz = self.z + self.vz * dt

        # Resolve X
        self._try_move('x', nx, platforms)
        # Resolve Y
        self._try_move('y', ny, platforms)
        # Resolve Z
        self._try_move('z', nz, platforms)

        self.vx *= 0.85
        self.vz *= 0.85

        # Fell
        if self.y < -10:
            return -50.0, True

        # Reached goal
        dist = math.sqrt((self.x - goal_pos[0])**2 + (self.y - goal_pos[1])**2 + (self.z - goal_pos[2])**2)
        if dist < 1.5:
            return 100.0, True

        # Progress reward
        db = math.sqrt((self.last_x - goal_pos[0])**2 + (self.last_z - goal_pos[2])**2)
        da = math.sqrt((self.x - goal_pos[0])**2 + (self.z - goal_pos[2])**2)
        self.last_x = self.x
        self.last_z = self.z

        return (db - da) * 10.0 - 0.1, False

    def _try_move(self, axis, new_val, platforms):
        old = getattr(self, axis)
        setattr(self, axis, new_val)
        if axis == 'y':
            self.on_ground = False
        for p in platforms:
            if self._collides(p):
                setattr(self, axis, old)
                if axis == 'y' and self.vy < 0:
                    self.y = p['y'] + p['h'] / 2 + 0.5
                    self.vy = 0
                    self.on_ground = True
                elif axis == 'x':
                    self.vx = 0
                elif axis == 'z':
                    self.vz = 0
                break

    def _collides(self, p):
        hw, hh, hd = p['w'] / 2 + 0.4, p['h'] / 2 + 0.5, p['d'] / 2 + 0.4
        return (abs(self.x - p['x']) < hw and
                abs(self.y - p['y']) < hh and
                abs(self.z - p['z']) < hd)


# --- Training Loop (no rendering) ---
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
        agent.reset(lvl['spawn'])

        state = agent.perceive(lvl['goal'], [])
        done = False
        steps = 0
        ep_reward = 0

        while not done and steps < MAX_STEPS:
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randrange(6)
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze()
                    action = int(q_vals.argmax().item())

            agent.apply_action(action)
            reward, done = agent.physics_step(0.02, lvl['platforms'], (lvl['goal']['x'], lvl['goal']['y'], lvl['goal']['z']))
            ep_reward += reward

            next_state = agent.perceive(lvl['goal'], [])
            replay.push(state, action, reward, next_state, float(done))
            state = next_state

            # Train
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

        if reward >= 50:
            wins += 1

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if ep % 20 == 0:
            print(f"Episode {ep}: steps={steps}, reward={ep_reward:.1f}, wins={wins}/{ep+1} ({wins/(ep+1)*100:.0f}%), epsilon={epsilon:.1%}")

    return wins, ep + 1


if __name__ == '__main__':
    import sys

    headless = len(sys.argv) > 1 and sys.argv[1] == '--headless'
    episodes = int(sys.argv[-1]) if len(sys.argv) > 1 and sys.argv[-1].isdigit() else 200

    if headless:
        # Pure headless training test
        wins, total = train_headless(episodes)
        print(f"\nFinal: {wins}/{total} wins ({wins/total*100:.0f}%)")
    else:
        # Full game with Ursina rendering
        from ursina import Ursina, Entity, Text, camera, DirectionalLight, AmbientLight, Sky, held_keys, time, color, destroy, Vec2
        from ursina.ursinastring import UrsinaString
        app = Ursina()

        # Window setup
        window.borderless = False
        window.fps_counter.enabled = False
        window.exit_button.enabled = False
        window.title = 'Obstacle Runner'

        # Camera
        camera.position = (0, 10, 18)
        camera.rotation_x = -25

        # Lighting
        DirectionalLight(position=(5, 10, -5), rotation=(45, -45, 0))
        AmbientLight(color=color.rgba(100, 100, 120, 100))
        Sky(color=color.rgb(15, 15, 25))

        # UI
        title_text = Text("Level 1: First Steps", position=(-0.85, 0.45), scale=1.5, color=color.white)
        stats_text = Text("Episode: 0 | Won: 0 | Eps: 100%", position=(-0.85, 0.38), scale=1, color=color.white)
        help_text = Text("[R] Reset  [N] Next Level  [H] Headless  [,/.] Epsilon", position=(-0.85, -0.47), scale=0.8, color=color.white)
        loss_text = Text("Loss: 0.0000", position=(-0.85, 0.32), scale=1, color=color.white)

        # Accumulator for fixed timestep
        accumulator_state = [0.0]  # list so closure can modify

        # RL state
        q_net = DQNet().to(device)
        target_net = DQNet().to(device)
        target_net.load_state_dict(q_net.state_dict())
        optimizer = optim.Adam(q_net.parameters(), lr=LR)
        replay = ReplayBuffer()
        loss_fn = nn.MSELoss()

        current_level = 0
        episode = 0
        wins = 0
        epsilon = EPSILON_START
        gen = 0
        total_loss = 0.0
        loss_count = 0
        steps = 0
        last_state = None
        last_action = None
        episode_reward = 0.0

        agent = Agent()
        platform_entities = []
        obstacle_entities = []
        goal_entity = None
        agent_entity = None
        trail_points = []

        def build_level(idx):
            global platform_entities, obstacle_entities, goal_entity, agent_entity, trail_points
            for e in platform_entities + obstacle_entities:
                destroy(e)
            if goal_entity: destroy(goal_entity)
            if agent_entity: destroy(agent_entity)
            for tp in trail_points: destroy(tp)
            trail_points.clear()
            platform_entities.clear()
            obstacle_entities.clear()

            lvl = LEVELS[idx]

            for p in lvl['platforms']:
                e = Entity(model='cube', position=(p['x'], p['y'], p['z']),
                           scale=(p['w'], p['h'], p['d']),
                           color=color.rgb(40, 40, 55), collider='box')
                platform_entities.append(e)

            for obs in lvl.get('obstacles', []):
                e = Entity(model='cube', position=(obs['x'], obs['y'], obs['z']),
                           scale=(obs['w'], obs['h'], obs['d']),
                           color=color.rgb(200, 80, 60), collider='box')
                obstacle_entities.append({'entity': e, 'axis': obs.get('axis', 'z'),
                                          'range': obs.get('range', 1.0),
                                          'speed': obs.get('speed', 2.0),
                                          'ox': obs['x'], 'oy': obs['y'], 'oz': obs['z']})

            goal_entity = Entity(model='cube',
                                  position=(lvl['goal']['x'], lvl['goal']['y'], lvl['goal']['z']),
                                  scale=(2, 0.3, 2),
                                  color=color.rgb(50, 200, 100),
                                  emission=color.rgb(30, 120, 60))

            agent_entity = Entity(model='cube',
                                   position=(lvl['spawn']['x'], lvl['spawn']['y'], lvl['spawn']['z']),
                                   scale=(0.8, 0.8, 0.8),
                                   color=color.rgb(220, 220, 240),
                                   collider='box')

            agent.reset(lvl['spawn'])
            title_text.text = f"Level {idx + 1}: {lvl['name']}"

        def reset_episode():
            global trail_points
            for tp in trail_points:
                destroy(tp)
            trail_points.clear()
            lvl = LEVELS[current_level]
            agent.reset(lvl['spawn'])
            agent_entity.position = (lvl['spawn']['x'], lvl['spawn']['y'], lvl['spawn']['z'])

        def select_action(state, eps):
            if random.random() < eps:
                return random.randrange(6)
            with torch.no_grad():
                return int(q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze().argmax().item())

        build_level(0)

        def update():
            global episode, wins, epsilon, total_loss, loss_count, gen
            global last_state, last_action, steps, current_level, episode_reward

            if held_keys['r']:
                reset_episode()
                return
            if held_keys['n']:
                current_level = (current_level + 1) % len(LEVELS)
                build_level(current_level)
                return
            if held_keys['h']:
                # Switch to headless
                app.destroy()
                print("\n=== HEADLESS MODE ===")
                wins, total = train_headless(episodes=500)
                print(f"\nFinal: {wins}/{total} wins ({wins/total*100:.0f}%)")
                sys.exit(0)

        def input(key):
            global epsilon
            if key == ',':
                epsilon = min(1.0, epsilon + 0.05)
            elif key == '.':
                epsilon = max(0.0, epsilon - 0.05)
            stats_text.text = f"Episode: {episode} | Won: {wins} | Eps: {epsilon:.0%}"

        # Physics + RL step (every frame, using fixed delta)
        accumulator = 0.0
        physics_dt = 0.02

        def update():
            global episode, wins, epsilon, total_loss, loss_count, gen, last_state, last_action
            global steps, current_level, episode_reward, loss_count, total_loss

            if held_keys['r']:
                reset_episode()
                return
            if held_keys['n']:
                current_level = (current_level + 1) % len(LEVELS)
                build_level(current_level)
                return

            accumulator_state[0] += time.dt

            while accumulator_state[0] >= physics_dt:
                accumulator_state[0] -= physics_dt

                lvl = LEVELS[current_level]

                # Animate obstacles
                t = time.time()
                for obs in obstacle_entities:
                    offset = math.sin(t * obs['speed']) * obs['range']
                    if obs['axis'] == 'z':
                        obs['entity'].z = obs['oz'] + offset
                    elif obs['axis'] == 'x':
                        obs['entity'].x = obs['ox'] + offset
                    elif obs['axis'] == 'y':
                        obs['entity'].y = obs['oy'] + offset

                # Obstacle positions for perception
                obstacles = []
                for obs in obstacle_entities:
                    p = obs['entity'].position
                    s = obs['entity'].scale
                    obstacles.append((p[0], p[1], p[2]))

                # State
                goal_pos = (lvl['goal']['x'], lvl['goal']['y'], lvl['goal']['z'])
                state = agent.perceive({'x': goal_pos[0], 'z': goal_pos[2]}, obstacles)

                # Action
                action = select_action(state, epsilon)

                # Execute
                agent.apply_action(action)
                reward, done = agent.physics_step(physics_dt, lvl['platforms'], goal_pos)
                episode_reward += reward

                # Store
                if last_state is not None:
                    replay.push(last_state, last_action, reward, state, float(done))

                    if len(replay) >= BATCH_SIZE:
                        batch = replay.sample()
                        s_t = torch.tensor(batch[0])
                        a_t = torch.tensor(batch[1])
                        r_t = torch.tensor(batch[2])
                        ns_t = torch.tensor(batch[3])
                        d_t = torch.tensor(batch[4])
                        q_sa = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze()
                        with torch.no_grad():
                            q_target = r_t + GAMMA * (1 - d_t) * target_net(ns_t).max(1)[0]
                        loss = loss_fn(q_sa, q_target)
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                        optimizer.step()
                        total_loss += loss.item()
                        loss_count += 1
                        loss_text.text = f"Loss: {total_loss / max(loss_count, 1):.4f}"

                last_state = state
                last_action = action

                # Visual
                agent_entity.position = (agent.x, agent.y, agent.z)

                if steps % 3 == 0:
                    tp = Entity(model='sphere', position=(agent.x, agent.y - 0.3, agent.z),
                                scale=0.1, color=color.rgb(200, 200, 255))
                    trail_points.append(tp)

                steps += 1
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
                stats_text.text = f"Episode: {episode} | Won: {wins} | Eps: {epsilon:.0%}"

                if done or steps >= MAX_STEPS:
                    gen += 1
                    episode += 1
                    if reward >= 50:
                        wins += 1

                    if gen % TARGET_UPDATE == 0:
                        target_net.load_state_dict(q_net.state_dict())

                    reset_episode()
                    steps = 0
                    episode_reward = 0.0
                    last_state = None
                    last_action = None

        app.run()
