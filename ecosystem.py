#!/usr/bin/env python3
"""
Learning Sandbox — Ecosystem Simulation
Predators: DQN neural net trained by TD error. Prey: tabular Q-learning.
Both inherit learned weights on reproduction.
"""

import pygame
import random
import math
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim

pygame.init()

# Screen
WIDTH, HEIGHT = 1200, 700
ARENA_W, ARENA_H = 900, 600
GRAPH_X, GRAPH_Y = ARENA_W + 20, 50
GRAPH_W, GRAPH_H = 250, 200

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Learning Sandbox")
clock = pygame.time.Clock()

BG = (10, 10, 20)
PREY_COLOR = (50, 220, 100)
PRED_COLOR = (220, 60, 60)
FOOD_COLOR = (255, 230, 80)
GRID_COLOR = (30, 30, 50)
WHITE = (220, 220, 220)
PANEL_BG = (20, 20, 35)

font_renderer = None


def draw_text(surface, text, pos, color=None):
    if color is None:
        color = WHITE
    if font_renderer:
        surf = font_renderer.render(text, True, color)
        surface.blit(surf, pos)
    else:
        w, h = len(text) * 7, 12
        pygame.draw.rect(surface, color, (*pos, w, h), border_radius=1)


# --- Constants ---
FPS = 60
ARENA_RECT = pygame.Rect(10, 50, ARENA_W, ARENA_H)
INITIAL_PREY = 50
INITIAL_PRED = 4
FOOD_COUNT = 50
PREY_REPRODUCE_THRESHOLD = 1.5
PRED_REPRODUCE_THRESHOLD = 5.0
PRED_STARVE_STEPS = 400
FOOD_RESPAWN_RATE = 0.03
REWARD_GATHER = 1.0
REWARD_CATCH = 5.0
REWARD_CLOSER = 0.05
PENALTY_DANGER = 0.2
PENALTY_STARVE = 0.1
CELL_SIZE = 60

# DQN Hyperparams
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 32
REPLAY_SIZE = 2000
TARGET_UPDATE = 10


# --- Device ---
device = torch.device("cpu")


# --- DQN Neural Net ---
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
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


# --- Experience Replay ---
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


# --- Spatial Hash ---
class SpatialHash:
    def __init__(self, cell_size=CELL_SIZE):
        self.cell_size = cell_size
        self.cells = {}

    def clear(self):
        self.cells = {}

    def _hash(self, x, y):
        return int(x // self.cell_size), int(y // self.cell_size)

    def insert(self, obj, x, y):
        k = self._hash(x, y)
        if k not in self.cells:
            self.cells[k] = []
        self.cells[k].append(obj)

    def query_radius(self, x, y, radius):
        cx, cy = self._hash(x, y)
        r = int(radius // self.cell_size) + 1
        results = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                k = (cx + dx, cy + dy)
                if k in self.cells:
                    for obj in self.cells[k]:
                        d = math.hypot(x - obj.x, y - obj.y)
                        if d <= radius:
                            results.append((obj, d))
        return results

    @staticmethod
    def nearest(x, y, candidates, max_n=3):
        if not candidates:
            return []
        dists = [(math.hypot(c.x - x, c.y - y), c) for c in candidates]
        dists.sort(key=lambda t: t[0])
        return [c for _, c in dists[:max_n]]


# --- Prey: Tabular Q-Learning ---
class Prey:
    def __init__(self, x, y, q_table=None):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.energy = 1.0
        self.detection_radius = random.uniform(40, 70)
        self.speed = random.uniform(1.0, 2.0)
        self.q_table = q_table.copy() if q_table else {}
        self.last_state = None
        self.last_action = None

    def _state(self, food_near, pred_near):
        """Coarse 5-D state: food_dist bin, food_angle bin, pred_dist bin, pred_angle bin, energy bin"""
        fd = fa = pd = pa = ed = 0
        if food_near:
            f = min(food_near, key=lambda i: i[1])
            fd = min(int(f[1] / self.detection_radius * 4), 4)
            fa = int((math.atan2(f[0].y - self.y, f[0].x - self.x) / math.pi + 1) * 2) % 5
        if pred_near:
            p = min(pred_near, key=lambda i: i[1])
            pd = min(int(p[1] / (self.detection_radius * 2) * 4), 4)
            pa = int((math.atan2(p[0].y - self.y, p[0].x - self.x) / math.pi + 1) * 2) % 5
        ed = min(int(self.energy * 3), 4)
        return (fd, fa, pd, pa, ed)

    def act(self, food_near, pred_near, epsilon=0.1):
        state = self._state(food_near, pred_near)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)

        if random.random() < epsilon:
            action = random.randrange(3)
        else:
            action = int(np.argmax(self.q_table[state]))

        self.last_state = state
        self.last_action = action

        # Execute action
        angle = random.uniform(0, 2 * math.pi)
        if action == 1 and food_near:
            f = min(food_near, key=lambda i: i[1])
            angle = math.atan2(f[0].y - self.y, f[0].x - self.x)
        elif action == 2 and pred_near:
            p = min(pred_near, key=lambda i: i[1])
            angle = math.atan2(self.y - p[0].y, self.x - p[0].x)

        self.vx = math.cos(angle) * self.speed
        self.vy = math.sin(angle) * self.speed
        self.x += self.vx
        self.y += self.vy
        self._bounce()

        return state

    def _bounce(self):
        if self.x < 10:
            self.x = 10
            self.vx *= -1
        if self.x > ARENA_W + 10:
            self.x = ARENA_W + 10
            self.vx *= -1
        if self.y < 50:
            self.y = 50
            self.vy *= -1
        if self.y > ARENA_H + 50:
            self.y = ARENA_H + 50
            self.vy *= -1

    def learn(self, reward, next_state):
        if self.last_state is None:
            return
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(3)
        s = self.last_state
        a = self.last_action
        self.q_table[s][a] += 0.2 * (reward + 0.9 * np.max(self.q_table[next_state]) - self.q_table[s][a])

    def reproduce(self):
        self.energy *= 0.5
        child = Prey(self.x + random.uniform(-10, 10), self.y + random.uniform(-10, 10),
                     q_table={k: v.copy() for k, v in self.q_table.items()})
        child.detection_radius = max(25, min(80, self.detection_radius + random.uniform(-5, 5)))
        child.speed = max(0.8, min(2.5, self.speed + random.uniform(-0.1, 0.1)))
        child.energy = 1.0
        return child


# --- Predator: DQN Neural Net ---
class PredatorDQN:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.energy = 2.5
        self.starve_counter = 0
        self.detection_radius = random.uniform(100, 150)
        self.speed = random.uniform(2.0, 3.5)
        self.angle = random.uniform(0, 2 * math.pi)
        self.q_net = DQN(input_dim=7, hidden_dim=64, output_dim=3).to(device)
        self.target_net = DQN(input_dim=7, hidden_dim=64, output_dim=3).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.replay = ReplayBuffer()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.last_state = None
        self.last_action = None
        self.steps_done = 0
        self.total_loss = 0.0
        self.loss_count = 0

    def perceive(self, nearest_prey):
        """7 inputs: 3 nearest prey (dx, dy normalized), own energy"""
        inputs = np.zeros(7, dtype=np.float32)
        for i, p in enumerate(nearest_prey[:3]):
            inputs[i * 2] = (p.x - self.x) / ARENA_W
            inputs[i * 2 + 1] = (p.y - self.y) / ARENA_H
        inputs[6] = self.energy / 3.0
        return inputs

    def act(self, perception, epsilon=None):
        if epsilon is None:
            epsilon = max(0.02, 1.0 - self.steps_done / 2000)

        state_t = torch.tensor(perception, dtype=torch.float32, device=device).unsqueeze(0)
        self.q_net.eval()
        with torch.no_grad():
            q_vals = self.q_net(state_t).squeeze()
        action = int(q_vals.argmax().item()) if random.random() > epsilon else random.randrange(3)

        self.last_state = perception
        self.last_action = action
        self.steps_done += 1

        # Execute action: 0=left, 1=right, 2=straight
        if action == 0:
            self.angle -= 0.4
        elif action == 1:
            self.angle += 0.4

        self.vx = math.cos(self.angle) * self.speed
        self.vy = math.sin(self.angle) * self.speed
        self.x += self.vx
        self.y += self.vy
        self._bounce()

        return perception

    def _bounce(self):
        if self.x < 10:
            self.x = 10
            self.vx *= -1
            self.angle = math.atan2(self.vy, self.vx)
        if self.x > ARENA_W + 10:
            self.x = ARENA_W + 10
            self.vx *= -1
            self.angle = math.atan2(self.vy, self.vx)
        if self.y < 50:
            self.y = 50
            self.vy *= -1
            self.angle = math.atan2(self.vy, self.vx)
        if self.y > ARENA_H + 50:
            self.y = ARENA_H + 50
            self.vy *= -1
            self.angle = math.atan2(self.vy, self.vx)

    def store(self, action, reward, next_perception, done):
        ns = next_perception if not done else np.zeros(7, dtype=np.float32)
        self.replay.push(self.last_state, action, reward, ns, float(done))

    def learn_batch(self):
        if len(self.replay) < BATCH_SIZE:
            return 0.0
        states, actions, rewards, next_states, dones = self.replay.sample()
        s_t = torch.tensor(states, dtype=torch.float32, device=device)
        a_t = torch.tensor(actions, dtype=torch.int64, device=device)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        ns_t = torch.tensor(next_states, dtype=torch.float32, device=device)
        d_t = torch.tensor(dones, dtype=torch.float32, device=device)

        self.q_net.train()
        q_sa = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_target = r_t + GAMMA * (1 - d_t) * self.target_net(ns_t).max(1)[0]
        loss = nn.MSELoss()(q_sa, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.total_loss += loss.item()
        self.loss_count += 1
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def avg_loss(self):
        return self.total_loss / max(self.loss_count, 1)

    def reproduce(self):
        self.energy *= 0.5
        child = PredatorDQN(self.x, self.y)
        child.q_net.load_state_dict(self.q_net.state_dict())
        child.target_net.load_state_dict(self.target_net.state_dict())
        child.energy = 2.5
        return child


# --- Food ---
class Food:
    def __init__(self, x=None, y=None):
        if x is None:
            self.respawn()
        else:
            self.x, self.y = x, y
        self.radius = random.uniform(4, 7)

    def respawn(self):
        self.x = random.uniform(20, ARENA_W - 20)
        self.y = random.uniform(60, ARENA_H - 20)


# --- Stats ---
class Stats:
    def __init__(self):
        self.prey_history = []
        self.pred_history = []
        self.loss_history = []
        self.max_history = 300

    def record(self, prey_count, pred_count, avg_loss=0.0):
        self.prey_history.append(prey_count)
        self.pred_history.append(pred_count)
        self.loss_history.append(avg_loss)
        if len(self.prey_history) > self.max_history:
            self.prey_history.pop(0)
            self.pred_history.pop(0)
            self.loss_history.pop(0)


# --- World ---
prey_list = []
pred_list = []
food_list = []
stats = Stats()
spatial = SpatialHash()
gen = 0


def init():
    global prey_list, pred_list, food_list, stats, gen
    prey_list = [Prey(random.uniform(50, ARENA_W - 50), random.uniform(80, ARENA_H - 50))
                 for _ in range(INITIAL_PREY)]
    pred_list = [PredatorDQN(random.uniform(50, ARENA_W - 50), random.uniform(80, ARENA_H - 50))
                 for _ in range(INITIAL_PRED)]
    food_list = [Food() for _ in range(FOOD_COUNT)]
    stats = Stats()
    gen = 0


init()

running = True
speed = 1
show_graph = True
global_epsilon = 0.2
update_counter = 0


def draw_agents():
    for f in food_list:
        pygame.draw.circle(screen, FOOD_COLOR, (int(f.x), int(f.y)), int(f.radius))
    for p in prey_list:
        r = max(3, int(p.detection_radius * 0.12))
        pygame.draw.circle(screen, PREY_COLOR, (int(p.x), int(p.y)), r)
    for pr in pred_list:
        pygame.draw.circle(screen, PRED_COLOR, (int(pr.x), int(pr.y)), 6)
        pygame.draw.line(screen, (255, 120, 120), (int(pr.x), int(pr.y)),
                         (int(pr.x + pr.vx * 7), int(pr.y + pr.vy * 7)), 2)


while running:
    dt = clock.tick(FPS) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                speed = 5 if speed == 1 else 1
                global_epsilon = 0.05 if speed > 1 else 0.2
            elif event.key == pygame.K_r:
                init()
            elif event.key == pygame.K_g:
                show_graph = not show_graph
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if ARENA_RECT.collidepoint(mx, my):
                food_list.append(Food(mx, my))

    for _ in range(speed):
        gen += 1
        update_counter += 1

        spatial.clear()
        for p in prey_list:
            spatial.insert(p, p.x, p.y)
        for f in food_list:
            spatial.insert(f, f.x, f.y)

        # --- Prey step ---
        new_prey = []
        for p in prey_list:
            food_near = [(f, d) for f, d in spatial.query_radius(p.x, p.y, p.detection_radius) if isinstance(f, Food)]
            pred_near = [(pr, d) for pr, d in spatial.query_radius(p.x, p.y, p.detection_radius * 2) if isinstance(pr, PredatorDQN)]

            state = p.act(food_near, pred_near, epsilon=global_epsilon)

            reward = -0.02
            eaten = False
            for food, dist in food_near:
                if dist < p.detection_radius * 0.6:
                    reward = REWARD_GATHER
                    p.energy += 0.5
                    food.respawn()
                    eaten = True
                    break
            if not eaten and food_near:
                reward += REWARD_CLOSER
            if pred_near:
                reward -= PENALTY_DANGER

            next_food = [(f, d) for f, d in spatial.query_radius(p.x, p.y, p.detection_radius) if isinstance(f, Food)]
            next_pred = [(pr, d) for pr, d in spatial.query_radius(p.x, p.y, p.detection_radius * 2) if isinstance(pr, PredatorDQN)]
            next_state = p._state(next_food, next_pred)
            p.learn(reward, next_state)
            p.energy += 0.005

            if p.energy > PREY_REPRODUCE_THRESHOLD:
                new_prey.append(p.reproduce())
            if p.energy < 0:
                p.energy = -999  # mark dead

        prey_list.extend(new_prey)
        prey_list = [p for p in prey_list if p.energy >= 0]

        # --- Predator step ---
        new_pred = []
        total_loss = 0.0
        loss_count = 0
        for pr in pred_list:
            prey_in = [(p, d) for p, d in spatial.query_radius(pr.x, pr.y, pr.detection_radius) if isinstance(p, Prey)]
            prey_cands = [p for p, _ in prey_in]
            nearest = SpatialHash.nearest(pr.x, pr.y, prey_cands, max_n=3)

            # Pad to 3
            while len(nearest) < 3:
                nearest.append(type('Dummy', (), {'x': pr.x, 'y': pr.y})())

            perception = pr.perceive(nearest)
            epsilon = max(0.02, global_epsilon)
            next_perception = pr.act(perception, epsilon=epsilon)

            reward = -PENALTY_STARVE
            pr.starve_counter += 1
            pr.energy -= PENALTY_STARVE * 0.015

            if prey_in:
                curr_d = min(d for _, d in prey_in)
                reward += REWARD_CLOSER * (1.0 - curr_d / pr.detection_radius)

            done = False
            caught = None
            for prey in list(prey_list):
                if math.hypot(pr.x - prey.x, pr.y - prey.y) < 14:
                    caught = prey
                    pr.energy += REWARD_CATCH
                    pr.starve_counter = 0
                    reward = REWARD_CATCH
                    break
            if caught:
                prey_list.remove(caught)

            pr.store(pr.last_action, reward, next_perception, done)
            loss = pr.learn_batch()
            if loss > 0:
                total_loss += loss
                loss_count += 1

            if pr.energy > PRED_REPRODUCE_THRESHOLD and len(new_pred) < 2:
                new_pred.append(pr.reproduce())
            if pr.starve_counter > PRED_STARVE_STEPS:
                pr.energy -= 0.15

        if update_counter % TARGET_UPDATE == 0:
            for pr in pred_list:
                pr.update_target()

        pred_list.extend(new_pred)
        pred_list = [p for p in pred_list if p.energy > 0]

        for f in food_list:
            if random.random() < FOOD_RESPAWN_RATE:
                f.respawn()

        avg_loss = total_loss / max(loss_count, 1) if loss_count else 0.0
        stats.record(len(prey_list), len(pred_list), avg_loss)

    # --- Render ---
    screen.fill(BG)
    pygame.draw.rect(screen, (15, 15, 30), ARENA_RECT)
    pygame.draw.rect(screen, GRID_COLOR, ARENA_RECT, 1)
    draw_agents()

    panel_rect = pygame.Rect(ARENA_W + 10, 10, WIDTH - ARENA_W - 20, HEIGHT - 20)
    pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=6)

    avg_loss = sum(p.avg_loss() for p in pred_list) / max(len(pred_list), 1)
    draw_text(screen, f"Gen: {gen}", (ARENA_W + 20, 20), WHITE)
    draw_text(screen, f"Prey: {len(prey_list)}", (ARENA_W + 20, 40), PREY_COLOR)
    draw_text(screen, f"Predators: {len(pred_list)}", (ARENA_W + 20, 60), PRED_COLOR)
    draw_text(screen, f"Speed: {speed}x [SPACE]", (ARENA_W + 20, HEIGHT - 90), WHITE)
    draw_text(screen, f"Explore: {global_epsilon:.0%}", (ARENA_W + 20, HEIGHT - 70), WHITE)
    draw_text(screen, "[R] Reset", (ARENA_W + 20, HEIGHT - 50), WHITE)
    draw_text(screen, "[Click] Add food", (ARENA_W + 20, HEIGHT - 30), WHITE)

    if show_graph and len(stats.prey_history) > 1:
        graph_rect = pygame.Rect(GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H)
        pygame.draw.rect(screen, (10, 10, 25), graph_rect, border_radius=4)
        pygame.draw.rect(screen, GRID_COLOR, graph_rect, 1)

        max_val = max(max(stats.prey_history), max(stats.pred_history), 1)
        pts_prey = [(GRAPH_X + int(i / stats.max_history * GRAPH_W),
                     GRAPH_Y + GRAPH_H - int(cnt / max_val * (GRAPH_H - 20)))
                    for i, cnt in enumerate(stats.prey_history)]
        pts_pred = [(GRAPH_X + int(i / stats.max_history * GRAPH_W),
                     GRAPH_Y + GRAPH_H - int(cnt / max_val * (GRAPH_H - 20)))
                    for i, cnt in enumerate(stats.pred_history)]
        if pts_prey:
            pygame.draw.lines(screen, PREY_COLOR, False, pts_prey, 2)
        if pts_pred:
            pygame.draw.lines(screen, PRED_COLOR, False, pts_pred, 2)

    pygame.display.flip()

pygame.quit()
