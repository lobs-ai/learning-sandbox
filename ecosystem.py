#!/usr/bin/env python3
"""
Learning Sandbox — Ecosystem Simulation
Prey and predators both have neural nets, both learn via policy gradient, both inherit weights on reproduction.
"""

import pygame
import random
import math
import numpy as np

# Initialize pygame
pygame.init()

# Screen
WIDTH, HEIGHT = 1200, 700
ARENA_W, ARENA_H = 900, 600
GRAPH_X, GRAPH_Y = ARENA_W + 20, 50
GRAPH_W, GRAPH_H = 250, 200

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Learning Sandbox")
clock = pygame.time.Clock()

# Colors — must be defined before draw_text
BG = (10, 10, 20)
PREY_COLOR = (50, 220, 100)
PRED_COLOR = (220, 60, 60)
FOOD_COLOR = (255, 230, 80)
GRID_COLOR = (30, 30, 50)
WHITE = (220, 220, 220)
PANEL_BG = (20, 20, 35)

# Font — lazy init to avoid Python 3.14 + pygame circular import bug
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
INITIAL_PREY = 120
INITIAL_PRED = 8
FOOD_COUNT = 80
PREY_REPRODUCE_THRESHOLD = 1.2
PRED_REPRODUCE_THRESHOLD = 4.0
PREY_DEATH_THRESHOLD = 0.0
PRED_STARVE_STEPS = 250
WEIGHT_MUTATION = 0.08
REWARD_GATHER = 0.3
REWARD_CATCH = 3.0
REWARD_NEAR_MISS = 0.05
PENALTY_DANGER = 0.1
PENALTY_STARVE = 0.3
SPEED_MULTIPLIER = 1
CELL_SIZE = 50


# --- Neural Net with Backprop ---
class MLP:
    """MLP with forward pass and backward gradient computation."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        scale = np.sqrt(2.0 / input_dim)
        self.fc1_w = np.random.randn(input_dim, hidden_dim) * scale
        self.fc1_b = np.zeros(hidden_dim)
        scale_out = np.sqrt(2.0 / hidden_dim)
        self.fc2_w = np.random.randn(hidden_dim, output_dim) * scale_out
        self.fc2_b = np.zeros(output_dim)

    def forward(self, x):
        """Returns (output, cache) where cache is needed for backward."""
        self.x = x  # store input
        self.z1 = self.fc1_w.T @ x + self.fc1_b
        self.h1 = np.tanh(self.z1)
        self.z2 = self.fc2_w.T @ self.h1 + self.fc2_b
        # Softmax for probability distribution
        e_z = np.exp(self.z2 - np.max(self.z2))
        self.probs = e_z / e_z.sum()
        return self.probs

    def backward(self, reward, action):
        """REINFORCE policy gradient: grad = (log_prob_of_action) * reward.
        Computes gradients w.r.t. weights and applies SGD step."""
        # Policy gradient: dlog_pi/a / dtheta * reward
        # grad_z2 = (probs - one_hot) * reward  (cross-entropy gradient)
        grad_z2 = self.probs.copy()
        grad_z2[action] -= 1.0  # one-hot
        grad_z2 *= reward

        # Gradient of output w.r.t. hidden weights
        # dz2/dW2 = outer(h1, grad_z2)
        grad_w2 = np.outer(self.h1, grad_z2)
        grad_b2 = grad_z2

        # Backprop through tanh: dh1/dz1 = (1 - tanh^2)
        grad_h1 = self.fc2_w @ grad_z2
        grad_z1 = grad_h1 * (1 - self.h1 ** 2)

        # Gradient of first layer
        grad_w1 = np.outer(self.x, grad_z1)
        grad_b1 = grad_z1

        lr = 0.005
        self.fc1_w -= lr * grad_w1
        self.fc1_b -= lr * grad_b1
        self.fc2_w -= lr * grad_w2
        self.fc2_b -= lr * grad_b2

    def copy(self):
        hidden_dim = self.fc1_w.shape[1]
        input_dim = self.fc1_w.shape[0]
        output_dim = self.fc2_w.shape[1]
        other = MLP(input_dim, hidden_dim, output_dim)
        other.fc1_w = self.fc1_w.copy()
        other.fc1_b = self.fc1_b.copy()
        other.fc2_w = self.fc2_w.copy()
        other.fc2_b = self.fc2_b.copy()
        return other

    def mutate(self, rate=WEIGHT_MUTATION):
        self.fc1_w += np.random.randn(*self.fc1_w.shape) * rate
        self.fc2_w += np.random.randn(*self.fc2_w.shape) * rate


# --- Spatial Hash ---
class SpatialHash:
    def __init__(self, cell_size=CELL_SIZE):
        self.cell_size = cell_size
        self.cells = {}

    def clear(self):
        self.cells = {}

    def hash_pos(self, x, y):
        return int(x // self.cell_size), int(y // self.cell_size)

    def insert(self, obj, x, y):
        k = self.hash_pos(x, y)
        if k not in self.cells:
            self.cells[k] = []
        self.cells[k].append(obj)

    def query_radius(self, x, y, radius):
        """Return (obj, dist) for all objects within radius. O(1) per cell."""
        cx, cy = self.hash_pos(x, y)
        results = []
        search_cells = int(radius // self.cell_size) + 1
        for dx in range(-search_cells, search_cells + 1):
            for dy in range(-search_cells, search_cells + 1):
                k = (cx + dx, cy + dy)
                if k in self.cells:
                    for obj in self.cells[k]:
                        d = math.hypot(x - obj.x, y - obj.y)
                        if d <= radius:
                            results.append((obj, d))
        return results

    def nearest(self, x, y, candidates, max_count=3):
        """Return up to max_count nearest candidates, sorted by distance."""
        if not candidates:
            return []
        dists = [(math.hypot(c.x - x, c.y - y), c) for c in candidates]
        dists.sort(key=lambda t: t[0])
        return [c for _, c in dists[:max_count]]


# --- Prey ---
class Prey:
    def __init__(self, x, y, brain=None):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.energy = 0.5
        self.detection_radius = random.uniform(30, 60)
        self.speed = random.uniform(0.8, 1.5)
        self.age = 0
        # Brain: 5 inputs, 8 hidden, 3 outputs (seek_food, flee, wander)
        self.brain = brain if brain else MLP(5, 8, 3)

    def perceive(self, food_in_range, pred_in_range, nearest_pred_dist):
        """Perception from nearby food and predators (already filtered by spatial hash)."""
        # Food: distance and angle to nearest food
        if food_in_range:
            f = min(food_in_range, key=lambda item: item[1])
            food_dist = f[1] / self.detection_radius
            food_angle = math.atan2(f[0].y - self.y, f[0].x - self.x) / math.pi
        else:
            food_dist = 1.0
            food_angle = 0.0

        # Predator: distance and angle to nearest predator
        if pred_in_range:
            p = min(pred_in_range, key=lambda item: item[1])
            pred_dist = p[1] / (self.detection_radius * 2)
            pred_angle = math.atan2(p[0].y - self.y, p[0].x - self.x) / math.pi
        else:
            pred_dist = 1.0
            pred_angle = 0.0

        # Energy level
        energy_norm = self.energy

        return np.array([food_dist, food_angle, pred_dist, pred_angle, energy_norm])

    def act(self, perception):
        """Forward through MLP, pick action, move."""
        probs = self.brain.forward(perception)
        action = np.random.choice(len(probs), p=probs)
        self.last_action = action
        self.last_probs = probs

        # Compute target direction based on action
        target_angle = random.uniform(0, 2 * math.pi)
        if action == 0 and perception[0] < 1.0:
            # Seek food
            target_angle = perception[1] * math.pi
        elif action == 1 and perception[2] < 1.0:
            # Flee predator
            target_angle = perception[3] * math.pi + math.pi
        # else wander (random direction)

        self.vx = math.cos(target_angle) * self.speed
        self.vy = math.sin(target_angle) * self.speed
        self.x += self.vx
        self.y += self.vy

        # Bounce off walls
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

    def learn(self, reward):
        """Policy gradient update using REINFORCE."""
        if hasattr(self, 'last_action') and hasattr(self, 'last_probs'):
            self.brain.backward(reward, self.last_action)

    def reproduce(self):
        self.energy *= 0.5
        child = Prey(self.x + random.uniform(-10, 10), self.y + random.uniform(-10, 10),
                     self.brain.copy())
        child.detection_radius = max(15, min(80, self.detection_radius + random.uniform(-5, 5)))
        child.speed = max(0.5, min(2.5, self.speed + random.uniform(-0.1, 0.1)))
        child.energy = 0.8
        return child


# --- Predator ---
class Predator:
    def __init__(self, x, y, brain=None):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.energy = 0.5
        self.starve_counter = 0
        self.detection_radius = random.uniform(60, 100)
        self.speed = random.uniform(1.5, 2.5)
        # Brain: 7 inputs (3 prey * 2 coords), hidden=10, output=3 (turn_left, turn_right, straight)
        self.brain = brain if brain else MLP(9, 10, 3)

    def perceive(self, nearest_prey):
        """Prey perception: 3 nearest prey (x, y offsets), plus own velocity and energy."""
        inputs = []
        for i in range(3):
            if i < len(nearest_prey):
                p = nearest_prey[i]
                inputs.append((p.x - self.x) / ARENA_W)
                inputs.append((p.y - self.y) / ARENA_H)
            else:
                inputs.append(0.0)
                inputs.append(0.0)
        inputs.append(self.vx / 3.0)
        inputs.append(self.vy / 3.0)
        inputs.append(self.energy)
        return np.array(inputs)

    def act(self, perception):
        """Forward through MLP, pick action, move."""
        probs = self.brain.forward(perception)
        action = np.random.choice(len(probs), p=probs)
        self.last_action = action
        self.last_probs = probs

        current_angle = math.atan2(self.vy, self.vx) if (self.vx or self.vy) else 0.0
        if action == 0:
            angle = current_angle - 0.4
        elif action == 1:
            angle = current_angle + 0.4
        else:
            angle = current_angle

        self.vx = math.cos(angle) * self.speed
        self.vy = math.sin(angle) * self.speed
        self.x += self.vx
        self.y += self.vy

        # Bounce off walls
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

    def learn(self, reward):
        if hasattr(self, 'last_action') and hasattr(self, 'last_probs'):
            self.brain.backward(reward, self.last_action)

    def reproduce(self):
        self.energy *= 0.5
        child = Predator(self.x, self.y, self.brain.copy())
        child.speed = max(1.0, min(4.0, self.speed + random.uniform(-0.2, 0.2)))
        child.energy = 0.5
        return child


# --- Food ---
class Food:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.radius = random.uniform(3, 6)

    def respawn(self):
        self.x = random.uniform(20, ARENA_W - 20)
        self.y = random.uniform(60, ARENA_H - 20)


# --- Stats ---
class Stats:
    def __init__(self):
        self.prey_history = []
        self.pred_history = []
        self.max_history = 300

    def record(self, prey_count, pred_count):
        self.prey_history.append(prey_count)
        self.pred_history.append(pred_count)
        if len(self.prey_history) > self.max_history:
            self.prey_history.pop(0)
            self.pred_history.pop(0)


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
    pred_list = [Predator(random.uniform(50, ARENA_W - 50), random.uniform(80, ARENA_H - 50))
                 for _ in range(INITIAL_PRED)]
    food_list = [Food(random.uniform(20, ARENA_W - 20), random.uniform(60, ARENA_H - 20))
                 for _ in range(FOOD_COUNT)]
    stats = Stats()
    gen = 0

init()

running = True
speed = 1
show_graph = True

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
            elif event.key == pygame.K_r:
                init()
            elif event.key == pygame.K_g:
                show_graph = not show_graph
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if ARENA_RECT.collidepoint(mx, my):
                food_list.append(Food(mx, my))

    # --- Simulation step ---
    for _ in range(speed):
        gen += 1

        # Rebuild spatial index
        spatial.clear()
        for p in prey_list:
            spatial.insert(p, p.x, p.y)
        for f in food_list:
            spatial.insert(f, f.x, f.y)

        # --- Prey step ---
        new_prey = []
        for p in prey_list:
            # Use spatial hash for O(1) neighbor lookup instead of O(n) scan
            food_in_range = spatial.query_radius(p.x, p.y, p.detection_radius)
            pred_in_range = spatial.query_radius(p.x, p.y, p.detection_radius * 2)
            nearest_pred = min(pred_in_range, key=lambda item: item[1], default=(None, float('inf')))
            nearest_pred_dist = nearest_pred[1] if nearest_pred[0] else float('inf')

            perception = p.perceive(food_in_range, pred_in_range, nearest_pred_dist)
            p.act(perception)

            # Gather food (only eat Food objects, not other prey)
            eaten = False
            for food, dist in food_in_range:
                if isinstance(food, Food) and dist < p.detection_radius:
                    p.energy += REWARD_GATHER
                    p.learn(REWARD_GATHER)
                    food.respawn()
                    eaten = True
                    break

            # Danger penalty
            if nearest_pred_dist < p.detection_radius * 2:
                p.learn(-PENALTY_DANGER)

            # Reproduce
            if p.energy > PREY_REPRODUCE_THRESHOLD:
                new_prey.append(p.reproduce())

            # Energy drain / death
            if p.energy < PREY_DEATH_THRESHOLD:
                p.energy -= 0.01
            else:
                p.energy += 0.005

        prey_list.extend(new_prey)
        prey_list = [p for p in prey_list if p.energy > 0]

        # --- Predator step ---
        new_pred = []
        for pr in pred_list:
            # Track previous distance to nearest prey for learning signal
            prev_nearest_dist = None
            if prey_list:
                prev_nearest = min(prey_list, key=lambda p: math.hypot(pr.x - p.x, pr.y - p.y))
                prev_nearest_dist = math.hypot(pr.x - prev_nearest.x, pr.y - prev_nearest.y)

            # Find 3 nearest prey using spatial hash (filter to Prey only)
            prey_in_range = spatial.query_radius(pr.x, pr.y, pr.detection_radius)
            prey_candidates = [item[0] for item in prey_in_range if isinstance(item[0], Prey)]
            nearest_prey = spatial.nearest(pr.x, pr.y, prey_candidates, max_count=3)

            perception = pr.perceive(nearest_prey)
            pr.act(perception)
            pr.starve_counter += 1
            pr.energy -= PENALTY_STARVE * 0.01

            # Hunt learning: reward for getting closer, small penalty for moving away
            if prey_list:
                curr_nearest = min(prey_list, key=lambda p: math.hypot(pr.x - p.x, pr.y - p.y))
                curr_nearest_dist = math.hypot(pr.x - curr_nearest.x, pr.y - curr_nearest.y)
                if prev_nearest_dist is not None and curr_nearest_dist < prev_nearest_dist:
                    # Getting closer — positive reward scaled by proximity
                    hunt_reward = REWARD_NEAR_MISS * (1.0 - min(curr_nearest_dist / pr.detection_radius, 1.0))
                    pr.learn(hunt_reward)
                elif prev_nearest_dist is not None and curr_nearest_dist > prev_nearest_dist + 5:
                    # Moving away from nearest prey — small penalty
                    pr.learn(-REWARD_NEAR_MISS * 0.5)

            # Catch prey
            caught = None
            for prey in prey_list:
                if math.hypot(pr.x - prey.x, pr.y - prey.y) < 12:
                    caught = prey
                    pr.energy += REWARD_CATCH
                    pr.starve_counter = 0
                    pr.learn(REWARD_CATCH)
                    break

            if caught:
                prey_list.remove(caught)

            # Reproduce (needs multiple catches, capped to avoid explosions)
            if pr.energy > PRED_REPRODUCE_THRESHOLD and len(new_pred) < 3:
                new_pred.append(pr.reproduce())

            # Starvation
            if pr.starve_counter > PRED_STARVE_STEPS:
                pr.energy -= 0.05

        pred_list.extend(new_pred)
        pred_list = [p for p in pred_list if p.energy > 0]

        # Food respawn (stochastic)
        for f in food_list:
            if random.random() < 0.02:
                f.respawn()

        stats.record(len(prey_list), len(pred_list))

    # --- Render ---
    screen.fill(BG)
    pygame.draw.rect(screen, (15, 15, 30), ARENA_RECT)
    pygame.draw.rect(screen, GRID_COLOR, ARENA_RECT, 1)

    # Food
    for f in food_list:
        pygame.draw.circle(screen, FOOD_COLOR, (int(f.x), int(f.y)), int(f.radius))

    # Prey
    for p in prey_list:
        r = max(3, int(p.detection_radius * 0.15))
        pygame.draw.circle(screen, PREY_COLOR, (int(p.x), int(p.y)), r)
        if abs(p.vx) > 0.1 or abs(p.vy) > 0.1:
            pygame.draw.line(screen, (100, 255, 150), (int(p.x), int(p.y)),
                             (int(p.x + p.vx * 6), int(p.y + p.vy * 6)), 1)

    # Predators
    for pr in pred_list:
        pygame.draw.circle(screen, PRED_COLOR, (int(pr.x), int(pr.y)), 6)
        if abs(pr.vx) > 0.1 or abs(pr.vy) > 0.1:
            pygame.draw.line(screen, (255, 150, 150), (int(pr.x), int(pr.y)),
                             (int(pr.x + pr.vx * 8), int(pr.y + pr.vy * 8)), 2)

    # Panel
    panel_rect = pygame.Rect(ARENA_W + 10, 10, WIDTH - ARENA_W - 20, HEIGHT - 20)
    pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=6)

    draw_text(screen, f"Gen: {gen}", (ARENA_W + 20, 20), WHITE)
    draw_text(screen, f"Prey: {len(prey_list)}", (ARENA_W + 20, 40), PREY_COLOR)
    draw_text(screen, f"Predators: {len(pred_list)}", (ARENA_W + 20, 60), PRED_COLOR)
    draw_text(screen, f"Speed: {speed}x [SPACE]", (ARENA_W + 20, HEIGHT - 70), WHITE)
    draw_text(screen, "[R] Reset", (ARENA_W + 20, HEIGHT - 50), WHITE)
    draw_text(screen, "[Click] Add food", (ARENA_W + 20, HEIGHT - 30), WHITE)

    # Graph
    if show_graph:
        graph_rect = pygame.Rect(GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H)
        pygame.draw.rect(screen, (10, 10, 25), graph_rect, border_radius=4)
        pygame.draw.rect(screen, GRID_COLOR, graph_rect, 1)

        if len(stats.prey_history) > 1:
            max_val = max(max(stats.prey_history), max(stats.pred_history), 1)
            # Prey line
            pts = [(GRAPH_X + int(i / stats.max_history * GRAPH_W),
                    GRAPH_Y + GRAPH_H - int(cnt / max_val * (GRAPH_H - 20)))
                   for i, cnt in enumerate(stats.prey_history)]
            if pts:
                pygame.draw.lines(screen, PREY_COLOR, False, pts, 2)
            # Predator line
            pts = [(GRAPH_X + int(i / stats.max_history * GRAPH_W),
                    GRAPH_Y + GRAPH_H - int(cnt / max_val * (GRAPH_H - 20)))
                   for i, cnt in enumerate(stats.pred_history)]
            if pts:
                pygame.draw.lines(screen, PRED_COLOR, False, pts, 2)

    pygame.display.flip()

pygame.quit()
